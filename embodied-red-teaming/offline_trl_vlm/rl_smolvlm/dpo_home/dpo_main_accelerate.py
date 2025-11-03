# -*- coding: utf-8 -*-
"""
DPO (offline, multimodal) with VERL-style single-controller
- 输入：JSONL，列包含：image(本地路径或URL)、prompt、chosen、rejected
- 模型：SmolVLM / Qwen2-VL 等，LoRA 省显存
- 训练：离线 DPO，参考政策来自同一底模的冻结副本
"""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"   # 推荐用 false

import warnings
warnings.filterwarnings("ignore")

import os
import json
import math
from dataclasses import dataclass
from typing import List, Dict, Any

from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import set_seed
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

from datasets import load_dataset, Features, Value
from datasets import Image as HFImage
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor

from transformers import AutoModelForImageTextToText as VLMClass

from datetime import datetime
import torch

# 获取当前时间
now = datetime.now()
NOW_TIME_STR = now.strftime("%Y-%m-%d_%H-%M-%S")

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from offline_trl_vlm.utils.utils import generate_text_from_sample

MODEL_ID = "HuggingFaceTB/SmolVLM-500M-Instruct"

INPUT_JSON_PATH = "/home/baoshuntong/code/saftyEmbodyAI/embodied-red-teaming/input_json_data/libero_spatial_dpo_new.json"

OUTPUT_PATH = f"outputs/{NOW_TIME_STR}/verl_dpo_vlm"

BATCH_SIZE = 1



# ============ 配置 ============
@dataclass
class TrainCfg:
    model_id: str = MODEL_ID  # 或 "Qwen/Qwen2.5-VL-7B-Instruct"
    data_path: str = INPUT_JSON_PATH
    output_dir: str = OUTPUT_PATH
    beta: float = 0.05                      # DPO 超参
    lr: float = 5e-5
    epochs: int = 10
    bsz: int = BATCH_SIZE
    grad_accum: int = 8
    max_prompt_len: int = 512
    max_new_tokens: int = 128
    use_bf16: bool = True
    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: List[str] = None

cfg = TrainCfg()
if cfg.target_modules is None:
    cfg.target_modules = ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]

# ============ 数据集 ============
def build_dpo_dataset(path: str):
    features = Features({
        "image": HFImage(decode=True),
        "prompt": Value("string"),
        "chosen": Value("string"),
        "rejected": Value("string"),
    })
    ds = load_dataset("json", data_files=path, features=features)["train"]

    # 转成 images(list[PIL]) 以适配 VLM processor
    def to_images(ex):
        ex["images"] = [ex["image"]]
        return ex
    ds = ds.map(to_images, remove_columns=["image"])
    return ds


def collate_keep_pil(batch):
    # batch: List[Dict]
    # 返回：Dict[str, List[Any]]，保持 PIL 不做任何拼接
    out = {}
    keys = batch[0].keys()
    for k in keys:
        out[k] = [ex[k] for ex in batch]
    return out


def compute_batch_pair_logps(
    model, processor,
    images_list,   # List[List[PIL.Image]]（每个样本的图像列表；你当前是每样本1张图 -> [image]）
    prompts,       # List[str], 长度 B
    chosens,       # List[str], 长度 B
    rejecteds,     # List[str], 长度 B
    requires_grad: bool = False,
):
    device = next(model.parameters()).device
    model_dtype = getattr(model, "dtype", torch.bfloat16)
    B = len(prompts)
    assert B == len(images_list) == len(chosens) == len(rejecteds)

    # 拼 2B 条样本（前 B: chosen；后 B: rejected）
    texts, images_all = [], []
    for i in range(B):
        texts.append((prompts[i] or "").strip() + "\n" + (chosens[i] or "").strip())
        images_all.append(images_list[i])
    for i in range(B):
        texts.append((prompts[i] or "").strip() + "\n" + (rejecteds[i] or "").strip())
        images_all.append(images_list[i])

    enc = processor(images=images_all, text=texts, return_tensors="pt",
                    padding=True, truncation=False)
    for k, v in enc.items():
        if isinstance(v, torch.Tensor):
            enc[k] = v.to(device=device, dtype=model_dtype if v.is_floating_point() else None)

    # 只监督 response：用 attention_mask 计算真实 prompt 长度
    with torch.no_grad():
        p_enc = processor(images=images_list, text=prompts, return_tensors="pt",
                          padding=True, truncation=False)
        prompt_len = p_enc["attention_mask"].sum(dim=1).tolist()  # List[int] 长度 B

    input_ids = enc["input_ids"]            # [2B, L]
    labels = input_ids.clone()
    for i in range(B):
        cut = min(int(prompt_len[i]), labels.size(1))
        labels[i, :cut] = -100           # chosen 的 prompt 掩掉
        labels[B+i, :cut] = -100         # rejected 的 prompt 也掩掉

    ctx = torch.enable_grad if requires_grad else torch.no_grad
    with ctx():
        out = model(input_ids=input_ids,
                    pixel_values=enc.get("pixel_values", None),
                    use_cache=False)
        logits = out.logits              # [2B, L, V]

    shift_logits = logits[..., :-1, :].contiguous().float()
    shift_labels = labels[..., 1:].contiguous()

    token_nll = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="none",
    ).view(shift_labels.size(0), -1)     # [2B, L-1]

    valid = (shift_labels != -100)
    denom = valid.sum(dim=1).clamp(min=1)         # [2B]
    nll_per_sample = (token_nll * valid).sum(dim=1) / denom
    avg_logp = -nll_per_sample                    # [2B]

    logp_c = avg_logp[:B]
    logp_r = avg_logp[B:]
    if not requires_grad:
        logp_c, logp_r = logp_c.detach(), logp_r.detach()
    return logp_c, logp_r

# ============ DPO 损失 ============ 
def dpo_loss(beta, logp_chosen, logp_rejected, ref_logp_chosen, ref_logp_rejected):
    # 参见 DPO 论文/TRL 实现：-log σ(β * [(logp_c - logp_r) - (logp_c_ref - logp_r_ref)])
    z = beta * ((logp_chosen - logp_rejected) - (ref_logp_chosen - ref_logp_rejected))
    return -torch.nn.functional.logsigmoid(z).mean()

# ============ 主训练 ============

def main():

    # ===== Accelerator 初始化 =====
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.grad_accum,
        mixed_precision=("bf16" if cfg.use_bf16 else "no"),
        kwargs_handlers=[ddp_kwargs],
    )
    is_main = accelerator.is_main_process
    set_seed(42)

    if is_main:
        print(f"[Accelerate] device={accelerator.device}, mp={accelerator.mixed_precision}")

    os.makedirs(cfg.output_dir, exist_ok=True)

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    print(f"Loading model: {cfg.model_id}")
    processor = AutoProcessor.from_pretrained(cfg.model_id)

    # Actor（可加 LoRA）
    actor = VLMClass.from_pretrained(cfg.model_id, torch_dtype=torch.bfloat16 if cfg.use_bf16 else "auto")
    if hasattr(actor, "gradient_checkpointing_enable"):
        actor.gradient_checkpointing_enable()
    lora = LoraConfig(
        r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout, bias="none",
        target_modules=cfg.target_modules, task_type="CAUSAL_LM"
    )
    actor = get_peft_model(actor, lora)


    # Reference（冻结）
    ref = VLMClass.from_pretrained(cfg.model_id, torch_dtype=torch.bfloat16 if cfg.use_bf16 else "auto")
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    ref = ref.to(accelerator.device)

    ds = build_dpo_dataset(cfg.data_path,)
    dl = DataLoader(
        ds,
        batch_size=cfg.bsz,
        shuffle=True,                # Accelerate 会自动替换为 DistributedSampler
        collate_fn=collate_keep_pil,
        num_workers=4,
        pin_memory=True,
    )

    opt = torch.optim.AdamW(actor.parameters(), lr=cfg.lr)

        # ===== 交给 Accelerate 并行/混精/通信 =====
    actor, opt, dl = accelerator.prepare(actor, opt, dl)

    # 构建 actor/ref 后
    if hasattr(actor, "gradient_checkpointing_enable"):
        actor.gradient_checkpointing_enable()

    from tqdm import tqdm

    dpo_loss_list = []

    global_step = 0
    for epoch in tqdm(range(cfg.epochs)):

        for step, batch in tqdm(enumerate(dl)):
            # B=1（多卡/大B请自行扩展累积）
            # 新：整批进入（长度 = B = cfg.bsz）
            images_list = batch["images"]     # List[List[PIL.Image]]
            prompts     = batch["prompt"]     # List[str]
            chosens     = batch["chosen"]     # List[str]
            rejecteds   = batch["rejected"]   # List[str]


            # 用 accumulate 做梯度累积（只在需要时同步 allreduce）
            with accelerator.accumulate(actor):

                # actor：带梯度，返回 [B]
                logp_c, logp_r = compute_batch_pair_logps(
                    actor, processor, images_list, prompts, chosens, rejecteds,
                    requires_grad=True,
                )

                # ref：无梯度，返回 [B]
                with torch.no_grad():
                    ref_logp_c, ref_logp_r = compute_batch_pair_logps(
                        ref, processor, images_list, prompts, chosens, rejecteds,
                        requires_grad=False,
                    )

                loss = dpo_loss(cfg.beta, logp_c, logp_r, ref_logp_c, ref_logp_r)  # .mean() 已在内部做
                accelerator.backward(loss)
                dpo_loss_list.append(loss.item())

                print(f"Epoch {epoch} Step {global_step} DPO Loss: {loss.item():.4f}")


                opt.step()
                opt.zero_grad()

            if accelerator.sync_gradients and (global_step % 10 == 0):
                loss_avg = accelerator.gather(loss.detach()).mean()
                accelerator.print(f"epoch {epoch} step {global_step} dpo_loss={loss_avg.item():.4f}")


            global_step += 1

    accelerator.wait_for_everyone()
    if is_main:
        unwrapped = accelerator.unwrap_model(actor)   # 取出原模型（PEFT 包）
        unwrapped.save_pretrained(os.path.join(cfg.output_dir, "lora_dpo"))
        # actor.save_pretrained(os.path.join(cfg.output_dir, "lora_dpo"))
        print("✅ DPO training done. LoRA adapter saved.")
    
        with open(os.path.join(cfg.output_dir, "dpo_loss_log.txt"), "w") as f:
            f.write(str(dpo_loss_list))

    # gene_instru = generate_text_from_sample(actor, processor, ds[0], max_new_tokens=128, device=accelerator.device)

    # print("Sample generation after DPO:")
    # print(gene_instru)


if __name__ == "__main__":
    main()
