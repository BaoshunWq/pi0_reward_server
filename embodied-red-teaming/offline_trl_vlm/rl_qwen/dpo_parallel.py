# -*- coding: utf-8 -*-
"""
DPO (offline, multimodal) with VERL-style single-controller
- 输入：JSONL，列包含：image(本地路径或URL)、prompt、chosen、rejected
- 模型：Qwen/Qwen3-VL-4B-Instruct，LoRA 省显存
- 训练：离线 DPO，参考策略来自同一底模的冻结副本
"""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import warnings
warnings.filterwarnings("ignore")

import json
from dataclasses import dataclass
from typing import List

from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import set_seed
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets import load_dataset, Features, Value
from datasets import Image as HFImage

from datetime import datetime

from transformers import AutoProcessor
from transformers import Qwen3VLForConditionalGeneration  # <<< 关键改动
from peft import LoraConfig, get_peft_model

# ============ 你自己的路径/配置 ============
MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
INPUT_JSON_PATH = "/home/baoshuntong/code/saftyEmbodyAI/embodied-red-teaming/input_json_data/libero_spatial_dpo_new.json"
NOW_TIME_STR = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
OUTPUT_PATH = f"outputs/verl_dpo_qwen3vl/{NOW_TIME_STR}"
BATCH_SIZE = 1

# ============ 配置 ============
@dataclass
class TrainCfg:
    model_id: str = MODEL_ID
    data_path: str = INPUT_JSON_PATH
    output_dir: str = OUTPUT_PATH
    beta: float = 0.05
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
    target_modules: List[str] = None  # LoRA 作用到文本侧投影/MLP

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

    # 统一成 images: List[PIL.Image] 以兼容多图
    def to_images(ex):
        img = ex["image"]
        ex["images"] = [img] if img is not None else []
        return ex
    ds = ds.map(to_images, remove_columns=["image"])
    return ds

def collate_keep_pil(batch):
    out = {}
    keys = batch[0].keys()
    for k in keys:
        out[k] = [ex[k] for ex in batch]
    return out

# ============ 构造 Qwen3-VL chats ============
def build_msgs(images, user_text, assistant_text=None):
    """
    images: List[PIL.Image.Image]  (可多张)
    """
    contents = []
    if images:
        for im in images:
            contents.append({"type": "image", "image": im})
    contents.append({"type": "text", "text": (user_text or "").strip()})
    msgs = [{"role": "user", "content": contents}]
    if assistant_text is not None:
        msgs.append({"role": "assistant", "content": [{"type": "text", "text": (assistant_text or '').strip()}]})
    return msgs

# ============ 计算成对 log p ============
@torch.no_grad()
def _prompt_only_token_len(processor, images_list, prompts, device, dtype):
    """得到每个样本的 user+assistant 起始标记长度（用于 label mask）"""
    convs = [build_msgs(images_list[i], prompts[i], None) for i in range(len(prompts))]
    enc = processor.apply_chat_template(
        convs, tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True, padding=True
    )
    enc.pop("token_type_ids", None)  # Qwen3-VL 文档建议移除
    # 直接用 attention_mask 求长度
    attn = enc["attention_mask"].to(device)
    return attn.sum(dim=1).tolist()

def compute_batch_pair_logps(
    model, processor,
    images_list, prompts, chosens, rejecteds,
    requires_grad: bool = False,
):
    device = next(model.parameters()).device
    model_dtype = getattr(model, "dtype", torch.bfloat16)

    B = len(prompts)
    assert B == len(images_list) == len(chosens) == len(rejecteds)

    # 2B 条样本：chosen 在前，rejected 在后
    convs = []
    for i in range(B):
        convs.append(build_msgs(images_list[i], prompts[i], chosens[i]))
    for i in range(B):
        convs.append(build_msgs(images_list[i], prompts[i], rejecteds[i]))

    enc = processor.apply_chat_template(
        convs, tokenize=True, add_generation_prompt=False, return_tensors="pt", return_dict=True, padding=True
    )
    enc.pop("token_type_ids", None)

    for k, v in list(enc.items()):
        if isinstance(v, torch.Tensor):
            enc[k] = v.to(device=device, dtype=(model_dtype if v.is_floating_point() else None))

    # prompt-only 长度（包含 assistant 起始标记）
    prompt_len = _prompt_only_token_len(processor, images_list, prompts, device, model_dtype)

    input_ids = enc["input_ids"]                   # [2B, L]
    labels = input_ids.clone()

    for i in range(B):
        cut = min(int(prompt_len[i]), labels.size(1))
        labels[i, :cut] = -100          # mask chosen 的用户侧
        labels[B+i, :cut] = -100        # mask rejected 的用户侧

    # 前向
    ctx = torch.enable_grad if requires_grad else torch.no_grad
    with ctx():
        # 只取需要的键，避免意外参数
        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": enc.get("attention_mask"),
            "use_cache": False,
            "image_grid_thw": enc.get("image_grid_thw"),
            "pixel_values": enc.get("pixel_values"),
        }


        out = model(**model_inputs)      # logits: [2B, L, V]
        logits = out.logits.float()

    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    token_nll = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="none",
    ).view(shift_labels.size(0), -1)     # [2B, L-1]

    valid = (shift_labels != -100)
    denom = valid.sum(dim=1).clamp(min=1)
    nll_per_sample = (token_nll * valid).sum(dim=1) / denom
    avg_logp = -nll_per_sample

    logp_c = avg_logp[:B]
    logp_r = avg_logp[B:]
    if not requires_grad:
        logp_c, logp_r = logp_c.detach(), logp_r.detach()
    return logp_c, logp_r

# ============ DPO 损失 ============
def dpo_loss(beta, logp_chosen, logp_rejected, ref_logp_chosen, ref_logp_rejected):
    z = beta * ((logp_chosen - logp_rejected) - (ref_logp_chosen - ref_logp_rejected))
    return -torch.nn.functional.logsigmoid(z).mean()

# ============ 主训练 ============
def main():
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

    print(f"Loading model: {cfg.model_id}")
    processor = AutoProcessor.from_pretrained(cfg.model_id)

    # Actor（LoRA）
    actor = Qwen3VLForConditionalGeneration.from_pretrained(  # <<< 关键改动
        cfg.model_id,
        torch_dtype=torch.bfloat16 if cfg.use_bf16 else "auto",
        # attn_implementation="flash_attention_2",  # 如安装了 FA2，可打开
    )
    if hasattr(actor, "gradient_checkpointing_enable"):
        actor.gradient_checkpointing_enable()
    lora = LoraConfig(
        r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout, bias="none",
        target_modules=cfg.target_modules, task_type="CAUSAL_LM"
    )
    actor = get_peft_model(actor, lora)

    # Reference（冻结同底模）
    ref = Qwen3VLForConditionalGeneration.from_pretrained(
        cfg.model_id,
        torch_dtype=torch.bfloat16 if cfg.use_bf16 else "auto",
    )
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    ref = ref.to(accelerator.device)

    ds = build_dpo_dataset(cfg.data_path)
    dl = DataLoader(
        ds, batch_size=cfg.bsz, shuffle=True,
        collate_fn=collate_keep_pil, num_workers=4, pin_memory=True,
    )

    opt = torch.optim.AdamW(actor.parameters(), lr=cfg.lr)

    actor, opt, dl = accelerator.prepare(actor, opt, dl)

    if hasattr(actor, "gradient_checkpointing_enable"):
        actor.gradient_checkpointing_enable()

    dpo_loss_log = []
    global_step = 0
    for epoch in range(cfg.epochs):
        accelerator.print(f"=== Epoch {epoch} ===")
        for step, batch in enumerate(dl):
            images_list = batch["images"]     # List[List[PIL.Image]]
            prompts     = batch["prompt"]     # List[str]
            chosens     = batch["chosen"]     # List[str]
            rejecteds   = batch["rejected"]   # List[str]

            with accelerator.accumulate(actor):
                # actor
                logp_c, logp_r = compute_batch_pair_logps(
                    actor, processor, images_list, prompts, chosens, rejecteds, requires_grad=True,
                )
                # ref
                with torch.no_grad():
                    ref_logp_c, ref_logp_r = compute_batch_pair_logps(
                        ref, processor, images_list, prompts, chosens, rejecteds, requires_grad=False,
                    )
                loss = dpo_loss(cfg.beta, logp_c, logp_r, ref_logp_c, ref_logp_r)
                accelerator.backward(loss)
                dpo_loss_log.append(loss.item())

                if accelerator.is_main_process:
                    print(f"Epoch {epoch} Step {global_step} DPO Loss: {loss.item():.4f}")

                opt.step()
                opt.zero_grad()

            if accelerator.sync_gradients and (global_step % 10 == 0):
                loss_avg = accelerator.gather(loss.detach()).mean()
                accelerator.print(f"epoch {epoch} step {global_step} dpo_loss={loss_avg.item():.4f}")
            global_step += 1

    accelerator.wait_for_everyone()
    if is_main:
        unwrapped = accelerator.unwrap_model(actor)
        unwrapped.save_pretrained(os.path.join(cfg.output_dir, "lora_dpo"))
        with open(os.path.join(cfg.output_dir, "dpo_loss_log.txt"), "w") as f:
            f.write(str(dpo_loss_log))
        print("✅ DPO training done. LoRA adapter saved.")

if __name__ == "__main__":
    main()
