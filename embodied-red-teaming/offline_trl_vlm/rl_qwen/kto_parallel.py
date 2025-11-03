# -*- coding: utf-8 -*-
"""
KTO (offline, multimodal, single-preference) for Qwen3-VL
- 输入：JSONL，列：image(本地路径或URL)、prompt、response、label(0/1)、task(可选)
- 模型：Qwen/Qwen3-VL-4B-Instruct，LoRA 省显存
- 训练：离线 KTO（无参考模型），对 response 段做对数几率损失；可选语义加权
"""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import warnings
warnings.filterwarnings("ignore")

from dataclasses import dataclass
from typing import List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import set_seed

from datasets import load_dataset, Features, Value
from datasets import Image as HFImage
from transformers import AutoProcessor
from transformers import Qwen3VLForConditionalGeneration  # <<< 关键

from peft import LoraConfig, get_peft_model
from semantic_keep import BiNLIScorer  # 你自己的模块
from datetime import datetime
from tqdm import tqdm

# ============ 可改配置 ============
MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
INPUT_JSON_PATH = "/home/baoshuntong/code/saftyEmbodyAI/embodied-red-teaming/input_json_data/libero_spatial_kto_new.json"
now = datetime.now()
OUTPUT_PATH = f"outputs/{now.strftime('%Y-%m-%d_%H-%M-%S')}/verl_kto_qwen3vl"
BATCH_SIZE = 2

@dataclass
class TrainCfg:
    model_id: str = MODEL_ID
    data_path: str = INPUT_JSON_PATH
    output_dir: str = OUTPUT_PATH
    beta: float = 0.1                      # KTO 超参
    lr: float = 5e-6                       # KTO 对 LR 较敏感（建议 5e-7 ~ 5e-6）
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
    # 类别不平衡时可调权重
    desirable_weight: float = 1.0          # 对 label=1
    undesirable_weight: float = 1.0        # 对 label=0
    positive_label_value: int = 1          # “正样本”标签值（默认1）

cfg = TrainCfg()
if cfg.target_modules is None:
    cfg.target_modules = ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]

# ============ 数据集 ============
def build_kto_dataset(path: str):
    """
    读取单样本KTO数据：image/prompt/response/label[/task]
    - 自动兼容不含 task 的数据（后续训练时退化为 task=prompt）
    - 将 PIL 放到 ex["images"] 以适配 VLM processor（每条样本1张图 -> [image]）
    """
    ds = load_dataset("json", data_files=path)["train"]
    # 确保 image 列按 HF Image 解码（可处理本地路径/URL）
    if "image" not in ds.column_names:
        raise ValueError("Input JSONL 必须包含字段 `image`。")
    ds = ds.cast_column("image", HFImage(decode=True))

    def to_images(ex):
        img = ex.get("image", None)
        ex["images"] = [img] if img is not None else []
        return ex
    keep_cols = [c for c in ["image"] if c in ds.column_names]
    ds = ds.map(to_images, remove_columns=keep_cols)
    return ds

def collate_keep_pil(batch):
    """保持 PIL 图像不被 DataLoader 拼接；返回字段为列表。"""
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

# ============ 计算“仅 response 段”的 avg log p ============
@torch.no_grad()
def _prompt_only_token_len(processor, images_list, prompts, device):
    """得到每个样本 user-only（带 generation prompt）的 token 长度，用于 label mask。"""
    convs = [build_msgs(images_list[i], prompts[i], None) for i in range(len(prompts))]
    enc = processor.apply_chat_template(
        convs, tokenize=True, add_generation_prompt=True,
        return_tensors="pt", return_dict=True, padding=True
    )
    # 某些版本会返回 token_type_ids；按官方示例删除
    enc.pop("token_type_ids", None)
    attn = enc["attention_mask"].to(device)
    return attn.sum(dim=1).tolist()

def compute_batch_logp(
    model, processor,
    images_list,   # List[List[PIL.Image]]
    prompts,       # List[str]
    responses,     # List[str]
    requires_grad: bool = False,
):
    """
    对每个样本计算“仅 response 段”的平均 token log 概率（形状 [B]）
    """
    device = next(model.parameters()).device
    model_dtype = getattr(model, "dtype", torch.bfloat16)
    B = len(prompts)
    assert B == len(images_list) == len(responses)

    # 构造  B 条（user+assistant）的完整对话
    convs = [build_msgs(images_list[i], prompts[i], responses[i]) for i in range(B)]

    enc = processor.apply_chat_template(
        convs, tokenize=True, add_generation_prompt=False,
        return_tensors="pt", return_dict=True, padding=True
    )
    enc.pop("token_type_ids", None)

    # 设备/精度
    for k, v in list(enc.items()):
        if isinstance(v, torch.Tensor):
            enc[k] = v.to(device=device, dtype=(model_dtype if v.is_floating_point() else None))

    # user-only（带 assistant 起始提示）长度
    prompt_len = _prompt_only_token_len(processor, images_list, prompts, device)

    input_ids = enc["input_ids"]            # [B, L]
    labels = input_ids.clone()
    for i in range(B):
        cut = min(int(prompt_len[i]), labels.size(1))
        labels[i, :cut] = -100              # 只监督 assistant 段

    ctx = torch.enable_grad if requires_grad else torch.no_grad
    with ctx():
        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": enc.get("attention_mask", None),
            "use_cache": False,
        }
        if "pixel_values" in enc:
            model_inputs["pixel_values"] = enc["pixel_values"]
        if "video_pixel_values" in enc:
            model_inputs["video_pixel_values"] = enc["video_pixel_values"]

        # 可按需开启 Flash-Attn2：
        # model.config.attn_implementation = "flash_attention_2"
        out = model(**model_inputs)         # logits: [B, L, V]
        logits = out.logits.float()

    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    token_nll = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="none",
    ).view(shift_labels.size(0), -1)        # [B, L-1]

    valid = (shift_labels != -100)
    denom = valid.sum(dim=1).clamp(min=1)
    nll_per_sample = (token_nll * valid).sum(dim=1) / denom
    avg_logp = -nll_per_sample              # [B]
    return avg_logp

# ============ KTO 损失 ============
def kto_loss(beta, logp, labels,
             desirable_weight=1.0, undesirable_weight=1.0,
             positive_label_value=1,
             sample_weights: torch.Tensor | None = None):
    """
    无参考的 KTO：
      - y=1（正样本）： L_pos = softplus(-beta * logp)
      - y=0（负样本）： L_neg = softplus( beta * logp)
    支持逐样本权重 sample_weights。
    """
    y = (labels == positive_label_value).float()            # [B]
    pos_loss = F.softplus(-beta * logp)                     # [B]
    neg_loss = F.softplus( beta * logp)
    loss_el = desirable_weight * y * pos_loss + undesirable_weight * (1 - y) * neg_loss  # [B]
    if sample_weights is not None:
        loss_el = loss_el * sample_weights
    return loss_el.mean()

def sem_weight(s: torch.Tensor, tau: float = 0.85, k: float = 12.0) -> torch.Tensor:
    return torch.sigmoid(k * (s - tau)).clamp(0.0, 1.0)

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

    print(f"Loading model: {cfg.model_id}")
    processor = AutoProcessor.from_pretrained(cfg.model_id)

    # Actor（LoRA）
    actor = Qwen3VLForConditionalGeneration.from_pretrained(
        cfg.model_id,
        torch_dtype=(torch.bfloat16 if cfg.use_bf16 else "auto"),
        # attn_implementation="flash_attention_2",  # 安装了 flash-attn 可启用
    )
    if hasattr(actor, "gradient_checkpointing_enable"):
        actor.gradient_checkpointing_enable()

    lora = LoraConfig(
        r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout, bias="none",
        target_modules=cfg.target_modules, task_type="CAUSAL_LM"
    )
    actor = get_peft_model(actor, lora)

    # 数据
    ds = build_kto_dataset(cfg.data_path)
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

    # 再次确保开启 ckpt
    if hasattr(actor, "gradient_checkpointing_enable"):
        actor.gradient_checkpointing_enable()

    device = accelerator.device
    nli = BiNLIScorer(device=device)  # 你自定义的语义打分器

    # 超参（语义加权/约束）
    tau_sem = 0.85
    k_sem   = 12.0
    # 方案A：语义分数作为样本权重
    use_sem_A = True

    kto_loss_list = []
    global_step = 0
    for epoch in tqdm(range(cfg.epochs)):
        for step, batch in tqdm(enumerate(dl)):
            images_list = batch["images"]             # List[List[PIL.Image]]
            prompts     = batch["prompt"]             # List[str]
            responses   = batch["response"]           # List[str]
            labels      = torch.tensor(batch["label"], dtype=torch.long, device=accelerator.device)  # [B]

            with torch.no_grad():
                # 兼容无 task 的数据
                tasks = batch["task"] if "task" in batch else prompts
                s_text = nli.score(tasks, responses).to(device)  # [B]

            if use_sem_A:
                w = sem_weight(s_text, tau=tau_sem, k=k_sem)
                # 对负样本给个下限，避免完全失声
                w = torch.where(labels == cfg.positive_label_value, w, torch.clamp(w, min=0.5))
            else:
                w = None

            with accelerator.accumulate(actor):
                # 计算“仅 response 段”的平均 logp
                logp = compute_batch_logp(
                    actor, processor, images_list, prompts, responses,
                    requires_grad=True,
                )  # [B]

                loss = kto_loss(
                    beta=cfg.beta, logp=logp, labels=labels,
                    desirable_weight=cfg.desirable_weight,
                    undesirable_weight=cfg.undesirable_weight,
                    positive_label_value=cfg.positive_label_value,
                    sample_weights=w
                )

                accelerator.backward(loss)
                kto_loss_list.append(loss.item())

                if accelerator.is_main_process:
                    print(f"Epoch {epoch} Step {global_step} KTO Loss: {loss.item():.4f}")

                opt.step()
                opt.zero_grad()

            if accelerator.sync_gradients and (global_step % 10 == 0):
                loss_avg = accelerator.gather(loss.detach()).mean()
                accelerator.print(f"epoch {epoch} step {global_step} kto_loss={loss_avg.item():.4f}")

            global_step += 1

    accelerator.wait_for_everyone()
    if is_main:
        unwrapped = accelerator.unwrap_model(actor)   # 取出原模型（PEFT 包）
        unwrapped.save_pretrained(os.path.join(cfg.output_dir, "lora_kto"))
        print("✅ KTO training done. LoRA adapter saved.")

        with open(os.path.join(cfg.output_dir, "kto_loss_log.txt"), "w") as f:
            f.write(str(kto_loss_list))

if __name__ == "__main__":
    main()
