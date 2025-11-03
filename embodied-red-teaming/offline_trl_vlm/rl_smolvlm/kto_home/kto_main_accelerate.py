# -*- coding: utf-8 -*-
"""
KTO (offline, multimodal, single-preference)
- 输入：JSONL，列包含：image(本地路径或URL)、prompt、response、label(0/1)、task(可选)
- 模型：SmolVLM / Qwen2-VL 等，LoRA 省显存
- 训练：离线 KTO（无需参考模型），按标签对 response 做对数几率损失
"""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"   # 推荐用 false

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
from transformers import AutoModelForImageTextToText as VLMClass

from peft import LoraConfig, get_peft_model
from semantic_keep import BiNLIScorer
from datetime import datetime
from tqdm import tqdm

# ============ 可改配置 ============
MODEL_ID = "HuggingFaceTB/SmolVLM-500M-Instruct"
INPUT_JSON_PATH = "/home/baoshuntong/code/saftyEmbodyAI/embodied-red-teaming/input_json_data/libero_spatial_kto_new.json"
# ↑ 你的 KTO jsonl 路径（字段需为 image/prompt/response/label[/task]）

now = datetime.now()
OUTPUT_PATH = f"outputs/{now.strftime('%Y-%m-%d_%H-%M-%S')}/verl_kto_vlm"
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
    positive_label_value: int = 1          # 你的“正样本”标签值（默认1）

cfg = TrainCfg()
if cfg.target_modules is None:
    cfg.target_modules = ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]

# ============ 数据集 ============
def build_kto_dataset(path: str):
    """
    读取单样本KTO数据：image/prompt/response/label[/task]
    将 PIL 放到 ex["images"] 列中以适配 VLM processor（每条样本1张图 => [image]）
    """
    features = Features({
        "image": HFImage(decode=True),
        "prompt": Value("string"),
        "response": Value("string"),
        "label": Value("int64"),
        "task": Value("string"),  # 若你的数据没有 task 字段，可删除此行
    })
    ds = load_dataset("json", data_files=path, features=features)["train"]

    def to_images(ex):
        ex["images"] = [ex["image"]]
        return ex

    ds = ds.map(to_images, remove_columns=["image"])
    return ds

def collate_keep_pil(batch):
    """保持 PIL 图像不被 DataLoader 拼接；返回字段为列表。"""
    out = {}
    keys = batch[0].keys()
    for k in keys:
        out[k] = [ex[k] for ex in batch]
    return out

# ============ 计算单样本 logp ============
def compute_batch_logp(
    model, processor,
    images_list,   # List[List[PIL.Image]]（每样本1张图 -> [image]）
    prompts,       # List[str], 长度 B
    responses,     # List[str], 长度 B
    requires_grad: bool = False,
):
    """
    对每个样本计算“仅 response 段”的平均 token log 概率（avg_logp），形状 [B]
    """
    device = next(model.parameters()).device
    model_dtype = getattr(model, "dtype", torch.bfloat16)
    B = len(prompts)
    assert B == len(images_list) == len(responses)

    # 拼接 prompt + response 文本
    texts = [ (prompts[i] or "").strip() + "\n" + (responses[i] or "").strip()
              for i in range(B) ]

    enc = processor(images=images_list, text=texts, return_tensors="pt",
                    padding=True, truncation=False)
    for k, v in enc.items():
        if isinstance(v, torch.Tensor):
            enc[k] = v.to(device=device, dtype=(model_dtype if v.is_floating_point() else None))

    # 计算每个样本的 prompt 长度（用 attention_mask）
    with torch.no_grad():
        p_enc = processor(images=images_list, text=prompts, return_tensors="pt",
                          padding=True, truncation=False)
        prompt_len = p_enc["attention_mask"].sum(dim=1).tolist()  # List[int]，长度 B
        p_enc = None

    input_ids = enc["input_ids"]            # [B, L]
    labels = input_ids.clone()
    for i in range(B):
        cut = min(int(prompt_len[i]), labels.size(1))
        labels[i, :cut] = -100              # 只监督 response 段

    ctx = torch.enable_grad if requires_grad else torch.no_grad
    with ctx():
        out = model(
            input_ids=input_ids,
            pixel_values=enc.get("pixel_values", None),
            use_cache=False
        )
        logits = out.logits                  # [B, L, V]

    shift_logits = logits[..., :-1, :].contiguous().float()
    shift_labels = labels[..., 1:].contiguous()

    token_nll = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="none",
    ).view(shift_labels.size(0), -1)        # [B, L-1]

    valid = (shift_labels != -100)
    denom = valid.sum(dim=1).clamp(min=1)   # [B]
    nll_per_sample = (token_nll * valid).sum(dim=1) / denom
    avg_logp = -nll_per_sample              # [B]
    return avg_logp

# ============ KTO 损失 ============
# def kto_loss(beta, logp, labels, desirable_weight=1.0, undesirable_weight=1.0, positive_label_value=1):
#     """
#     无参考的 KTO：
#       - 对正样本 y=1： L_pos = -log σ(beta * logp) = softplus(-beta * logp)
#       - 对负样本 y=0： L_neg = -log σ(-beta * logp) = softplus( beta * logp)
#     labels: [B]，元素为 {0,1}（或与 positive_label_value 一致）
#     """
#     y = (labels == positive_label_value).float()  # 1=正样本
#     pos_loss = F.softplus(-beta * logp)           # 形状 [B]
#     neg_loss = F.softplus( beta * logp)

#     # 分别加权再求平均（避免类别不平衡）
#     if y.sum() > 0:
#         pos_mean = (pos_loss * y).sum() / y.sum()
#     else:
#         pos_mean = torch.tensor(0.0, device=logp.device)

#     if (1 - y).sum() > 0:
#         neg_mean = (neg_loss * (1 - y)).sum() / (1 - y).sum()
#     else:
#         neg_mean = torch.tensor(0.0, device=logp.device)

#     return desirable_weight * pos_mean + undesirable_weight * neg_mean

def kto_loss(beta, logp, labels,
             desirable_weight=1.0, undesirable_weight=1.0,
             positive_label_value=1,
             sample_weights: torch.Tensor | None = None):
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

    # Actor（可加 LoRA）
    actor = VLMClass.from_pretrained(
        cfg.model_id,
        torch_dtype=(torch.bfloat16 if cfg.use_bf16 else "auto")
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
    nli = BiNLIScorer(device=device)

    # 超参
    tau_sem = 0.85       # 语义阈值：越高越严格
    k_sem   = 12.0       # 阶跃陡峭程度
    lambda_sem = 0.5     # 方案B：语义约束项的系数（0.2~1.0 可扫）
    use_sem_A = True     # 方案A 开关（加权KTO）
    # use_sem_B = True     # 方案B 开关（约束项）

    kto_loss_list = []
    global_step = 0
    for epoch in tqdm(range(cfg.epochs)):
        for step, batch in tqdm(enumerate(dl)):
            images_list = batch["images"]             # List[List[PIL.Image]]
            prompts     = batch["prompt"]             # List[str]
            responses   = batch["response"]           # List[str]
            labels      = torch.tensor(batch["label"], dtype=torch.long, device=accelerator.device)  # [B]

            with torch.no_grad():
                tasks = batch["task"]  # 你数据有 task 字段；若没有就退化为用 prompt
                s_text = nli.score(tasks, responses).to(device)  # [B]
            
            # 2) 方案A：把 s→w 作为逐样本权重
            if use_sem_A:
                w = sem_weight(s_text, tau=tau_sem, k=k_sem)     # [B]
                # 对 label=0（失败样本），不强制等价：权重至少设为 0.5 防止完全失声
                w = torch.where(labels == cfg.positive_label_value, w, torch.clamp(w, min=0.5))
            else:
                w = None

            with accelerator.accumulate(actor):
                # 计算“仅 response 段”的平均 logp
                logp = compute_batch_logp(
                    actor, processor, images_list, prompts, responses,
                    requires_grad=True,
                )  # [B]

                if use_sem_A:
                    # 3) KTO 主损失（支持 sample_weights）
                    loss = kto_loss(
                        beta=cfg.beta, logp=logp, labels=labels,
                        desirable_weight=cfg.desirable_weight,
                        undesirable_weight=cfg.undesirable_weight,
                        positive_label_value=cfg.positive_label_value,
                        sample_weights=w
                    )
                else:
                    loss = kto_loss(
                        beta=cfg.beta,
                        logp=logp,
                        labels=labels,
                        desirable_weight=cfg.desirable_weight,
                        undesirable_weight=cfg.undesirable_weight,
                        positive_label_value=cfg.positive_label_value
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

    # 可选：做一次采样
    # from utils import generate_text_from_sample
    # gene_instru = generate_text_from_sample(actor, processor, ds[0], max_new_tokens=128, device=accelerator.device)
    # print("Sample generation after KTO:")
    # print(gene_instru)

if __name__ == "__main__":
    main()
