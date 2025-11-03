import os
from pathlib import Path
from typing import List, Dict, Any

from datasets import Dataset
from PIL import Image, ImageDraw, ImageFont

import torch
from transformers import AutoProcessor

# 兼容不同 Transformers 版本的 VLM 类名
try:
    from transformers import AutoModelForImageTextToText as VLMClass
except ImportError:
    from transformers import AutoModelForVision2Seq as VLMClass

from trl.trl.trainer import KTOTrainer, KTOConfig
from peft import LoraConfig, get_peft_model

from datasets import load_dataset
from PIL import Image

ds = load_dataset("json", data_files="your.jsonl")["train"]

def to_kto(ex):
    # 路径->PIL，并放进 images 列
    img = ex["image"]
    if isinstance(img, str):
        img = Image.open(img).convert("RGB")
    ex["images"] = [img]
    # 字段改名对齐
    ex["response"] = ex.get("response") or ex.get("answer") or ex.get("output")
    # reward->label（把 {True,False} / {1,0} 统一为 0/1）
    r = ex.get("reward") or ex.get("label")
    ex["label"] = int(bool(r))
    return ex

ds = ds.map(to_kto, remove_columns=[c for c in ds.column_names if c not in {"images","prompt","response","label"}])



def ensure_dummy_image(path: str, text: str = "demo"):
    """如果没有图片，生成一张占位图，保证脚本可独立运行。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not p.exists():
        img = Image.new("RGB", (512, 384), color=(240, 240, 240))
        draw = ImageDraw.Draw(img)
        try:
            # 字体可选；容器里通常没有 ttf，按默认字体也行
            draw.text((20, 20), f"{text}", fill=(10, 10, 10))
        except Exception:
            pass
        img.save(p)


def build_demo_dataset() -> Dataset:
    """
    构造一个最小可跑的数据集：
      - images: list[PIL.Image]  （多图时可放多张，这里就放1张）
      - prompt: str
      - response: str
      - label: int (0/1) —— 1 表示偏好/正样本，0 表示负样本
    """
    img_path = "data/imgs/1.jpg"
    ensure_dummy_image(img_path, text="a running dog")

    # 两条样本：同一张图，不同回答，分别标 1（正）和 0（负）
    raw = [
        {
            "image": img_path,
            "prompt": "Describe the image briefly.",
            "response": "A dog is running on the grass.",  # 正样本
            "label": 1,
        },
        {
            "image": img_path,
            "prompt": "Describe the image briefly.",
            "response": "A cat is sleeping indoors.",       # 负样本
            "label": 0,
        },
    ]

    def to_pil(ex):
        img = ex["image"]
        if isinstance(img, str):
            img = Image.open(img).convert("RGB")
        # VLM 推荐把多模态输入放到一个 list 里（支持多图）
        ex["images"] = [img]
        return ex

    ds = Dataset.from_list(raw).map(to_pil)
    return ds


def main():
    # 1) 模型与处理器（换成你自己的权重 ID 也行）
    model_id = "HuggingFaceTB/SmolVLM-Instruct"
    print(f"Loading model: {model_id}")
    processor = AutoProcessor.from_pretrained(model_id)
    model = VLMClass.from_pretrained(model_id, torch_dtype="auto")

    # 可选：开启 gradient checkpointing 节省显存
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    # 2) LoRA（强烈建议：只调少量参数，稳定省显存）
    lora = LoraConfig(
        r=16, lora_alpha=16, lora_dropout=0.05, bias="none",
        target_modules=[
            # 下面这些是常见的注意力/MLP投影，若部分不匹配，可打印 model.named_modules() 调整
            "q_proj","k_proj","v_proj","o_proj",
            "gate_proj","up_proj","down_proj"
        ],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    # 3) 构造/加载数据集（这里用示例数据）
    train_ds = build_demo_dataset()

    # 4) KTO 配置
    args = KTOConfig(
        output_dir="smolvlm-kto",
        learning_rate=5e-6,
        beta=0.1,                       # KTO 关键超参，可在 0.05~0.2 间调
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=1,
        bf16=torch.cuda.is_available(), # 有 bfloat16 就开；否则可用 fp16=True
        remove_unused_columns=False,    # 关键：保留 images 列，给 processor 用
        max_prompt_length=512,
        max_length=640,
        max_completion_length=128,
        logging_steps=5,
        save_steps=50,
        report_to=[],                   # 需要 WandB/MLflow 再改
    )

    # 5) Trainer（VLM 一定要传 processing_class=processor）
    trainer = KTOTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        processing_class=processor,
        # 如果你有验证集：eval_dataset=eval_ds,
    )

    # 6) 训练 & 保存（保存的是 LoRA 适配器）
    trainer.train()
    trainer.save_model("smolvlm-kto-lora")
    print("✅ Training done. LoRA adapter saved to ./smolvlm-kto-lora")


if __name__ == "__main__":
    main()
