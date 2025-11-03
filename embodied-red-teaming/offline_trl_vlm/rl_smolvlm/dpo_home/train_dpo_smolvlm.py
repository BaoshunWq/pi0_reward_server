from datasets import Dataset
from PIL import Image
from transformers import AutoProcessor
try:
    from transformers import AutoModelForImageTextToText as VLMClass   # 新版
except ImportError:
    from transformers import AutoModelForVision2Seq as VLMClass        # 兼容旧版

from trl.trainer import DPOTrainer, DPOConfig
from peft import LoraConfig, get_peft_model

# 1) 你的本地数据（举例）：每条包含 image(路径或URL)、prompt、chosen、rejected
# raw = [
#   {"image":"data/imgs/1.jpg",
#   "prompt":"Describe the image",
#    "chosen":"A dog is running on the grass.",
#    "rejected":"A cat is sleeping indoors."},
# ]
from trl_relate.utils import make_data_for_trl
from datasets import load_dataset, Features, Value, Image as HFImage


DPO_DATA_PATH = "libero_spatial_dpo_new.json"   # 你的 DPO 数据

ds = make_data_for_trl(DPO_DATA_PATH, data_type="dpo")  # DPO 数据集

# 3) 模型与处理器（SmolVLM）
model_id = "HuggingFaceTB/SmolVLM-Instruct"
processor = AutoProcessor.from_pretrained(model_id)
model = VLMClass.from_pretrained(model_id, torch_dtype="auto")

# 可选：LoRA（只训练少量参数，显存友好）
peft_cfg = LoraConfig(r=16, lora_alpha=16, lora_dropout=0.05,
                      target_modules=["q_proj","k_proj","v_proj","o_proj",
                                      "gate_proj","up_proj","down_proj"], bias="none")
model = get_peft_model(model, peft_cfg)

model.to("cuda")

# 4) DPO 配置（VLM 要把 processing_class 换成 processor）
args = DPOConfig(
    output_dir="smolvlm-dpo",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=5e-6,
    num_train_epochs=1,
    beta=0.1,
    bf16=True,
    remove_unused_columns=False,     # 关键！让图像列保留下来
    max_prompt_length=512,
    max_completion_length=128,
    max_length=640,
)

trainer = DPOTrainer(
    model=model,
    args=args,
    train_dataset=ds,
    processing_class=processor,      # <-- VLM 用 processor（不是 tokenizer）
)

trainer.train()
trainer.save_model("smolvlm-dpo")
