from transformers import AutoModelForVision2Seq
model = AutoModelForVision2Seq.from_pretrained(
    "openvla/openvla-7b",
    trust_remote_code=True
)
print(model.norm_stats.keys())
