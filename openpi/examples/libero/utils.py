from generate_intruction import build_red_team_generator, CLIPEmbeddingModel
from transformers import pipeline



def load_relate_model(cfg):

    device_model = "cuda"  # 绑定到“可见列表索引0”，也就是模型卡

    if cfg.semantic_type == "clip":

        semantic_model = CLIPEmbeddingModel(device=device_model)
    else:
        semantic_model = pipeline("text-classification", model="MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli")   # deberta-v3-large-mnli    roberta-large-mnli
        # stop = set(stopwords.words("english")) | set(string.punctuation)

    # 2) 选择生成后端：'smolvlm' 或 'qwenvl'

    print(f"[ERT] Using backend: {cfg.BACKEND}")

    if cfg.BACKEND == "smolvlm":
        red_team = build_red_team_generator(
            backend="smolvlm",
            embedding_model=semantic_model,
            model_path="HuggingFaceTB/SmolVLM-Instruct",  # "HuggingFaceTB/SmolVLM-500M-Instruct"。HuggingFaceTB/SmolVLM-Instruct
            custom_lora_path=cfg.custom_lora_path,  # 如需可加
            device=device_model,
        )
    else:
        # Qwen-VL：本地或 API
        # - 本地：model="Qwen/Qwen2.5-VL-7B-Instruct", mode="local"
        # - API ：model="qwen2.5-vl-72b-instruct",  mode="api"（需 export DASHSCOPE_API_KEY=...）
        red_team = build_red_team_generator(
            backend="qwenvl",
            embedding_model=semantic_model,
            mode=cfg.qwen_mode,
            model=cfg.qwen_model_id,
            device=device_model,
        )
    
    return red_team
    

def parse_task_and_links(task_suite_name,task_to_links):

    task_suite = task_to_links[task_suite_name]
    task_language_list = task_suite.keys()

    return task_language_list, task_suite