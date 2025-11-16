from generate_intruction import build_red_team_generator, CLIPEmbeddingModel
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch



def load_relate_model(cfg):

    device_model = "cuda"  # 绑定到“可见列表索引0”，也就是模型卡

    if cfg.semantic_type == "clip":

        semantic_model = CLIPEmbeddingModel(device=device_model)
    else:
        semantic_model = pipeline("text-classification", model="MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli")   # deberta-v3-large-mnli    roberta-large-mnli
        # stop = set(stopwords.words("english")) | set(string.punctuation)

    # 2) 选择生成后端：'smolvlm', 'qwenvl'、'verl_qwen' 或 'qwen_llm'

    print(f"[ERT] Using backend: {cfg.BACKEND}")

    if cfg.BACKEND == "qwenvl":
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
    elif cfg.BACKEND == "verl_qwen":
        # VERL 训练的 Qwen 语言模型（纯文本）
        # 需要配置 cfg.verl_model_path: VERL 训练的模型路径
        red_team = build_red_team_generator(
            backend="verl_qwen",
            embedding_model=semantic_model,
            model_path=cfg.verl_model_path,
            device=device_model,
        )
    else:
        raise ValueError(f"Unsupported backend: {cfg.BACKEND}. Must be 'smolvlm', 'qwenvl', 'verl_qwen' or 'qwen_llm'")
    
    return red_team
    

def parse_task_and_links(task_suite_name,task_to_links):

    task_suite = task_to_links[task_suite_name]
    task_language_list = task_suite.keys()

    return task_language_list, task_suite


def load_verl_qwen_rewriter(model_path, device="cuda"):
    """
    加载 VERL 训练的 Qwen 语言模型用于指令重写
    
    Args:
        model_path: VERL 训练的 Qwen 模型路径
        device: 设备，默认 "cuda"
    
    Returns:
        dict: 包含 model 和 tokenizer 的字典
    """
    print(f"[VERL Rewriter] Loading model from {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        padding_side='left'
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True
    )
    
    model.eval()
    
    print(f"[VERL Rewriter] Model loaded successfully on {device}")
    
    return {
        "model": model,
        "tokenizer": tokenizer
    }


# def rewrite_instruction_with_verl_qwen(
#     instruction,
#     rewriter_dict,
#     prompt_template=None,
#     max_new_tokens=128,
#     temperature=0.7,
#     top_p=0.9,
#     do_sample=True
# ):
#     """
#     使用 VERL 训练的 Qwen 模型重写指令
    
#     Args:
#         instruction: 原始指令文本
#         rewriter_dict: load_verl_qwen_rewriter 返回的字典
#         prompt_template: 提示词模板，如果为 None 则使用默认模板
#         max_new_tokens: 最大生成 token 数
#         temperature: 采样温度
#         top_p: nucleus sampling 参数
#         do_sample: 是否使用采样
    
#     Returns:
#         str: 重写后的指令
#     """
#     model = rewriter_dict["model"]
#     tokenizer = rewriter_dict["tokenizer"]
    
#     # 默认提示词模板
#     if prompt_template is None:
#         prompt_template = """You are a helpful assistant that rewrites robotic manipulation instructions to be clearer and more precise.

# Original instruction: {instruction}

# Rewritten instruction:"""
    
#     # 构建输入
#     prompt = prompt_template.format(instruction=instruction)
    
#     # 使用 Qwen 的 chat 模板（如果支持）
#     if hasattr(tokenizer, "apply_chat_template"):
#         messages = [
#             {"role": "system", "content": "You are a helpful assistant for rewriting robotic instructions."},
#             {"role": "user", "content": f"Rewrite this instruction to be clearer: {instruction}"}
#         ]
#         prompt = tokenizer.apply_chat_template(
#             messages,
#             tokenize=False,
#             add_generation_prompt=True
#         )
    
#     # Tokenize
#     inputs = tokenizer(
#         prompt,
#         return_tensors="pt",
#         padding=True,
#         truncation=True,
#         max_length=512
#     ).to(model.device)
    
#     # 生成
#     with torch.no_grad():
#         outputs = model.generate(
#             **inputs,
#             max_new_tokens=max_new_tokens,
#             temperature=temperature,
#             top_p=top_p,
#             do_sample=do_sample,
#             pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
#         )
    
#     # 解码
#     generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
#     # 提取重写后的指令（去掉 prompt 部分）
#     if prompt in generated_text:
#         rewritten = generated_text[len(prompt):].strip()
#     else:
#         # 如果使用了 chat 模板，可能需要更智能的提取
#         rewritten = generated_text.split("Rewritten instruction:")[-1].strip()
#         if not rewritten:
#             rewritten = generated_text.strip()
    
#     print(f"[VERL Rewriter] Original: {instruction}")
#     print(f"[VERL Rewriter] Rewritten: {rewritten}")
    
#     return rewritten


# def rewrite_instructions_batch(
#     instructions,
#     rewriter_dict,
#     prompt_template=None,
#     max_new_tokens=128,
#     temperature=0.7,
#     top_p=0.9,
#     do_sample=True,
#     batch_size=4
# ):
#     """
#     批量重写多条指令
    
#     Args:
#         instructions: 指令列表
#         rewriter_dict: load_verl_qwen_rewriter 返回的字典
#         prompt_template: 提示词模板
#         max_new_tokens: 最大生成 token 数
#         temperature: 采样温度
#         top_p: nucleus sampling 参数
#         do_sample: 是否使用采样
#         batch_size: 批处理大小
    
#     Returns:
#         list: 重写后的指令列表
#     """
#     rewritten_instructions = []
    
#     for i in range(0, len(instructions), batch_size):
#         batch = instructions[i:i + batch_size]
        
#         for instruction in batch:
#             rewritten = rewrite_instruction_with_verl_qwen(
#                 instruction,
#                 rewriter_dict,
#                 prompt_template=prompt_template,
#                 max_new_tokens=max_new_tokens,
#                 temperature=temperature,
#                 top_p=top_p,
#                 do_sample=do_sample
#             )
#             rewritten_instructions.append(rewritten)
    
#     return rewritten_instructions