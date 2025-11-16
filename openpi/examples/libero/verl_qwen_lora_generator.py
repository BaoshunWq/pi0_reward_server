from typing import List, Optional, Union, Callable
import os
import json
import shutil
import tempfile
import torch
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoProcessor,
    AutoModelForVision2Seq,
    AutoConfig,
    AutoModelForCausalLM,
)
from qwen_vl_utils import process_vision_info
from vllm import LLM, SamplingParams
from vllm.multimodal.utils import encode_image_base64

def _clean_line(s: str) -> str:
    """清理文本行，去除首尾空白和特殊前缀"""
    s = s.strip()
    if not s:
        return ""
    return s.lstrip("-* \t").strip()


def _dedup_preserve_order(items: List[str]) -> List[str]:
    """去重但保持顺序"""
    seen = set()
    result = []
    for it in items:
        if it and it not in seen:
            seen.add(it)
            result.append(it)
    return result


DEVICE_DEFAULT = "cuda" if torch.cuda.is_available() else "cpu"


def _select_topk_by_similarity(embedding_model: Callable, semantic_type: str, task: str, candidates: List[str], select_topk: int):
    """根据相似度选择 top-k 候选指令"""
    if semantic_type == "clip":
        sims = (
            embedding_model(source_instruct=task, text=candidates)
            .detach()
            .cpu()
            .numpy()
            .squeeze(-1)  # (N,)
        )
    else:
        # 对于其他语义类型，使用 semantic_shift
        def semantic_shift(nli, a, b):
            entail_ab = nli(f"{a} </s></s> {b}")[0]['label']
            entail_ba = nli(f"{b} </s></s> {a}")[0]['label']
            if entail_ab == 'entailment':
                entail_ab_smi = 1.0
            elif entail_ab == 'nautral':
                entail_ab_smi = 0.5
            else:
                entail_ab_smi = 0.0
            
            if entail_ba == 'entailment':
                entail_ba_smi = 1.0
            elif entail_ba == 'nautral':
                entail_ba_smi = 0.5
            else:
                entail_ba_smi = 0.0

            similarity = (entail_ab_smi + entail_ba_smi) / 2.0
            return similarity
        
        sims = [semantic_shift(embedding_model, task, candidate) for candidate in candidates]
        sims = np.array(sims)
    
    if len(candidates) == 1:
        # 确保返回列表格式
        if isinstance(sims, np.ndarray):
            sims_list = sims.tolist() if sims.ndim > 0 else [float(sims)]
        else:
            sims_list = [sims] if not isinstance(sims, list) else sims
        return candidates, sims_list
    
    top_idx = np.argsort(sims)[-select_topk:][::-1]  # 从大到小
    
    # 确保相似度值是 Python 原生类型
    selected_sims = [float(sims[i]) for i in top_idx]
    return [candidates[i] for i in top_idx], selected_sims


class VERLQwenLoRAGenerator:
    def __init__(self, embedding_model: Callable, lora_path: Optional[str] = None, device: str = DEVICE_DEFAULT):
        self.embedding_model = embedding_model
        self.device = device
        self.model_path = "Qwen/Qwen2-VL-2B-Instruct"  # 基础模型路径，用于加载 processor
        
        # 加载模型（匹配训练时的配置）
        # 如果有 LoRA，需要在初始化时启用 LoRA 支持
        llm_kwargs = {
            "model": self.model_path,
            "tensor_parallel_size": 1,
            "gpu_memory_utilization": 0.6,
            "enforce_eager": True,
            "enable_chunked_prefill": False,
            # disable_mm_preprocessor_cache=True,  # Qwen2-VL 特殊配置
        }
        
        # 如果有 LoRA，启用 LoRA 支持并加载
        self.lora_name = None
        if lora_path and os.path.exists(lora_path):
            print(f"[VERL-Qwen-LoRA] Attempting to load LoRA adapter from: {lora_path}")
            try:
                # 尝试启用 LoRA 支持
                llm_kwargs_with_lora = llm_kwargs.copy()
                llm_kwargs_with_lora["enable_lora"] = True
                llm_kwargs_with_lora["max_lora_rank"] = 16  # 根据实际 LoRA 配置调整
                # 初始化 LLM
                self.llm = LLM(**llm_kwargs_with_lora)
                # 尝试加载 LoRA（使用不同的可能方法名）
                lora_loaded = False
                if hasattr(self.llm, "add_lora"):
                    try:
                        self.llm.add_lora(lora_path, lora_name="verl_lora")
                        self.lora_name = "verl_lora"
                        lora_loaded = True
                        print(f"[VERL-Qwen-LoRA] Successfully loaded LoRA using add_lora()")
                    except Exception as e:
                        print(f"[VERL-Qwen-LoRA] add_lora() failed: {e}")
                
                if not lora_loaded and hasattr(self.llm, "load_lora"):
                    try:
                        self.llm.load_lora(lora_path, lora_name="verl_lora")
                        self.lora_name = "verl_lora"
                        lora_loaded = True
                        print(f"[VERL-Qwen-LoRA] Successfully loaded LoRA using load_lora()")
                    except Exception as e:
                        print(f"[VERL-Qwen-LoRA] load_lora() failed: {e}")
                
                if not lora_loaded and hasattr(self.llm, "llm_engine") and hasattr(self.llm.llm_engine, "add_lora"):
                    try:
                        self.llm.llm_engine.add_lora(lora_path, lora_name="verl_lora")
                        self.lora_name = "verl_lora"
                        lora_loaded = True
                        print(f"[VERL-Qwen-LoRA] Successfully loaded LoRA using llm_engine.add_lora()")
                    except Exception as e:
                        print(f"[VERL-Qwen-LoRA] llm_engine.add_lora() failed: {e}")
                
                if not lora_loaded:
                    print(f"[VERL-Qwen-LoRA] Warning: vLLM LoRA API not found or failed. Using base model without LoRA.")
                    # 如果 LoRA 加载失败，重新初始化基础模型
                    self.llm = LLM(**llm_kwargs)
                    self.lora_name = None
            except Exception as e:
                print(f"[VERL-Qwen-LoRA] Error during LoRA initialization: {e}")
                print(f"[VERL-Qwen-LoRA] Falling back to base model without LoRA...")
                # 如果 LoRA 加载失败，回退到基础模型
                self.llm = LLM(**llm_kwargs)
                self.lora_name = None
        else:
            # 没有 LoRA，直接初始化
            if lora_path:
                print(f"[VERL-Qwen-LoRA] Warning: LoRA path provided but does not exist: {lora_path}")
            self.llm = LLM(**llm_kwargs)
        
        # 预加载 processor（用于处理多模态输入）
        self.processor = AutoProcessor.from_pretrained(
            self.model_path, trust_remote_code=True
        )
    


    def __call__(self,
                 task: str,
                 semantic_type: str = "clip",
                 image_url: Optional[str] = None,
                 num_instructions: int = 10,
                 return_all_annotations: bool = False,
                 select_topk: int = 3,
                 temperature: float = 1.0,
                 max_new_tokens: int = 512) -> Union[List[str], tuple]:
        # 使用预加载的 processor
        processor = self.processor
        
        # 构建提示词
        PROMPT_SYSTEM = (
            "You are an adversarial instruction generator for testing robot vision-language models. "
            "Your goal is to create instructions that describe the task correctly but may challenge or confuse the robot. "
            "You can add distracting details, ambiguous phrasing, or misleading context while keeping the core task intact."
        )
        
        if num_instructions == 1:
            user_text = (
                f"Task: {task}. "
                f"Rewrite this as a natural instruction that describes the same task. "
                f"You may add extra details, constraints, or context to make it more challenging for the robot. "
                f"Output ONLY the rewritten instruction. Keep it concise but allow adding adversarial elements."
            )
        else:
            user_text = (
                f"Task: {task}. "
                f"Generate {num_instructions} variations of this instruction. "
                f"Each should describe the task but may include distracting or challenging details. "
                f"Output one per line."
            )
        messages_with_images = [
            {"role": "system", "content": PROMPT_SYSTEM},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image_url": image_url} if image_url else {"type": "text", "text": "[No image provided]"},
                    {"type": "text", "text": user_text},
                ],
            },
        ]
        
        text = processor.apply_chat_template(
            messages_with_images, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages_with_images)
        
        # 准备多模态输入
        multi_modal_data = {}
        if image_inputs:
            multi_modal_data["image"] = image_inputs
        
        # 设置采样参数（与训练时一致）
        sampling_params = SamplingParams(
            temperature=temperature,  # 使用传入的 temperature 参数
            top_p=0.9,
            max_tokens=max_new_tokens,
            n=1,  # 每次生成一个，如果需要多个可以循环
        )
        
        # 生成（如果使用 LoRA，传递 lora_request；否则不传）
        generate_kwargs = {
            "prompts": [text],
            "sampling_params": sampling_params,
        }
        if multi_modal_data:
            generate_kwargs["multi_modal_data"] = multi_modal_data
        if self.lora_name:
            generate_kwargs["lora_request"] = self.lora_name
        
        outputs = self.llm.generate(**generate_kwargs)
        
        # 解析结果
        response = outputs[0].outputs[0].text
        
        if num_instructions == 1:
            parsed = [_clean_line(response)] if response else []
        else:
            parsed = [_clean_line(line) for line in response.split("\n") if _clean_line(line)]
        
        # return _dedup_preserve_order(parsed) or [task]
        
        candidates = _dedup_preserve_order(parsed) or [task]
        
        # 如果没有候选指令，返回默认值
        if not candidates:
            candidates = ["Please pick up the object and place it as requested."]
        
        # 使用 embedding_model 进行相似度筛选
        if return_all_annotations:
            selected, selected_sim = _select_topk_by_similarity(
                self.embedding_model, semantic_type, task, candidates, 1
            )
            return selected[0], selected_sim, candidates
        
        selected, selected_sim = _select_topk_by_similarity(
            self.embedding_model, semantic_type, task, candidates, select_topk
        )
        
        return selected, selected_sim
