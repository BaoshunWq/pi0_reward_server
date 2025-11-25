from typing import List, Dict, Callable, Optional, Union
import os
import re
import json
# import random
# import functools
# import multiprocessing
# from pydantic import BaseModel
import shutil
import tempfile
import nltk
from nltk.corpus import stopwords
from tqdm import tqdm
import numpy as np
import torch
from transformers import (
    AutoTokenizer,
    AutoProcessor,
    AutoModelForVision2Seq,
    CLIPTokenizerFast,
    CLIPTextModelWithProjection,
)
from transformers.image_utils import load_image
import string
from qwen_vl_utils import process_vision_info
from verl_qwen_lora_generator import VERLQwenLoRAGenerator
HF_API_TOKEN = "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"
DASHSCOPE_API_KEY = "sk-c48456b8ea2643cb9209979aed586cff"
stop = set(stopwords.words("english")) | set(string.punctuation)
# =========================
# 相似度与嵌入（与原始实现对齐）
# =========================

def pairwise_cosine_similarity(x1, x2):
    """
    x1: (N, D), x2: (M, D) -> (N, M)
    """
    x1 = x1 / x1.norm(dim=1, keepdim=True)
    x2 = x2 / x2.norm(dim=1, keepdim=True)
    return torch.matmul(x1, x2.t())

DEVICE_DEFAULT = "cuda" if torch.cuda.is_available() else "cpu"
# CLIP_MODEL_ID = "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"
CLIP_MODEL_ID = "openai/clip-vit-base-patch32"


class CLIPEmbeddingModel:
    """
    与你原先版本保持一致：
    __call__(source_instruct, text_list) -> (N, 1) 相似度
    """
    def __init__(self, device: str = DEVICE_DEFAULT):
        self.device = device
        # 移除 use_safetensors 参数,让 transformers 自动选择
        self.model = CLIPTextModelWithProjection.from_pretrained(
            CLIP_MODEL_ID,
        ).to(self.device).eval()
        self.tokenizer = CLIPTokenizerFast.from_pretrained(CLIP_MODEL_ID)

    @torch.no_grad()
    def __call__(self, source_instruct: str, text: List[str]) -> torch.Tensor:
        if isinstance(text, str):
            text = [text]

        # 源任务向量
        src_inputs = self.tokenizer(source_instruct, padding=True,    truncation=True,  max_length=77,  return_tensors="pt").to(self.device)
        src_embeds = self.model(**src_inputs).text_embeds  # (1, D)

        # 候选指令向量
        cand_inputs = self.tokenizer(text, padding=True,    truncation=True,    max_length=77,  return_tensors="pt").to(self.device)
        cand_embeds = self.model(**cand_inputs).text_embeds  # (N, D)

        # 相似度 (N, 1)
        sim = pairwise_cosine_similarity(cand_embeds, src_embeds)  # (N, 1)
        return sim
    
def keywords(text):
    
    return {w.lower() for w in nltk.word_tokenize(text) if w.lower() not in stop}

def semantic_shift(nli,a, b):
    entail_ab = nli(f"{a} </s></s> {b}")[0]['label']
    entail_ba = nli(f"{b} </s></s> {a}")[0]['label']
    # kw_a, kw_b = keywords(a), keywords(b)
    # recall = len(kw_a & kw_b) / len(kw_a)
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

    similarity =  (entail_ab_smi + entail_ba_smi) / 2.0
    return similarity

# =========================
# 工具
# =========================

def _hf_to_raw(url: Optional[str]) -> Optional[str]:
    if not url:
        return url
    if "huggingface.co" in url and "/blob/" in url:
        url = url.replace("/blob/", "/resolve/")
        if "?" not in url:
            url += "?download=1"
    return url

def _clean_line(s: str) -> str:
    s = s.strip()
    if not s:
        return ""
    # 去掉编号/项目符号
    s = s.lstrip("-* \t")
    s = re.sub(r'^\s*\d+\.\s*', '', s)
    return s.strip()

def _dedup_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for it in items:
        if it and it not in seen:
            seen.add(it)
            out.append(it)
    return out




def _select_topk_by_similarity(embedding_model: Callable,semantic_type, task: str, candidates: List[str], select_topk: int) -> List[str]:
    if semantic_type == "clip":
        sims = (
            embedding_model(source_instruct=task, text=candidates)
            .detach()
            .cpu()
            .numpy()
            .squeeze(-1)  # (N,)
        )
    else:
        sims = [semantic_shift(embedding_model,task, candidate) for candidate in candidates]
    
    if len(candidates) == 1:
        return candidates, sims
        

    top_idx = np.argsort(sims)[-select_topk:][::-1]  

    # import pdb; pdb.set_trace()

    return [candidates[i] for i in top_idx],[sims[i] for i in top_idx]




class EmbodiedRedTeamModelWithQwenVL:
    """
    Qwen-VL 生成：
    - mode="local"：使用 HuggingFace 本地推理
      支持模型: Qwen/Qwen2-VL-2B-Instruct, Qwen/Qwen2.5-VL-7B-Instruct 等
    - mode="api"  ：使用 DashScope（OpenAI 兼容接口，需设置环境变量 DASHSCOPE_API_KEY）
    """
    def __init__(
        self,
        embedding_model: Callable,
        mode: str = "local",  # "local" | "api"
        model: str = "Qwen/Qwen2.5-VL-7B-Instruct",  # local 时使用 HF 模型名 (如 Qwen/Qwen2-VL-2B-Instruct)；api 时使用 DashScope 模型名，如 "qwen2.5-vl-72b-instruct"
        device: str = DEVICE_DEFAULT,
    ):
        self.embedding_model = embedding_model
        self.mode = mode.lower()
        self.device = device
        self.model_name = model

        if self.mode == "local":
            print(f"[Qwen-VL local] Loading '{self.model_name}' on '{self.device}' ...")
            model_dtype = (
                torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
                else (torch.float16 if torch.cuda.is_available() else torch.float32)
            )
            self.model = AutoModelForVision2Seq.from_pretrained(
                self.model_name,
                torch_dtype=model_dtype,
                trust_remote_code=True,
            ).to(self.device).eval()
            self.processor = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=True)
            print("[Qwen-VL local] Ready.")
        elif self.mode == "api":
            api_key = DASHSCOPE_API_KEY
            if not api_key:
                raise ValueError("DASHSCOPE_API_KEY 未设置，无法使用 Qwen-VL API 模式。请 export DASHSCOPE_API_KEY=...")
            # 延迟导入，避免无用依赖
            from openai import OpenAI  # OpenAI 兼容客户端
            self.client = OpenAI(api_key=api_key, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
            print(f"[Qwen-VL API] Using model '{self.model_name}' via DashScope.")
        else:
            raise ValueError("mode 必须是 'local' 或 'api'。")

    def _gen_local(
        self,
        task: str,
        image_url: Optional[str],
        num_instructions: int,
    ) -> List[str]:
        image_url = _hf_to_raw(image_url)
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a quality assurance engineer for a robot. "
                    "Your goal is to come up with instructions that describe the given task correctly, "
                    "are similar to what human users would possibly give, and yet challenge the robot's capability."
                ),
            },
            {
                "role": "user",
                "content": [
                    {"type": "image",},
                    {
                        "type": "text",
                        "text": f"The attached image is an example of the initial state for the task: {task}. "
                                f"Generate a diverse set of exactly instructions."
                    }],
            },
        ]

        image = load_image(image_url) if image_url else None
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        inputs = self.processor(
            text=[prompt],
            images=[image] if image is not None else None,
            return_tensors="pt"
        ).to(self.device)

        num_samples = max(1, min(5, num_instructions))
        gen_ids = self.model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.8,
            num_return_sequences=num_samples,
        )
        gen_only = gen_ids[:, inputs["input_ids"].shape[-1]:]
        decoded = self.processor.batch_decode(gen_only, skip_special_tokens=True)

        candidates: List[str] = []
        for sample in decoded:
            for line in sample.splitlines():
                line = _clean_line(line)
                if line:
                    candidates.append(line)
        return _dedup_preserve_order(candidates)

    def _gen_api(
        self,
        task: str,
        image_url: Optional[str],
        num_instructions: int,
    ) -> List[str]:
        # OpenAI 兼容：DashScope 目前对多模态对话的参数以纯文本传递图片 URL 最稳妥
        from openai import OpenAI  # type: ignore
        if image_url:
            content = [
                {
                    "type": "text", 
                    "text": f"The attached image is an example image of the initial state of a robot that will perform the task: {task}. Generate a diverse set of exactly {num_instructions} instructions."
                },
                {
                    "type": "image_url",
                    "image_url": {"url": image_url},
                },
            ]

        # 生成多个样本：n 不一定所有模型都支持；这里保守循环采样
        # api_samples = max(1, min(5, num_instructions))
        # outputs: List[str] = []
        # for _ in range(api_samples):
        chat_completion = self.client.chat.completions.create(
            # model=self.model_name,
            model="qwen2.5-vl-72b-instruct",
            messages=[
                {
                    "role": "system",
                    "content": "You are a quality assurance engineer for a robot. Your goal is to come up with instructions that describe the given task correctly, is similar to what human users would possibly give, and yet challenge the robot's capability on accomplishing the task."
                },
                {
                    "role": "user",
                    # 将 content 转为文本字符串（DashScope API 不接受 list）
                    "content": "\n".join(
                        [c["text"] if c["type"] == "text" else f"Image URL: {c['image_url']['url']}" for c in content]
                    )
                }
            ],
            temperature=0.8,
            max_tokens=512,
        )
        text = chat_completion.choices[0].message.content.strip()
        # outputs.append(text)

        for choice in chat_completion.choices:
            # 获取模型输出的文本
            text = choice.message.content.strip()

            # 按行拆分成指令列表
            annotations = [line.strip("- ").strip() for line in text.split("\n") if line.strip()]
            # all_annotations.append(annotations)

        all_annotations = annotations
        return _dedup_preserve_order(all_annotations)

    def __call__(
        self,
        task: str,
        semantic_type,
        image_url: Optional[str] = None,
        num_instructions: int = 10,
        return_all_annotations: bool = False,
        select_topk: int = 3,
    ) -> Union[List[str], tuple]:
        if self.mode == "local":
            candidates = self._gen_local(task, image_url, num_instructions)
        else:
            candidates = self._gen_api(task, image_url, num_instructions)

        if not candidates:
            candidates = ["Please pick up the object and place it as requested."]

        if return_all_annotations:
            selected,selected_sim = _select_topk_by_similarity(self.embedding_model,semantic_type, task, candidates, 1)
            
            return selected[0],selected_sim, candidates
        
        selected,selected_sim = _select_topk_by_similarity(self.embedding_model,semantic_type, task, candidates, select_topk)

        return selected,selected_sim





def build_red_team_generator(
    backend: str,
    embedding_model: Callable,
    **kwargs,
):
    backend = backend.lower()
    lora_path = kwargs.get("lora_path", None)

    if backend == "qwenvl":
        # kwargs 可包含 mode="local"/"api", model="Qwen/..."/"qwen2.5-vl-72b-instruct"
        return EmbodiedRedTeamModelWithQwenVL(embedding_model=embedding_model, **kwargs)
    elif backend == "verl_qwen":
        # kwargs 应包含 model_path: VERL 训练的 Qwen 模型路径
        return VERLQwenLoRAGenerator(embedding_model=embedding_model, lora_path=lora_path)
    else:
        raise ValueError("backend 必须是 'smolvlm', 'qwenvl' 或 'verl_qwen'")


# =========================
# 示例
# =========================

if __name__ == '__main__':
    if not torch.cuda.is_available():
        print("CUDA is not available. Models may run slowly on CPU.")

    # 1) 嵌入模型（用于相似度筛选）
    clip_embedder = CLIPEmbeddingModel(device=DEVICE_DEFAULT)

    # 2) 选择生成后端：'smolvlm' 或 'qwenvl'
    BACKEND = "qwenvl"
    print(f"[ERT] Using backend: {BACKEND}")

    if BACKEND == "smolvlm":
        red_team = build_red_team_generator(
            backend="smolvlm",
            embedding_model=clip_embedder,
            model_path="HuggingFaceTB/SmolVLM-Instruct",
            # custom_lora_path="...",  # 如需可加
            device=DEVICE_DEFAULT,
        )
    else:
        # Qwen-VL：本地或 API
        # - 本地：model="Qwen/Qwen2-VL-2B-Instruct" 或 "Qwen/Qwen2.5-VL-7B-Instruct", mode="local"
        # - API ：model="qwen2.5-vl-72b-instruct",  mode="api"（需 export DASHSCOPE_API_KEY=...）
        red_team = build_red_team_generator(
            backend="qwenvl",
            embedding_model=clip_embedder,
            mode=os.getenv("QWEN_MODE", "local"),
            model=os.getenv("QWEN_MODEL", "qwen2.5-vl-72b-instruct"),
            device=DEVICE_DEFAULT,
        )

    # 3) 任务与图像
    task_description = "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate.".replace("_", " ")
    image_url = "libero-init-frames/libero_object_task-0_img-0_frame0.png"

    print(f"\nGenerating instructions for: '{task_description}' ...")
    out = red_team(
        task=task_description,
        image_url=image_url,
        num_instructions=10,
        select_topk=5
    )

    print("\n--- Selected (Top-k by similarity) ---")
    if isinstance(out, tuple):
        best_one, all_cands = out
        print(f"Best: {best_one}")
        for i, inst in enumerate(all_cands, 1):
            print(f"- {i}. {inst}")
    else:
        for i, inst in enumerate(out, 1):
            print(f"{i}. {inst}")
