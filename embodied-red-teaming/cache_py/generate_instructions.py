from typing import List, Dict, Callable
import os
import yaml
import json
import openai
import random
# import backoff
import functools
import multiprocessing
from openai import OpenAI
from pydantic import BaseModel
import requests

from tqdm import tqdm
import numpy as np
import torch
from transformers import AutoTokenizer, CLIPTextModelWithProjection

HF_API_TOKEN = "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"
DASHSCOPE_API_KEY = "sk-c48456b8ea2643cb9209979aed586cff"

PREFER_PROMPTS = {
    "REL_SYN": "Replace only spatial relation phrases with synonyms (e.g., 'next to'→'beside/adjacent to/by', 'in'↔'inside/into').",  # 关系词同义：仅替换空间关系短语为近义表达，保持原意不变
    "VERB_STYLE": "Change only verbs and tone (e.g., 'put'↔'place/set'); optionally add/remove 'please' and switch active/passive voice.",  # 动词/语气：只改动词与语气，可加/去 please，主动/被动互换
    "ORDER": "Reorder phrases or split/merge sentences without changing meaning (e.g., prepose/postpose prepositional phrases).",  # 词序/结构：只调整短语顺序或拆并句子，不改变语义
    "FORMAT": "Change numbering format only while keeping the same index (e.g., 'drawer-2'↔'drawer 2'↔'second drawer').",  # 格式变体：仅改编号/连字符/序数的写法，编号语义不变
    "FUNC_WORDS": "Add or remove function words only ('the', 'a', 'an', 'please'); keep meaning identical.",  # 功能词微扰：仅增删 the/a/an/please，含义不变
    "CHAR_NOISE": "Inject light keyboard/OCR-like typos or small punctuation insertions; keep text readable and natural.",  # 字符噪声：少量键盘/OCR错字或标点插入，保持可读自然
    "PRONOUN": "Replace specific nouns with pronouns ('it', 'that one') or omit details to increase ambiguity while sounding natural.",  # 指代/省略：名词改代词或省略信息，使表达更含糊但自然
    "UNDER_OVER": "Under-specify by removing necessary modifiers or over-specify by adding irrelevant ones to challenge grounding.",  # 欠/过度指定：删必要修饰或加无关修饰以增加歧义
    "FRAME": "Change the frame of reference (e.g., 'to your left', 'to the robot’s left', 'left of the lamp').",  # 参照系变体：切换参照系表达方式（你的左/机器人的左/某物左侧等）
    "NEGATION": "Use a two-clause form 'Do not X; instead Y.' where Y matches the original intended action.",  # 否定/双指令：构造“不要做X；而是做Y”双子句，Y与原意一致
    "CONFLICT": "Introduce a positional conflict by mentioning incompatible cues (e.g., both 'left' and 'right') while sounding natural.",  # 冲突方位：在一句中同时出现互相冲突的方位提示且保持自然
    "DISTRACTOR": "Add plausible but irrelevant distractor details (e.g., similar colors or nearby objects) to increase confusion.",  # 干扰注入：加入看似合理但无关的细节（颜色/邻近物）以增加混淆
}

def backoff_hdlr(details):
    print ("Backing off {wait:0.1f} seconds after {tries} tries "
           "calling function {target} with args {args} and kwargs "
           "{kwargs}".format(**details))

def pairwise_cosine_similarity(x1, x2):
    """
    Computes pairwise cosine similarity between two tensors.
    Args:
        x1 (torch.Tensor): First tensor of shape (N, D)
        x2 (torch.Tensor): Second tensor of shape (M, D)
    Returns:
        torch.Tensor: Pairwise cosine similarity matrix of shape (N, M)
    """
    x1 = x1 / x1.norm(dim=1, keepdim=True)
    x2 = x2 / x2.norm(dim=1, keepdim=True)
    return torch.matmul(x1, x2.t())

class CLIPEmbeddingModel:

    def __init__(self, device="cuda"):
        self.device = device
        self.model = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-base-patch32",).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32", 
                            clean_up_tokenization_spaces=True)

    def __call__(self, text: List[str]) -> torch.Tensor:
        with torch.no_grad():
            inputs = self.tokenizer(text, padding=True, return_tensors="pt").to(self.device)
            outputs = self.model(**inputs)
            text_embeds = outputs.text_embeds
        return text_embeds


class InstructionSet(BaseModel):
    instructions: List[str]

    

class EmbodiedRedTeamModelWithQwen2:
    """
    用 Hugging Face Inference API 调用 Qwen2.5-VL 生成攻击指令
    """
    def __init__(self,
                embedding_model: Callable,
                model: str = "qwen2.5-vl-72b-instruct",
                num_rejection_samples: int = 5):
        """
        model: HuggingFace 模型名
        num_rejection_samples: 为多样性采样的数量
        """
        # 使用 DashScope（OpenAI-compatible）客户端
        api_key = DASHSCOPE_API_KEY
        if not api_key:
            raise ValueError("请先设置环境变量 DASHSCOPE_API_KEY（DashScope API Key）")
        # 北京区 base_url。若使用新加坡区，请替换为 "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
        self.client = OpenAI(api_key=api_key, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
        # 推荐把 model 改为 DashScope 提供的 Qwen 视觉-语言模型名称
        self.model = model  # 或 "qwen-plus" / "qwen-vl-plus" 按你的账户可用模型调整

        # self.num_rejection_samples = num_rejection_samples
        self.embedding_model = embedding_model
        # self.api_url = f"https://api-inference.huggingface.co/{self.model}"
        # self.headers = {"Authorization": f"Bearer {self.api_token}"}


    def __call__(self, task: str, image_url: str = None, prefer_prompt_key: str = "", num_instructions: int = 10, 
                 return_all_annotations=False,select_topk: int = 3):

        prefer_prompt = PREFER_PROMPTS[prefer_prompt_key]

        # Compose the prompt depending on providing image or not
        if image_url:
            content = [
                {
                    "type": "text", 
                    "text": f"The attached image is an example image of the initial state of a robot that will perform the task: {task}.{prefer_prompt} Generate a diverse set of exactly {num_instructions} instructions."
                },
                {
                    "type": "image_url",
                    "image_url": {"url": image_url},
                },
            ]
        else:
            content = [
                {
                    "type": "text", 
                    "text": f"The robot will perform the task: {task}.{prefer_prompt} Generate a diverse set of exactly {num_instructions} instructions."
                },
            ]

        # ===== 修改的核心部分 =====
        # 用 DashScope（千问）替代原来的 GPT-4o
        # self.client 需在 __init__ 中已这样初始化：
        # self.client = OpenAI(api_key=os.getenv("DASHSCOPE_API_KEY"),
        #                      base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
        chat_completion = self.client.chat.completions.create(
            model=self.model,   # 如 "qwen2.5-vl-72b-instruct" 或 "qwen-plus"
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
            # n=self.num_rejection_samples,
            temperature=0.8,
            max_tokens=512,
        )

        # DashScope 返回格式与 OpenAI 一致
        # all_annotations: List[str] = []
        for choice in chat_completion.choices:
            # 获取模型输出的文本
            text = choice.message.content.strip()

            # 按行拆分成指令列表
            annotations = [line.strip("- ").strip() for line in text.split("\n") if line.strip()]
            # all_annotations.append(annotations)

        all_annotations = annotations

        all_sim: torch.Tensor = [self.embedding_model(annotations).mean().item() for annotations in all_annotations]

        
        top5_idx = np.argsort(all_sim)[:select_topk]          # 相似度最低的前 5 个索引
        lowest5_annotations = [all_annotations[i] for i in top5_idx]


        if return_all_annotations:
            return all_annotations[np.argmin(all_sim)], all_annotations

        return lowest5_annotations



# def _vlm_worker(task_and_links, examples):
#     task, links = task_and_links
#     embedding_model = CLIPEmbeddingModel("cuda")
#     red_team = EmbodiedRedTeamModelWithQwen2(embedding_model=embedding_model)
#     annotations = red_team(task, image_url=random.sample(links, k=1)[0], examples=examples[task])
#     return task, annotations

# def _lm_worker(task, examples):
#     embedding_model = CLIPEmbeddingModel("cuda")
#     red_team = EmbodiedRedTeamModelWithQwen2(embedding_model=embedding_model)
#     annotations = red_team(task, examples=examples.get(task, []))
#     return task, annotations

if __name__ == "__main__":
    # import argparse
    # parser = argparse.ArgumentParser(description="Generate instructions for red teaming")
    # parser.add_argument("--output_path", required=True, default="", type=str, help="Output directory of the instructions.")
    # parser.add_argument("--examples_path", type=str, help="YAML file for the previously generated task-annotation pairs")
    # parser.add_argument("--task_images", type=str, default="vlm_initial_state_links.json", help="YAML file of all tasks and image links")
    # parser.add_argument("--use_image", action="store_true", default=True, help="Include eimage or not")
    # parser.add_argument("--max_num_workers", type=int, default=8, help="Number of parallel workers")
    # args = parser.parse_args()

    test_task = "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate.".replace("_", " ")

    test_img_url = "https://huggingface.co/datasets/TBS2001/libero-init-frames/resolve/main/libero_init_frames/libero_spatial_task-0_img-0_frame0.png"

    embedding_model = CLIPEmbeddingModel("cuda")
    red_team = EmbodiedRedTeamModelWithQwen2(embedding_model=embedding_model)
    annotations = red_team(test_task, image_url=test_img_url, prefer_prompt_key="FRAME")

    print("生成的攻击指令：")
    print(annotations)
    
    # os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    # multiprocessing.set_start_method('spawn', force=True)
    # with open(args.task_images, "r") as f:
    #     vlm_task_to_links = json.load(f)

    # examples = {}
    # if args.examples_path and os.path.exists(args.examples_path):
    #     with open(args.examples_path, "r") as f:
    #         examples: Dict[str, List[str]] = yaml.safe_load(f)
        
    # with multiprocessing.Pool(args.max_num_workers) as pool:
    #     if args.use_image:
    #         results = list(tqdm(pool.imap(functools.partial(_vlm_worker, examples=examples), vlm_task_to_links.items()), total=len(vlm_task_to_links)))
    #     else:
    #         results = list(tqdm(pool.imap(functools.partial(_lm_worker, examples=examples), vlm_task_to_links.keys()), total=len(vlm_task_to_links)))
    
    # with open(args.output_path, "w") as f:
    #     yaml.dump({k: v for k, v in results}, f)

    # print(f"Save the output at: {args.output_path}")