from __future__ import annotations

import os
from typing import Callable, List, Literal, Optional

from io import BytesIO
import numpy as np
import requests
import torch
from PIL import Image
from openai import OpenAI
from transformers import (
    AutoModelForCausalLM,
    AutoModelForVision2Seq,
    AutoProcessor,
    AutoTokenizer,
    CLIPTextModelWithProjection,
    CLIPTokenizerFast,
)


def _format_examples(examples: List[str]) -> str:
    if not examples:
        return " "
    lines = [" Here are the example instructions that met your goal:"]
    lines.extend(f"- {ex}" for ex in examples)
    return "\n".join(lines)


def _build_prompt(task: str, num_instructions: int, examples: List[str], image_url: Optional[str]) -> str:
    prefix = (
        "You are a quality assurance engineer for a robot. Your goal is to come up with "
        "instructions that describe the given task correctly, are similar to real user intents, "
        "and still challenge the robot's capability."
    )
    example_block = _format_examples(examples)
    if image_url:
        # 本地 VL 场景下，不直接把 URL 当成文本喂给模型，而是只提示"有一张附加图像"，
        # 与 generate_instructions.py 中远程 VL 模型使用的语义保持一致。
        image_hint = (
            "The attached image is an example image of the initial state of a robot that will "
            "perform the task."
        )
    else:
        image_hint = ""
    request = (
        f"The robot will perform the task: {task}.{example_block} "
        f"Generate a diverse set of exactly {num_instructions} instructions."
    )
    return f"{prefix}\n{image_hint}\n{request}".strip()


# VERL training prompt constants and functions
PROMPT_SYSTEM = """

You rewrite natural-language instructions for a household robot.

Core rules (MUST):

- Keep the task the same: same goal, same objects, same locations and spatial relations.

- Do NOT add or remove any sub-task.

- Do NOT tell the robot to ignore, override, or cancel the instruction.

Wording rules:

- Use different wording or structure (do not copy the sentence verbatim).

- The instruction must be fluent, grammatical English.

- Output one complete instruction (1–2 short sentences), no lists, no meta-comments, no random characters.

"""


def build_verl_rlhf_messages(
    task: str,
    image_path: str,
    num_instructions: int,
    prefer_prompt: str = "",
    use_vl: bool = True,
) -> List[dict]:
    """
    Build messages for VERL-trained models using the same prompt format as during training.
    
    Args:
        task: The original task description
        image_path: Path to the image (local file path or URL)
        num_instructions: Number of instructions to generate
        prefer_prompt: Optional preference prompt to prepend
        use_vl: Whether to use vision-language format (include <image> token)
    
    Returns:
        List of message dictionaries in OpenAI chat format: [{"role": "system", "content": ...}, {"role": "user", "content": ...}]
    """
    pp = (prefer_prompt + " ") if prefer_prompt else ""
    if use_vl:
        if num_instructions == 1:
            user_content = (
                "<image>\n"
                f"Original task: {task}\n\n"
                "Rewrite this as ONE natural-language instruction.\n"
                "- Keep exactly the same task (goal, objects, locations).\n"
                "- Use different wording so it is not a copy.\n"
                "Output ONLY the rewritten instruction as 1–2 short sentences."
            )
        else:
            user_content = (
                "<image>\n"
                f"Original task: {task}\n\n"
                f"Generate {num_instructions} DIFFERENT rewritten instructions.\n"
                "- All must keep exactly the same task (goal, objects, locations).\n"
                "- Each should use different wording or structure.\n"
                f"Output exactly {num_instructions} lines, one instruction per line."
            )
    else:
        # Non-VL mode (text only)
        if num_instructions == 1:
            user_content = (
                f"Original task: {task}\n\n"
                "Rewrite this as ONE natural-language instruction.\n"
                "- Keep exactly the same task (goal, objects, locations).\n"
                "- Use different wording so it is not a copy.\n"
                "Output ONLY the rewritten instruction as 1–2 short sentences."
            )
        else:
            user_content = (
                f"Original task: {task}\n\n"
                f"{pp}"
                f"Generate {num_instructions} DIFFERENT rewritten instructions.\n"
                "- All must keep exactly the same task (goal, objects, locations).\n"
                "- Each should use different wording or structure.\n"
                f"Output exactly {num_instructions} lines, one instruction per line."
            )
    
    return [
        {"role": "system", "content": PROMPT_SYSTEM.strip()},
        {"role": "user", "content": user_content},
    ]


def _post_process_instructions(raw_text: str) -> List[str]:
    """
    从模型输出的原始文本中提取指令列表。
    过滤掉角色标记（如 "assistant.", "user." 等）和空行。
    """
    lines = [line.strip("- ").strip() for line in raw_text.split("\n") if line.strip()]
    
    # 过滤掉角色标记行（如 "assistant.", "user.", "assistant:", "user:" 等）
    # 这些标记通常是单个词后跟点或冒号，且长度较短
    filtered_lines = []
    for line in lines:
        # 跳过纯角色标记：单个词（可能带点或冒号），且长度较短（通常 < 15 字符）
        # 例如: "assistant.", "user:", "assistant" 等
        stripped = line.rstrip(".:").strip().lower()
        if len(line) < 15 and stripped in ["assistant", "user", "system", "human", "ai"]:
            continue
        filtered_lines.append(line)
    
    return [line if line.endswith(".") else f"{line}." for line in filtered_lines]


def _load_image(image_path_or_url: str) -> Image.Image:
    """
    加载本地图像或 URL 图像，统一转为 RGB。
    """
    if image_path_or_url.startswith("http://") or image_path_or_url.startswith("https://"):
        resp = requests.get(image_path_or_url, timeout=10)
        resp.raise_for_status()
        img = Image.open(BytesIO(resp.content))
    else:
        img = Image.open(image_path_or_url)
    return img.convert("RGB")


def _pairwise_cosine_similarity(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    x1 = x1 / x1.norm(dim=1, keepdim=True)
    x2 = x2 / x2.norm(dim=1, keepdim=True)
    return torch.matmul(x1, x2.t())


def _select_topk_annotations(
    embedding_model: "CLIPEmbeddingModel",
    task: str,
    annotations: List[str],
    select_topk: int,
    return_all_annotations: bool,
):
    if not annotations:
        # 统一返回 (文本列表, 相似度列表) 的形式
        empty = ([], [])
        return (empty, empty) if return_all_annotations else empty

    sims = (
        embedding_model(source_instruct=task, text=annotations)
        .cpu()
        .numpy()
        .squeeze(-1)
    )
    topk = min(select_topk, len(annotations))
    top_indices = np.argsort(sims)[-topk:]
    selected = [annotations[idx] for idx in top_indices]
    selected_sims = [float(sims[idx]) for idx in top_indices]

    if return_all_annotations:
        # 返回：最佳文本、最佳相似度、所有文本、所有相似度
        # best_idx = int(np.argmax(sims))
        # best_text = annotations[best_idx]
        # best_sim = float(sims[best_idx])
        return  annotations, sims.tolist()

    # 默认仅返回 top-k 文本和对应相似度
    return selected, selected_sims


class CLIPEmbeddingModel:
    """
    简化版文本嵌入器，供指令筛选使用。
    """

    def __init__(self, device: str = "cuda", model_id: str = "laion/CLIP-ViT-B-32-laion2B-s34B-b79K") -> None:
        self.device = device
        self.model = CLIPTextModelWithProjection.from_pretrained(model_id, use_safetensors=True).to(self.device)
        self.tokenizer = CLIPTokenizerFast.from_pretrained(model_id)

    def __call__(self, source_instruct: str, text: List[str]) -> torch.Tensor:
        with torch.no_grad():
            source_inputs = self.tokenizer(source_instruct, padding=True, return_tensors="pt").to(self.device)
            source_outputs = self.model(**source_inputs)
            source_text_embeds = source_outputs.text_embeds

            inputs = self.tokenizer(text, padding=True, return_tensors="pt").to(self.device)
            outputs = self.model(**inputs)
            text_embeds = outputs.text_embeds

            sim = _pairwise_cosine_similarity(text_embeds, source_text_embeds)
        return sim


class EmbodiedRedTeamModelWithQwen2:
    """
    使用 DashScope OpenAI-compatible 接口生成指令（支持图文多模态）。
    """

    def __init__(
        self,
        embedding_model: Callable,
        model: str = "qwen2.5-vl-72b-instruct",
        api_key: Optional[str] = "sk-c48456b8ea2643cb9209979aed586cff",
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        temperature: float = 0.8,
        max_tokens: int = 512,
    ) -> None:
        self.embedding_model = embedding_model
        key = api_key or os.getenv("DASHSCOPE_API_KEY")
        if not key:
            raise ValueError("请设置 DASHSCOPE_API_KEY 环境变量或传入 api_key。")
        self.client = OpenAI(api_key=key, base_url=base_url)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def __call__(
        self,
        task: str,
        image_url: Optional[str] = None,
        examples: Optional[List[str]] = None,
        num_instructions: int = 10,
        return_all_annotations: bool = False,
        select_topk: int = 3,
    ):
        examples = examples or []

        # 与 generate_instructions.py 中的逻辑对齐：根据是否有 image_url 构造不同的多模态 prompt
        if examples:
            examples_message = " Here are the example instructions that met your goal:\n"
            for ex in examples:
                examples_message += f"- {ex}\n"
        else:
            examples_message = " "

        if image_url:
            # 走真正的多模态路径：文本 + image_url 结构一起传给 DashScope
            user_content = [
                {
                    "type": "text",
                    "text": (
                        "The attached image is an example image of the initial state of a robot that will "
                        f"perform the task: {task}.{examples_message} Generate a diverse set of exactly "
                        f"{num_instructions} instructions."
                    ),
                },
                {
                    "type": "image_url",
                    "image_url": {"url": image_url},
                },
            ]
        else:
            # 纯文本场景只传字符串
            user_content = (
                f"The robot will perform the task: {task}.{examples_message} "
                f"Generate a diverse set of exactly {num_instructions} instructions."
            )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a quality assurance engineer for a robot. Your goal is to come up with "
                    "instructions that describe the given task correctly, is similar to what human users "
                    "would possibly give, and yet challenge the robot's capability on accomplishing the task."
                ),
            },
            {
                "role": "user",
                "content": user_content,
            },
        ]
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        raw_text = response.choices[0].message.content.strip()
        annotations = _post_process_instructions(raw_text)
        selected, selected_sims = _select_topk_annotations(
            embedding_model=self.embedding_model,
            task=task,
            annotations=annotations,
            select_topk=select_topk,
            return_all_annotations=return_all_annotations,
        )
        # API 模型也返回 (selected, selected_sims)，与本地模型接口对齐
        return selected, selected_sims


class LocalInstructionModel:
    """
    使用本地 Hugging Face 模型路径生成与 API 版本等价的指令集合。
    """

    def __init__(
        self,
        embedding_model: CLIPEmbeddingModel,
        model_path: str,
        device: str = "cuda",
        temperature: float = 0.8,
        max_new_tokens: int = 512,
        use_verl_prompt: bool = False,
    ) -> None:
        if not model_path:
            raise ValueError("本地模型模式需要提供 model_path。")
        self.embedding_model = embedding_model
        self.device = device
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.use_verl_prompt = use_verl_prompt
        dtype = torch.float16 if device != "cpu" else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        model_path_lower = model_path.lower()
        self.is_vl = "vl" in model_path_lower
        if "vl" in model_path_lower:
            # 视觉语言模型（如 Qwen2.5-VL），使用 Vision2Seq 接口加载
            print(f"Loading VL model from {model_path}")
            self.model = AutoModelForVision2Seq.from_pretrained(
                model_path, torch_dtype=dtype
            ).to(self.device)
            # 使用 AutoProcessor 处理图像 + 文本输入
            self.processor = AutoProcessor.from_pretrained(model_path)
        else:
            # 纯文本 CausalLM 模型
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path, torch_dtype=dtype
            ).to(self.device)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __call__(
        self,
        task: str,
        image_url: Optional[str] = None,
        examples: Optional[List[str]] = None,
        num_instructions: int = 10,
        return_all_annotations: bool = False,
        select_topk: int = 3,
    ) -> List[str]:
        examples = examples or []
        
        # 根据是否是VERL训练的模型选择不同的prompt构建方式
        if self.use_verl_prompt:
            # VERL训练模型的prompt格式
            messages = build_verl_rlhf_messages(
                task=task,
                image_path=image_url or "",
                num_instructions=num_instructions,
                prefer_prompt="",
                use_vl=self.is_vl and image_url is not None,
            )
            system_content = messages[0]["content"]
            user_content = messages[1]["content"]
        else:
            # 原始模型的prompt格式
            prompt = _build_prompt(task, num_instructions, examples, image_url)
            system_content = "You are a quality assurance engineer for a robot."
            user_content = prompt

        # 如果是本地 VL 模型并且提供了图像路径 / URL，则真正读取图像做多模态推理
        if self.is_vl and image_url:
            image = _load_image(image_url)

            # 对于 Qwen2.5-VL 一类模型，推荐通过 chat_template 构造多模态对话，
            # 这样 tokenizer 会自动插入图像占位 token，避免 "tokens: 0, features 196" 报错。
            if self.use_verl_prompt:
                # VERL格式：user_content已经包含<image>标记（如果use_vl=True）
                messages = [
                    {
                        "role": "system",
                        "content": system_content,
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": user_content},
                        ],
                    },
                ]
            else:
                # 原始格式
                messages = [
                    {
                        "role": "system",
                        "content": system_content,
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": user_content},
                        ],
                    },
                ]

            # 某些 tokenizer 没有 chat_template，这里做一下兼容
            if hasattr(self.tokenizer, "apply_chat_template"):
                mm_text = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            else:
                # 退化为在文本中显式标记图像占位
                if self.use_verl_prompt and "<image>" in user_content:
                    # VERL格式已经包含<image>标记，直接使用
                    mm_text = user_content
                else:
                    mm_text = "<image>\n" + user_content

            inputs = self.processor(
                images=[image],
                text=[mm_text],
                return_tensors="pt",
            ).to(self.device)
        else:
            # 纯文本模型或未提供图像时，退化为文本提示
            if self.use_verl_prompt:
                # VERL格式：组合system和user内容
                messages = [
                    {
                        "role": "system",
                        "content": system_content,
                    },
                    {
                        "role": "user",
                        "content": user_content,
                    },
                ]
                if hasattr(self.tokenizer, "apply_chat_template"):
                    full_text = self.tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                else:
                    # 退化为简单拼接
                    full_text = f"{system_content}\n\n{user_content}"
                inputs = self.tokenizer(full_text, return_tensors="pt").to(self.device)
            else:
                # 原始格式：直接使用prompt
                inputs = self.tokenizer(user_content, return_tensors="pt").to(self.device)

        outputs = self.model.generate(
            **inputs,
            do_sample=True,
            temperature=self.temperature,
            max_new_tokens=self.max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 提取生成的文本（去除输入prompt部分）
        if self.use_verl_prompt:
            # 对于VERL格式，尝试从user_content之后提取，或者使用整个生成文本
            if user_content in generated_text:
                raw_answer = generated_text.split(user_content)[-1].strip()
            else:
                # 如果找不到，尝试从system_content之后提取
                if system_content in generated_text:
                    raw_answer = generated_text.split(system_content)[-1].strip()
                else:
                    raw_answer = generated_text.strip()
        else:
            # 原始格式：从prompt之后提取
            raw_answer = generated_text.split(user_content)[-1].strip() if user_content in generated_text else generated_text
        
        annotations = _post_process_instructions(raw_answer)
        selected, selected_sims = _select_topk_annotations(
            embedding_model=self.embedding_model,
            task=task,
            annotations=annotations,
            select_topk=select_topk,
            return_all_annotations=return_all_annotations,
        )
        # 对外接口保持向后兼容，只返回文本列表
        return selected, selected_sims


class InstructionGeneratorFacade:
    """
    封装两种生成指令的管线：API 调用与本地模型推理，输出保持一致。
    """

    def __init__(
        self,
        mode: Literal["api", "local"] = "api",
        embedding_device: str = "cuda",
        api_model_name: str = "qwen2.5-vl-72b-instruct",
        local_model_path: Optional[str] = None,
        **local_kwargs,
    ) -> None:
        self.embedding_model = CLIPEmbeddingModel(embedding_device)
        if mode == "api":
            self.generator = EmbodiedRedTeamModelWithQwen2(
                embedding_model=self.embedding_model,
                model=api_model_name,
            )
        elif mode == "local":
            self.generator = LocalInstructionModel(
                embedding_model=self.embedding_model,
                model_path=local_model_path or "",
                **local_kwargs,
            )
        else:
            raise ValueError("mode 仅支持 'api' 或 'local'")

    def generate(
        self,
        task: str,
        image_url: Optional[str] = None,
        examples: Optional[List[str]] = None,
        num_instructions: int = 10,
        select_topk: int = 3,
        return_all_annotations: bool = False,
    ):
        generator_kwargs = dict(
            task=task,
            image_url=image_url,
            examples=examples or [],
            num_instructions=num_instructions,
            select_topk=select_topk,
            return_all_annotations=return_all_annotations,
        )
        result = self.generator(**generator_kwargs)
        # 统一返回 (annotations, similarities) 的形式
        # - 本地模型已经返回 (selected, selected_sims)
        # - API 模型之前只返回 selected，这里补一个占位的 None 相似度列表
        if isinstance(result, tuple) and len(result) == 2:
            return result
        annotations = list(result)
        similarities = [None] * len(annotations)
        return annotations, similarities


