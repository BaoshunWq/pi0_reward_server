#!/usr/bin/env python3
import dataclasses
import json
import logging
import os
import pathlib
import re
import sys
from typing import Dict, List

import draccus
import numpy as np
import tqdm
import yaml
import torch
from transformers import CLIPTextModelWithProjection, CLIPTokenizerFast

CURRENT_DIR = os.path.dirname(__file__)
REPO_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from embodyRedTeaming_baseline.instruction_pipeline import InstructionGeneratorFacade
from datetime import datetime
NOW_TIME_STR = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

@dataclasses.dataclass
class DiversityEvalConfig:
    task_suite_name: str = "libero_spatial"
    generator_mode: str = "local"
    api_model_name: str = "qwen2.5-vl-72b-instruct"
    local_model_path: str = "verl_trained_ckpts_vl/rover/merged_hf_model"  # Qwen/Qwen3-VL-4B-Instruct
    embedding_device: str = "cuda"
    num_instructions: int = 5
    select_topk: int = 1
    n_iter_attack: int = 1
    use_verl_prompt: bool = ("verl" in local_model_path)
    local_mode_task_to_huglinks_json_path: str = "libero-init-frames_new/json_data_for_rl/vlm_initial_state_links_new.json"
    api_mode_task_to_huglinks_json_path: str = "libero-init-frames/json_data_for_rl/vlm_initial_state_links.json"
    examples_path: str = ""
    output_path: str = f"./output/{local_model_path}/diversity_eval_results_{NOW_TIME_STR}.json"
    task_stats_path: str = f"./output/{local_model_path}/diversity_eval_task_stats_{NOW_TIME_STR}.json"


def _ensure_dir(path: str) -> None:
    if path:
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)


def _normalize_annotation(text: str) -> str:
    text = text.strip()
    match = re.match(r'^\s*\d+\.\s*"(.*?)"\s*$', text)
    if match:
        text = match.group(1)
    else:
        text = re.sub(r"^\s*\d+\.\s*", "", text).strip().strip('"')
    return text


def _load_examples(path: str) -> Dict[str, List[Dict]]:
    if path and os.path.exists(path):
        with open(path, "r") as f:
            return yaml.safe_load(f) or {}
    return {}


_CLIP_TEXT_CACHE: Dict[str, object] = {}


def _get_clip_text_model(model_id: str, device: str):
    key = (model_id, device)
    cached = _CLIP_TEXT_CACHE.get(key)
    if cached is None:
        model = CLIPTextModelWithProjection.from_pretrained(model_id, use_safetensors=True).to(device)
        tokenizer = CLIPTokenizerFast.from_pretrained(model_id)
        _CLIP_TEXT_CACHE[key] = (model, tokenizer)
    return _CLIP_TEXT_CACHE[key]


def _compute_annotations_pairwise_similarity(
    annotations: List[str],
    device: str = "cuda",
    model_id: str = "laion/CLIP-ViT-B-32-laion2B-s34B-b79K",
):
    if not annotations:
        return np.zeros((0, 0)), None, None
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    model, tokenizer = _get_clip_text_model(model_id, device)
    with torch.no_grad():
        inputs = tokenizer(annotations, padding=True, return_tensors="pt").to(device)
        outputs = model(**inputs)
        embeds = outputs.text_embeds
        embeds = embeds / embeds.norm(dim=1, keepdim=True)
        sim = torch.matmul(embeds, embeds.t())
    n = sim.shape[0]
    if n <= 1:
        return sim.detach().cpu().numpy(), None, None
    total = sim.sum().item()
    diag = sim.diag().sum().item()
    mean_off = (total - diag) / (n * (n - 1))
    diversity = 1.0 - mean_off
    return sim.detach().cpu().numpy(), mean_off, diversity


def _init_instruction_generator(cfg: DiversityEvalConfig) -> InstructionGeneratorFacade:
    if cfg.generator_mode == "api":
        return InstructionGeneratorFacade(
            mode="api",
            embedding_device=cfg.embedding_device,
            api_model_name=cfg.api_model_name,
        )
    if cfg.generator_mode == "local":
        if not cfg.local_model_path:
            raise ValueError("local_model_path required for local mode")
        return InstructionGeneratorFacade(
            mode="local",
            embedding_device=cfg.embedding_device,
            local_model_path=cfg.local_model_path,
            use_verl_prompt=cfg.use_verl_prompt,
        )
    raise ValueError("generator_mode must be 'api' or 'local'")


@draccus.wrap()
def run_openpi_diversity(cfg: DiversityEvalConfig):
    _ensure_dir(os.path.dirname(cfg.output_path) or ".")
    _ensure_dir(os.path.dirname(cfg.task_stats_path) or ".")

    if cfg.generator_mode == "local":
        task_to_huglinks_json_path = cfg.local_mode_task_to_huglinks_json_path
    elif cfg.generator_mode == "api":
        task_to_huglinks_json_path = cfg.api_mode_task_to_huglinks_json_path
    else:
        raise ValueError(cfg.generator_mode)

    with open(task_to_huglinks_json_path, "r") as f:
        task_to_links = json.load(f)

    examples = _load_examples(cfg.examples_path)
    task_language_list = list(task_to_links[cfg.task_suite_name].keys())

    instruction_generator = _init_instruction_generator(cfg)
    results: List[Dict] = []
    task_stats: List[Dict] = []

    suite_sim_sum = 0.0
    suite_sim_count = 0
    suite_pair_mean_sum = 0.0
    suite_pair_mean_count = 0
    suite_div_sum = 0.0
    suite_div_count = 0

    for task_language in tqdm.tqdm(task_language_list, desc="Tasks"):
        instruction = task_language.replace("_", " ")
        task_examples = examples.get(task_language, [])
        in_context_examples = []
        for record in task_examples:
            in_context_examples.extend(record.get("failed_examples", []))

        raw_links = task_to_links[cfg.task_suite_name][task_language]
        if cfg.generator_mode == "local":
            image_url = raw_links
        else:
            if isinstance(raw_links, dict):
                image_url = raw_links.get("agentview") or next(iter(raw_links.values()))
            elif isinstance(raw_links, list) and raw_links:
                image_url = raw_links[0]
            else:
                image_url = raw_links

        task_pair_mean_sum = 0.0
        task_pair_mean_count = 0
        task_div_sum = 0.0
        task_div_count = 0
        task_src_sim_sum = 0.0
        task_src_sim_count = 0

        latest_annotations: List[str] = []
        latest_src_sims: List[float] = []

        for _ in range(cfg.n_iter_attack):
            annotations, annotations_smi = instruction_generator.generate(
                task=instruction,
                image_url=image_url,
                examples=in_context_examples,
                num_instructions=cfg.num_instructions,
                select_topk=cfg.select_topk,
                return_all_annotations=True,
            )
            norm_annotations = [_normalize_annotation(a) for a in annotations]
            _, mean_pairwise, diversity = _compute_annotations_pairwise_similarity(
                norm_annotations, device=cfg.embedding_device
            )
            if annotations_smi is None or len(annotations_smi) != len(norm_annotations):
                annotations_smi = [None] * len(norm_annotations)
            valid_src_sims = [float(s) for s in annotations_smi if s is not None]
            if valid_src_sims:
                task_src_sim_sum += float(np.mean(valid_src_sims))
                task_src_sim_count += 1
            if mean_pairwise is not None:
                task_pair_mean_sum += float(mean_pairwise)
                task_pair_mean_count += 1
            if diversity is not None:
                task_div_sum += float(diversity)
                task_div_count += 1
            latest_annotations = norm_annotations
            latest_src_sims = [float(s) if s is not None else None for s in annotations_smi]

        task_pair_mean = (task_pair_mean_sum / task_pair_mean_count) if task_pair_mean_count > 0 else None
        task_diversity = (task_div_sum / task_div_count) if task_div_count > 0 else None
        task_mean_similarity_to_source = (
            (task_src_sim_sum / task_src_sim_count) if task_src_sim_count > 0 else None
        )

        if task_pair_mean is not None:
            suite_pair_mean_sum += float(task_pair_mean)
            suite_pair_mean_count += 1
        if task_diversity is not None:
            suite_div_sum += float(task_diversity)
            suite_div_count += 1
        if task_mean_similarity_to_source is not None:
            suite_sim_sum += float(task_mean_similarity_to_source)
            suite_sim_count += 1

        task_result = {
            "task": instruction,
            "image_url": image_url,
            "annotations": latest_annotations,
            "annotation_similarity_to_source": latest_src_sims,
            "annotation_pairwise_mean_similarity": task_pair_mean,
            "annotation_diversity": task_diversity,
            "annotation_mean_similarity_to_source": task_mean_similarity_to_source,
        }
        results.append(task_result)
        with open(cfg.output_path, "w") as f:
            json.dump(results, f, indent=4)

        task_stats.append(
            {
                "task_suite_name": cfg.task_suite_name,
                "task_language": task_language,
                "annotation_pairwise_mean_similarity": task_pair_mean,
                "annotation_diversity": task_diversity,
                "annotation_mean_similarity_to_source": task_mean_similarity_to_source,
            }
        )
        with open(cfg.task_stats_path, "w") as f:
            json.dump(task_stats, f, indent=4)

    suite_mean_pairwise = (suite_pair_mean_sum / suite_pair_mean_count) if suite_pair_mean_count > 0 else None
    suite_mean_diversity = (suite_div_sum / suite_div_count) if suite_div_count > 0 else None
    suite_mean_similarity_to_source = (suite_sim_sum / suite_sim_count) if suite_sim_count > 0 else None

    task_stats.append(
        {
            "task_suite_name": cfg.task_suite_name,
            "summary": "suite_overall",
            "annotation_pairwise_mean_similarity": suite_mean_pairwise,
            "annotation_mean_diversity": suite_mean_diversity,
            "annotation_mean_similarity_to_source": suite_mean_similarity_to_source,
        }
    )
    with open(cfg.task_stats_path, "w") as f:
        json.dump(task_stats, f, indent=4)

    logging.info(
        f"Suite pairwise={suite_mean_pairwise}, diversity={suite_mean_diversity}, source_sim={suite_mean_similarity_to_source}"
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_openpi_diversity()

