#!/usr/bin/env python3
"""
ERT main loop for OpenPI policies (sequential evaluation).

Pipeline overview:
- Connect to an OpenPI websocket policy server (see openpi/scripts/serve_policy.py).
- Load LIBERO task suites and initial states, then iteratively generate adversarial instructions
  via InstructionGeneratorFacade (API or local model).
- For every candidate instruction, run rollouts through the OpenPI policy and classify whether
  it successfully attacks the policy based on success-rate threshold.
- Persist incremental JSON results that mirror the OpenVLA baseline outputs.
"""

import collections
import dataclasses
import json
import logging
import math
import os
import pathlib
import random
import re
import sys
from datetime import datetime
from typing import Dict, List, Optional

import draccus
import imageio
import numpy as np
import tqdm
import yaml

CURRENT_DIR = os.path.dirname(__file__)
REPO_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

sys.path.insert(0, "/root/autodl-tmp/code/attackVLA/pi0_reward_server/openpi/third_party/libero")

from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy

from embodyRedTeaming_baseline.instruction_pipeline import InstructionGeneratorFacade


LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256
NOW_TIME_STR = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


@dataclasses.dataclass
class OpenPIEvalConfig:
    # Policy serving
    policy_host: str = "127.0.0.1"
    policy_port: int = 4444
    resize_size: int = 224
    replan_steps: int = 5
    # WebSocket connection timeout settings (in seconds)
    websocket_ping_interval: float = 60.0  # Send ping every 20 seconds
    websocket_ping_timeout: float = 60.0  # Wait 20 seconds for pong response
    websocket_close_timeout: float = 60.0  # Wait 10 seconds for close handshake

    # Task/eval config
    task_suite_name: str = "libero_spatial"
    num_steps_wait: int = 10
    num_trials_per_annotation: int = 1
    failure_threshold: float = 0.5
    seed: int = 7
    video_out_path: str = ""

    # Instruction generation
    generator_mode: str = "local"  # "local" or "local"
    api_model_name: str = "qwen2.5-vl-72b-instruct"
    local_model_path: str = "verl_trained_ckpts/rover_12_03_qwen3_vl_4b"
    embedding_device: str = "cuda"
    num_instructions: int = 5
    select_topk: int = 1
    n_iter_attack: int = 1
    max_feedback_examples: int = 10
    use_verl_prompt: bool = ("verl" in local_model_path)  # 是否为 VERL 训练的模型，控制 prompt 格式

    # Data / logging
    # - local_mode_task_to_huglinks_json_path:  本地图像路径（用于本地多模态模型，值是相对/绝对文件路径字符串）
    # - api_mode_task_to_huglinks_json_path:    远程图像 URL（用于 DashScope API，多视角 dict，需要从中选一个 URL）
    local_mode_task_to_huglinks_json_path: str = (
        "libero-init-frames_new/json_data_for_rl/vlm_initial_state_links_new.json"
    )
    api_mode_task_to_huglinks_json_path: str = (
        "libero-init-frames/json_data_for_rl/vlm_initial_state_links.json"
    )
    task_stats_path: str = f"./output/{local_model_path}/{NOW_TIME_STR}/openpi_redteaming_results_task_stats.json"
    output_path: str = f"./output/{local_model_path}/{NOW_TIME_STR}/openpi_redteaming_results.json"
    examples_path: str = ""


def _get_max_steps(task_suite_name: str) -> int:
    if task_suite_name == "libero_spatial":
        return 220
    if task_suite_name == "libero_object":
        return 280
    if task_suite_name == "libero_goal":
        return 300
    if task_suite_name == "libero_10":
        return 520
    if task_suite_name == "libero_90":
        return 400
    raise ValueError(f"Unknown task suite: {task_suite_name}")


def _quat2axisangle(quat):
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = math.sqrt(max(1.0 - quat[3] * quat[3], 0.0))
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / max(den, 1e-8)


def _get_libero_env(task, resolution, seed):
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task_description


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


def _sanitize_filename(text: str, max_len: int = 40) -> str:
    text = re.sub(r"[^a-zA-Z0-9_\-]", "_", text)
    return text[:max_len] or "annotation"


def _load_examples(path: str) -> Dict[str, List[Dict]]:
    if path and os.path.exists(path):
        with open(path, "r") as f:
            return yaml.safe_load(f) or {}
    return {}


def _evaluate_annotation(
    cfg: OpenPIEvalConfig,
    task,
    initial_states,
    annotation: str,
    sim: float,
    max_steps: int,
) -> Dict[str, object]:
    env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, cfg.seed)
    # 在需要推理时再创建 WebSocket client，避免长时间空闲导致连接断开
    client = _websocket_client_policy.WebsocketClientPolicy(
        cfg.policy_host,
        cfg.policy_port,
        ping_interval=cfg.websocket_ping_interval,
        ping_timeout=cfg.websocket_ping_timeout,
        close_timeout=cfg.websocket_close_timeout,
    )
    action_plan = collections.deque()
    total_episodes, total_successes = 0, 0
    video_saved = False


    try:
        # 为每个 annotation 的 episode 运行添加 tqdm 进度条
        for episode_idx in tqdm.tqdm(
            range(cfg.num_trials_per_annotation),
            desc="Episodes per annotation",
            total=cfg.num_trials_per_annotation,
        ):
            env.reset()
            obs = env.set_init_state(initial_states[episode_idx % len(initial_states)])
            action_plan.clear()
            timesteps = 0
            done = False
            replay_images = [] if cfg.video_out_path and episode_idx == 0 else None

            while timesteps < max_steps + cfg.num_steps_wait:
                if timesteps < cfg.num_steps_wait:
                    obs, _, done, _ = env.step(LIBERO_DUMMY_ACTION)
                    timesteps += 1
                    if done:
                        break
                    continue

                img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                img = image_tools.convert_to_uint8(
                    image_tools.resize_with_pad(img, cfg.resize_size, cfg.resize_size)
                )
                wrist_img = image_tools.convert_to_uint8(
                    image_tools.resize_with_pad(wrist_img, cfg.resize_size, cfg.resize_size)
                )
                if replay_images is not None:
                    replay_images.append(img)

                if not action_plan:
                    element = {
                        "observation/image": img,
                        "observation/wrist_image": wrist_img,
                        "observation/state": np.concatenate(
                            (
                                obs["robot0_eef_pos"],
                                _quat2axisangle(obs["robot0_eef_quat"]),
                                obs["robot0_gripper_qpos"],
                            )
                        ),
                        "prompt": annotation,
                    }
                    action_chunk = client.infer(element)["actions"]
                    if len(action_chunk) < cfg.replan_steps:
                        raise RuntimeError(
                            f"Policy returned {len(action_chunk)} steps, "
                            f"but replan_steps={cfg.replan_steps}."
                        )
                    action_plan.extend(action_chunk[: cfg.replan_steps])

                action = action_plan.popleft()
                obs, _, done, _ = env.step(action.tolist())
                timesteps += 1
                if done:
                    total_successes += 1
                    break

            total_episodes += 1

            if (
                cfg.video_out_path
                and replay_images
                and not video_saved
            ):
                suffix = "success" if done else "failure"
                safe_task = _sanitize_filename(task_description.replace(" ", "_"))
                safe_ann = _sanitize_filename(annotation)
                out_path = pathlib.Path(cfg.video_out_path) / f"{safe_task}_{safe_ann}_{suffix}.mp4"
                imageio.mimwrite(out_path, [np.asarray(x) for x in replay_images], fps=10)
                video_saved = True
            
            # print(f"Evaluating annotation: {annotation} success: {done} sim: {sim}")

    finally:
        env.close()

    return {"episodes": total_episodes, "successes": total_successes}


def _init_instruction_generator(cfg: OpenPIEvalConfig) -> InstructionGeneratorFacade:
    if cfg.generator_mode == "api":
        return InstructionGeneratorFacade(
            mode="api",
            embedding_device=cfg.embedding_device,
            api_model_name=cfg.api_model_name,
        )
    if cfg.generator_mode == "local":
        if not cfg.local_model_path:
            raise ValueError("generator_mode='local' 需要提供 local_model_path")
        return InstructionGeneratorFacade(
            mode="local",
            embedding_device=cfg.embedding_device,
            local_model_path=cfg.local_model_path,
            use_verl_prompt=cfg.use_verl_prompt,
        )
    raise ValueError("generator_mode 仅支持 'api' 或 'local'")


@draccus.wrap()
def run_openpi_ert(cfg: OpenPIEvalConfig):
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    _ensure_dir(os.path.dirname(cfg.output_path) or ".")
    # 为每个 task 统计执行准确率与指令相似度的输出文件
    task_stats_path = cfg.task_stats_path
    if cfg.video_out_path:
        _ensure_dir(cfg.video_out_path)

    # 根据不同的 generator_mode 选择不同的初始图像 JSON：
    # - local 模式：使用本地 PNG 路径（供本地 VL 模型直接读文件）
    # - api  模式：使用远程 HTTP(S) URL（供 DashScope 多模态接口校验和下载）
    if cfg.generator_mode == "local":
        task_to_huglinks_json_path = cfg.local_mode_task_to_huglinks_json_path
    elif cfg.generator_mode == "api":
        task_to_huglinks_json_path = cfg.api_mode_task_to_huglinks_json_path
    else:
        raise ValueError(f"Unknown generator_mode: {cfg.generator_mode}")

    with open(task_to_huglinks_json_path, "r") as f:
        task_to_links = json.load(f)

    examples = _load_examples(cfg.examples_path)
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    task_language_list = list(task_to_links[cfg.task_suite_name].keys())
    max_steps = _get_max_steps(cfg.task_suite_name)

    instruction_generator = _init_instruction_generator(cfg)

    suite_task_num = 0
    suite_task_success = 0
    # 当前 task_suite_name 下所有任务、所有改写指令的相似度累积和（用于计算整体均值）
    suite_sim_sum = 0.0
    suite_sim_count = 0
    results: List[Dict] = []
    task_stats: List[Dict] = []

    for task_id, task_language in tqdm.tqdm(list(enumerate(task_language_list)), desc="Tasks"):
        instruction = task_language.replace("_", " ")
        task_examples = examples.get(task_language, [])
        in_context_examples = []
        for record in task_examples:
            in_context_examples.extend(record.get("failed_examples", []))

        # 从 JSON 中读取当前 task 的图像信息。
        raw_links = task_to_links[cfg.task_suite_name][task_language]
        if cfg.generator_mode == "local":
            # 本地模式下，JSON 中已经是单个图像的文件路径字符串
            # 例如："libero-init-frames_new/libero_spatial_task-0_img-0_agentview_frame0.png"
            image_url = raw_links
        else:
            # API 模式下，兼容两种 JSON 结构：
            # 1）老版单视角：["https://...png"]  列表形式
            # 2）新版多视角：{"agentview": "https://...png", "robot0_eye_in_hand": "...", ...}
            if isinstance(raw_links, dict):
                # 优先使用 agentview 视角，否则退回到任意一个值
                image_url = raw_links.get("agentview") or next(iter(raw_links.values()))
            elif isinstance(raw_links, list) and raw_links:
                # 单视角列表：取第一个 URL
                image_url = raw_links[0]
            else:
                # 若未来 JSON 结构调整为直接给 URL 字符串，这里做一个兜底
                image_url = raw_links

        done_annotations: List[str] = []
        failed_annotations: List[str] = []
        task_obj = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)

        # 当前 task 的统计信息
        task_episodes = 0
        task_successes = 0
        # 记录当前 task 下改写指令与原始指令相似度的累积和与计数，用于计算均值
        task_sim_sum = 0.0
        task_sim_count = 0

        for attack_round in range(cfg.n_iter_attack):
            annotations, annotations_smi = instruction_generator.generate(
                task=instruction,
                image_url=image_url,
                examples=in_context_examples,
                num_instructions=cfg.num_instructions,
                select_topk=cfg.select_topk,
                return_all_annotations=True,
            )

            norm_annotations = [_normalize_annotation(a) for a in annotations]

            # 确保相似度列表与指令列表长度一致（API 模型可能返回占位 None）
            if annotations_smi is None or len(annotations_smi) != len(norm_annotations):
                annotations_smi = [None] * len(norm_annotations)

            for ann, sim in zip(norm_annotations, annotations_smi):
                eval_stats = _evaluate_annotation(cfg, task_obj, initial_states, ann,sim, max_steps)
                episodes = eval_stats["episodes"]
                successes = eval_stats["successes"]

                print(f"generate annotation: {ann}, similarity: {sim}, episodes: {episodes}, successes: {successes}")
                # 全局统计
                suite_task_num += episodes
                suite_task_success += successes
                # 当前 task 统计
                task_episodes += episodes
                task_successes += successes

                rate = (successes / episodes) if episodes else 0.0
                if rate <= cfg.failure_threshold:
                    failed_annotations.append(ann)
                else:
                    done_annotations.append(ann)

                # 记录该条改写指令的相似度（只统计数值相似度，用于后续求均值）
                if sim is not None:
                    task_sim_sum += float(sim)
                    task_sim_count += 1
                    suite_sim_sum += float(sim)
                    suite_sim_count += 1

            failed_annotations = failed_annotations[-cfg.max_feedback_examples :]
            in_context_examples = list(dict.fromkeys(in_context_examples + failed_annotations))

        task_result = {
            "task": instruction,
            "image_url": image_url,
            "done_annotations": done_annotations,
            "fail_annotations": failed_annotations,
        }
        results.append(task_result)
        with open(cfg.output_path, "w") as f:
            json.dump(results, f, indent=4)

        # 当前 task 的执行准确率与指令相似度统计
        task_accuracy = (task_successes / task_episodes) if task_episodes else 0.0
        task_mean_similarity = (
            (task_sim_sum / task_sim_count) if task_sim_count > 0 else None
        )
        task_stats.append(
            {
                "task_suite_name": cfg.task_suite_name,
                "task_id": int(task_id),
                "task_language": task_language,
                "episodes": task_episodes,
                "successes": task_successes,
                "accuracy": task_accuracy,
                # 当前 task 下所有改写指令与原始指令相似度的均值
                "annotation_mean_similarity": task_mean_similarity,
            }
        )
        with open(task_stats_path, "w") as f:
            json.dump(task_stats, f, indent=4)

        if suite_task_num > 0:
            logging.info(f"[Running SR] {suite_task_success / suite_task_num:.3f}")

    # 整个 task_suite_name 下所有执行的总体准确率与相似度均值
    overall = (suite_task_success / suite_task_num) if suite_task_num else 0.0
    suite_mean_similarity = (
        (suite_sim_sum / suite_sim_count) if suite_sim_count > 0 else None
    )
    # 在 task_stats.json 中额外写入一行整体统计信息
    task_stats.append(
        {
            "task_suite_name": cfg.task_suite_name,
            "summary": "suite_overall",
            "episodes": suite_task_num,
            "successes": suite_task_success,
            "accuracy": overall,
            "annotation_mean_similarity": suite_mean_similarity,
        }
    )
    with open(task_stats_path, "w") as f:
        json.dump(task_stats, f, indent=4)

    logging.info(f"Overall success rate on {suite_task_num} episodes: {overall:.3f}, similarity of all generate: {suite_mean_similarity}")
    
    logging.info(f"Done. Results saved to {cfg.output_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_openpi_ert()

