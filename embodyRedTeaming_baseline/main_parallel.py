#!/usr/bin/env python3
# main.py (parallel evaluation per-annotation & per-round)

"""
ERT main loop (并行评测版):
- 加载任务->image links
- 用 EmbodiedRedTeamModel 逐任务/逐轮生成对抗指令
- 每一轮把 (task_id, annotation) 分发给多个 worker 并行评测 OpenVLA+LIBERO 成功率
- 用该轮失败样本更新 in_context_examples，再开始下一轮
- 任务完成后增量落盘 JSON
"""

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

import os
import yaml
import json
import random
from tqdm import tqdm
from dataclasses import dataclass, asdict
import draccus
from typing import Optional, Union, Dict, List
from pathlib import Path
import re
from datetime import datetime
import sys
import gc

# ====== 路径 ======
sys.path.insert(0, "/home/baoshuntong/code/saftyEmbodyAI/embodied-red-teaming/LIBERO")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# ====== 时间戳 ======
now = datetime.now()
NOW_TIME_STR = now.strftime("%Y-%m-%d_%H-%M-%S")

# ====== 多进程（使用标准库，避免过早导入 torch）======
import multiprocessing as mp


@dataclass
class GenerateConfig:
    # Model
    model_family: str = "openvla"
    pretrained_checkpoint: Union[str, Path] = "openvla/openvla-7b-finetuned-libero-spatial"
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    center_crop: bool = True

    # LIBERO
    task_suite_name: str = "libero_spatial"
    num_steps_wait: int = 10
    num_trials_per_task: int = 5

    # Utils
    run_id_note: Optional[str] = None
    local_log_dir: str = "./experiments/logs"
    use_wandb: bool = False
    wandb_project: str = "redTeamingOpenvla"
    wandb_entity: str = "tongbs-sysu"
    seed: int = 7

    # Custom
    is_save_video: bool = False
    task_to_huglinks_json_path: str = "libero-init-frames/json_data_for_rl/vlm_initial_state_links.json"
    redTeaming_vlm_model: str = "qwen2.5-vl-72b-instruct"
    num_rejection_samples: int = 5
    num_instructions: int = 10
    failure_threshold: float = 0.5
    max_feedback_examples: int = 10
    output_path: str = f"./output/{NOW_TIME_STR}/baseline_redteaming_results.json"
    examples_path: str = ""
    n_iter_attack: int = 1

    # 并行新增
    num_workers: int = 2                # 并行 worker 数（建议 <= GPU 数）
    devices: Optional[str] = "3,4"      # 逗号分隔GPU列表，如 "0,1"；为空则沿用 CUDA_VISIBLE_DEVICES 或默认 "0"


def parse_task_and_links(cfg, task_to_links):
    task_suite = task_to_links[cfg.task_suite_name]
    task_language_list = list(task_suite.keys())
    return task_language_list, task_suite


# ============== Worker 全局（在子进程内常驻） ==============
_WORKER_CFG = None
_WORKER_MODEL = None
_WORKER_PROCESSOR = None
_GET_BENCHMARK_DICT = None
_GET_PROCESSOR = None
_GET_MODEL = None
_GET_LIBERO_ENV = None
_EVAL_LIBERO = None


def _init_worker(cfg_dict: Dict, device_id: str):
    """
    子进程初始化：限制可见 GPU 为单张卡；懒加载处理器与模型。
    必须在 import torch/libero/openvla 之前设置环境变量！
    """
    global _WORKER_CFG, _WORKER_MODEL, _WORKER_PROCESSOR
    global _GET_BENCHMARK_DICT, _GET_PROCESSOR, _GET_MODEL, _GET_LIBERO_ENV, _EVAL_LIBERO

    # 限定该进程只见一张 GPU（device_id）
    os.environ["CUDA_VISIBLE_DEVICES"] = device_id
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ["EGL_DEVICE_ID"] = "0"

    # 现在再导入会触发 CUDA 的库（确保环境已生效）
    import torch
    from libero.libero import benchmark
    from a_evalUnderAttack_noTraining.utils import eval_libero as _eval
    from openvla.experiments.robot.openvla_utils import get_processor as _gp
    from openvla.experiments.robot.robot_utils import get_model as _gm
    from openvla.experiments.robot.libero.libero_utils import get_libero_env as _gle

    # 缓存函数引用，供 _eval_one 使用
    _GET_BENCHMARK_DICT = benchmark.get_benchmark_dict
    _EVAL_LIBERO = _eval
    _GET_PROCESSOR = _gp
    _GET_MODEL = _gm
    _GET_LIBERO_ENV = _gle

    torch.cuda.set_device(0)  # 本地0号 = 物理 device_id

    _WORKER_CFG = GenerateConfig(**cfg_dict)
    _WORKER_PROCESSOR = _GET_PROCESSOR(_WORKER_CFG) if _WORKER_CFG.model_family == "openvla" else None
    _WORKER_MODEL = _GET_MODEL(_WORKER_CFG).eval().to("cuda")

    print(f"[worker pid={os.getpid()}] visible={os.environ['CUDA_VISIBLE_DEVICES']}, "
          f"count={torch.cuda.device_count()}, "
          f"name={torch.cuda.get_device_name(0)}")


def _eval_one(job: Dict):
    """
    在 worker 内评测单条 (task_id, annotation)。
    """
    global _WORKER_CFG, _WORKER_MODEL, _WORKER_PROCESSOR
    global _GET_BENCHMARK_DICT, _GET_LIBERO_ENV, _EVAL_LIBERO

    task_id = job["task_id"]
    annotation = job["annotation"]

    print(f"[worker pid={os.getpid()}] Evaluating task_id={task_id} annotation='{annotation}'")

    # 重建 suite/task/env
    benchmark_dict = _GET_BENCHMARK_DICT()
    task_suite = benchmark_dict[_WORKER_CFG.task_suite_name]()
    task = task_suite.get_task(task_id)
    initial_states = task_suite.get_task_init_states(task_id)
    env, _ = _GET_LIBERO_ENV(task, _WORKER_CFG.model_family, resolution=256)

    try:
        task_episodes, task_successes = _EVAL_LIBERO(
            _WORKER_CFG, _WORKER_MODEL, _WORKER_PROCESSOR, env, initial_states, annotation
        )
    finally:
        env.close()
        del env
        import torch
        torch.cuda.synchronize()
        gc.collect(); torch.cuda.empty_cache()

    return {
        "task_id": task_id,
        "annotation": annotation,
        "episodes": task_episodes,
        "successes": task_successes,
    }


def _worker_loop(rank: int, device_id: str, cfg_dict: Dict, job_q: mp.Queue, res_q: mp.Queue):
    _init_worker(cfg_dict, device_id)
    while True:
        job = job_q.get()
        if job is None:
            break
        try:
            out = _eval_one(job)
            res_q.put(out)
        except Exception as e:
            res_q.put({
                "task_id": job.get("task_id", -1),
                "annotation": job.get("annotation", ""),
                "episodes": 0,
                "successes": 0,
                "error": repr(e),
            })


# -------------- 主循环（并行版） --------------
@draccus.wrap()
def run_ert_loop(cfg: GenerateConfig):

    # 根据套件名修正 openvla checkpoint（与原逻辑一致）
    if cfg.task_suite_name == "libero_spatial":
        cfg.pretrained_checkpoint = "openvla/openvla-7b-finetuned-libero-spatial"
    elif cfg.task_suite_name == "libero_object":
        cfg.pretrained_checkpoint = "openvla/openvla-7b-finetuned-libero-object"
    elif cfg.task_suite_name == "libero_goal":
        cfg.pretrained_checkpoint = "openvla/openvla-7b-finetuned-libero-goal"
    elif cfg.task_suite_name == "libero_10":
        cfg.pretrained_checkpoint = "openvla/openvla-7b-finetuned-libero-10"
    else:
        raise Exception("Unknown task_suite_name")

    # 读取任务->首帧链接
    with open(cfg.task_to_huglinks_json_path, "r") as f:
        task_to_links = json.load(f)

    # 读取历史样例
    examples = {}
    if cfg.examples_path and os.path.exists(cfg.examples_path):
        with open(cfg.examples_path, "r") as f:
            examples = yaml.safe_load(f) or {}

    # —— 并行编排 ——（先解析 devices，再把父进程也限制到这些卡，避免误占 GPU0）
    if cfg.devices and cfg.devices.strip():
        device_list = [d.strip() for d in cfg.devices.split(",") if d.strip()]
        print(f"[Parallel Eval] Using devices from cfg.devices: {device_list}")
    else:
        env_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
        if env_devices:
            device_list = [d.strip() for d in env_devices.split(",") if d.strip()]
        else:
            device_list = ["0"]
    num_workers = max(1, min(cfg.num_workers, len(device_list)))
    print(f"[Parallel Eval] Using {num_workers} worker(s) on device(s): {device_list[:num_workers]}")

    # 关键：让父进程也只“看见”你选择的那几张卡（比如 3,4），防止父进程碰到物理 GPU0
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(device_list[:num_workers])
    os.environ.setdefault("EGL_DEVICE_ID", "0")

    # 红队生成器（父进程放 CPU，避免占显存）
    from embodyRedTeaming_baseline.generate_instructions import EmbodiedRedTeamModelWithQwen2, CLIPEmbeddingModel
    embedding_model = CLIPEmbeddingModel(device="cpu")
    red_team = EmbodiedRedTeamModelWithQwen2(
        embedding_model=embedding_model,
        model=cfg.redTeaming_vlm_model,
        num_rejection_samples=cfg.num_rejection_samples
    )

    # 任务名与链接
    task_language_list, task_links_suite = parse_task_and_links(cfg, task_to_links)

    ctx = mp.get_context("spawn")
    job_q: mp.Queue = ctx.Queue(maxsize=2 * num_workers)
    res_q: mp.Queue = ctx.Queue()

    workers = []
    cfg_dict = asdict(cfg)
    for rk in range(num_workers):
        p = ctx.Process(
            target=_worker_loop,
            args=(rk, device_list[rk], cfg_dict, job_q, res_q),
            daemon=True
        )
        p.start()
        workers.append(p)

    # 统计量
    suite_task_num = 0
    suite_task_success = 0
    results: List[Dict] = []

    print(f"Task suite: {cfg.task_suite_name}")

    # ========= 逐任务进行（保持原本“按轮反馈”的语义）=========
    for task_id, task_language in tqdm(list(enumerate(task_language_list)), desc="Tasks"):
        instruction = task_language.replace("_", " ")

        # 历史失败样例（用于红队 in-context）
        task_examples = examples.get(task_language, [])
        if task_examples:
            in_context_examples = []
            for task_example in task_examples:
                in_context_examples.extend(task_example.get('failed_examples', []))
        else:
            in_context_examples = []

        # 记录一张首帧
        current_task_links = task_links_suite[task_language]
        image_url = random.choice(current_task_links)

        # per-task 汇总（最终写盘）
        done_results = []
        failed_examples = []

        # —— 多轮攻击（每一轮：生成→并行评测→更新失败样例）——
        for attack_round in range(cfg.n_iter_attack):
            # 生成候选指令（主进程）
            annotations = red_team(
                instruction,
                image_url=image_url,
                examples=in_context_examples,
                num_instructions=cfg.num_instructions,
                select_topk=5
            )
            # 规范化
            norm_annotations = []
            for a in annotations:
                # 兼容 `1. "text"` 或前缀编号
                m = re.search(r'^\s*\d+\.\s*"(.*?)"\s*$', a)
                if m:
                    a = m.group(1)
                else:
                    a = re.sub(r'^\s*\d+\.\s*', '', a).strip()
                    a = a.strip('"')
                norm_annotations.append(a)

            # 提交该轮所有作业
            total_jobs = 0
            for a in norm_annotations:
                job_q.put({"task_id": task_id, "annotation": a})
                total_jobs += 1

            # 回收该轮结果
            round_failed = []
            round_done = []

            with tqdm(total=total_jobs, desc=f"Eval task {task_id} round {attack_round+1}", dynamic_ncols=True, leave=False) as pbar:
                finished = 0
                while finished < total_jobs:
                    out = res_q.get()
                    finished += 1
                    pbar.update(1)

                    if "error" in out and out["error"]:
                        print(f"[Worker ERROR][task {out['task_id']}] {out['error']}")
                        out.setdefault("episodes", 0)
                        out.setdefault("successes", 0)

                    episodes = out["episodes"]
                    successes = out["successes"]
                    annotation = out["annotation"]

                    suite_task_num += episodes
                    suite_task_success += successes

                    rate = (successes / episodes) if episodes > 0 else 0.0
                    if rate <= cfg.failure_threshold:
                        round_failed.append(annotation)
                    else:
                        round_done.append(annotation)

            # 更新 in-context（限长 & 去重）
            failed_examples.extend(round_failed)
            done_results.extend(round_done)
            if len(failed_examples) > cfg.max_feedback_examples:
                failed_examples = failed_examples[-cfg.max_feedback_examples:]
            # 组合并去重（保留近端失败优先）
            in_context_examples = list(dict.fromkeys(in_context_examples + round_failed))

        # —— 该任务完成：写盘一次 —— 
        task_results = {
            "task": instruction,
            "image_url": image_url,
            "done_annotations": done_results,
            "fail_annotations": failed_examples,
        }
        results.append(task_results)

        os.makedirs(os.path.dirname(cfg.output_path) or ".", exist_ok=True)
        with open(cfg.output_path, "w") as fout:
            json.dump(results, fout, indent=4)

        if suite_task_num > 0:
            print(f"[Running SR] {suite_task_success / suite_task_num:.3f}")

    # 关闭 worker
    for _ in workers:
        job_q.put(None)
    for p in workers:
        p.join()

    # 总结
    overall = (suite_task_success / suite_task_num) if suite_task_num > 0 else 0.0
    print(f"Overall success rate on {suite_task_num} episodes: {overall:.3f}")
    print(f"Done. Results saved to {cfg.output_path}")


if __name__ == "__main__":
    # 仅在主进程设置 start_method
    mp.set_start_method("spawn", force=True)
    run_ert_loop()