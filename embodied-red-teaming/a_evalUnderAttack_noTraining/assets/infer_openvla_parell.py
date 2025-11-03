#!/usr/bin/env python3
# main.py (parallel evaluation per-annotation; quiet workers)

"""
ERT main loop (并行评测版):
- 加载任务->image links
- 调用 EmbodiedRedTeamModel 生成 candidate instructions
- 将 (task_id, annotation) 分发到多个 worker 并行评测 OpenVLA+LIBERO success_rate
- 按阈值归类为 done/fail，并增量写盘
- 仅主进程打印（worker 静音，除非开启 WORKER_DEBUG=1）
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
import contextlib

# ====== 路径 ======
sys.path.insert(0, "/home/baoshuntong/code/saftyEmbodyAI/embodied-red-teaming/LIBERO")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# ====== 时间戳 ======
now = datetime.now()
NOW_TIME_STR = now.strftime("%Y-%m-%d_%H-%M-%S")

# ====== 多进程（标准库，避免过早导入 torch）======
import multiprocessing as mp

# from cache_py.image_perturb import perturb_image  # 如需图像扰动，可在 worker 内按需启用


@dataclass
class GenerateConfig:
    # Model-specific parameters
    model_family: str = "openvla"
    pretrained_checkpoint: Union[str, Path] = "openvla/openvla-7b-finetuned-libero-spatial"
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    center_crop: bool = True

    # LIBERO environment
    task_suite_name: str = "libero_spatial"   # libero_spatial, libero_object, libero_goal, libero_10, libero_90
    num_steps_wait: int = 10
    num_trials_per_task: int = 3

    # Utils
    run_id_note: Optional[str] = None
    local_log_dir: str = "./experiments/logs"
    use_wandb: bool = False
    wandb_project: str = "redTeamingOpenvla"
    wandb_entity: str = "tongbs-sysu"
    seed: int = 7

    # user-customized
    is_save_video: bool = False
    save_folder: str = "./output_videos/"
    task_to_huglinks_json_path: str = "/home/baoshuntong/code/saftyEmbodyAI/embodied-red-teaming/libero-init-frames/json_data_for_rl/vlm_initial_state_links_new.json"
    examples_path: str = ""

    redTeaming_vlm_model: str = "HuggingFaceTB/SmolVLM-500M-Instruct"
    failure_threshold: float = 0.5
    output_path: str = f"./output/{NOW_TIME_STR}/redteaming_results.json"

    num_instructions: int = 5
    select_topk: int = 3
    prefer_prompt_key: str = "PRONOUN"

    whole_acc_log_path: str = "/home/baoshuntong/code/saftyEmbodyAI/embodied-red-teaming/logs/whole_acc.log"
    is_red_teaming_attack: bool = True
    custom_lora_path: Optional[str] = "/home/baoshuntong/code/saftyEmbodyAI/embodied-red-teaming/outputs/2025-10-22_11-43-33/verl_kto_vlm/lora_kto"

    # 并行新增
    num_workers: int = 2                # 建议 <= GPU 数
    devices: Optional[str] = "3,4"      # 逗号分隔，如 "0,1"
    worker_quiet: bool = True           # 静音 worker（只主进程打印）


def parse_task_and_links(cfg, task_to_links):
    task_suite_name = cfg.task_suite_name
    task_suite = task_to_links[task_suite_name]
    task_language_list = list(task_suite.keys())
    return task_language_list, task_suite


# ============== Worker 全局句柄（在子进程内常驻） ==============
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

    # 设备/渲染隔离
    os.environ["CUDA_VISIBLE_DEVICES"] = device_id
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ["EGL_DEVICE_ID"] = "0"

    # 现在再导入会触发 CUDA 的库
    import torch
    from LIBERO.libero.libero import benchmark
    from utils import eval_libero as _eval
    from openvla.experiments.robot.openvla_utils import get_processor as _gp
    from openvla.experiments.robot.robot_utils import get_model as _gm
    from openvla.experiments.robot.libero.libero_utils import get_libero_env as _gle

    _GET_BENCHMARK_DICT = benchmark.get_benchmark_dict
    _EVAL_LIBERO = _eval
    _GET_PROCESSOR = _gp
    _GET_MODEL = _gm
    _GET_LIBERO_ENV = _gle

    torch.cuda.set_device(0)  # 本地0号 = 物理 device_id

    _WORKER_CFG = GenerateConfig(**cfg_dict)
    _WORKER_PROCESSOR = _GET_PROCESSOR(_WORKER_CFG) if _WORKER_CFG.model_family == "openvla" else None
    _WORKER_MODEL = _GET_MODEL(_WORKER_CFG).eval().to("cuda")

    if os.environ.get("WORKER_DEBUG", "0") == "1":
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


def _worker_entry(rank: int, device_id: str, cfg_dict: Dict, job_q: mp.Queue, res_q: mp.Queue):
    """
    包一层入口：根据配置将 worker 的 stdout/stderr 静音（避免重复打印）。
    """
    quiet = True
    try:
        quiet = GenerateConfig(**cfg_dict).worker_quiet
    except Exception:
        pass

    if quiet and os.environ.get("WORKER_DEBUG", "0") != "1":
        with open(os.devnull, "w") as devnull, \
             contextlib.redirect_stdout(devnull), \
             contextlib.redirect_stderr(devnull):
            _worker_loop(rank, device_id, cfg_dict, job_q, res_q)
    else:
        _worker_loop(rank, device_id, cfg_dict, job_q, res_q)


# -------------- 主循环（并行版） --------------
@draccus.wrap()
def run_ert_loop(cfg: GenerateConfig):

    # 按套件名修正 checkpoint
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

    # —— 设备编排（先解析 devices，再限制父进程可见设备，避免碰到 GPU0）——
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

    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(device_list[:num_workers])
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("EGL_DEVICE_ID", "0")

    # 准备红队生成器（在主进程，放 CPU）
    from generate_intru_smolvlm import EmbodiedRedTeamModelWithSmolVLM, CLIPEmbeddingModel
    embedding_model = CLIPEmbeddingModel(device="cpu")
    red_team = EmbodiedRedTeamModelWithSmolVLM(
        embedding_model=embedding_model,
        model_path=cfg.redTeaming_vlm_model,
        custom_lora_path=cfg.custom_lora_path
    )

    # 任务名与链接
    task_language_list, task_links_suite = parse_task_and_links(cfg, task_to_links)

    # 启动 worker
    ctx = mp.get_context("spawn")
    job_q: mp.Queue = ctx.Queue(maxsize=2 * num_workers)
    res_q: mp.Queue = ctx.Queue()

    workers = []
    cfg_dict = asdict(cfg)
    for rk in range(num_workers):
        p = ctx.Process(
            target=_worker_entry,
            args=(rk, device_list[rk], cfg_dict, job_q, res_q),
            daemon=True
        )
        p.start()
        workers.append(p)

    # 统计量
    suite_task_num = 0
    suite_task_success = 0
    results: List[Dict] = []

    # 打印一次套件名
    print(f"Task suite: {cfg.task_suite_name}")

    # 逐任务
    for task_id, task_language in enumerate(task_language_list):
        instruction = task_language.replace("_", " ")
        print(f"\ntask {task_id+1} :  {instruction}")

        # 随机一张初始帧
        current_task_links = task_links_suite[task_language]
        image_url = random.choice(current_task_links)

        # 生成候选指令
        if cfg.is_red_teaming_attack:
            annotations = red_team(
                instruction,
                image_url=image_url,
                prefer_prompt_key=cfg.prefer_prompt_key,
                num_instructions=cfg.num_instructions,
                select_topk=cfg.select_topk
            )
        else:
            annotations = [instruction]

        # 规范化 & 打印一次
        norm_annotations = []
        for a in annotations:
            a = re.sub(r'^\s*\d+\.\s*', '', a).strip().strip('"')
            norm_annotations.append(a)
            print(f"Red team generated instruction: {a}")

        # 提交作业
        total_jobs = 0
        for a in norm_annotations:
            job_q.put({"task_id": task_id, "annotation": a})
            total_jobs += 1

        # 回收并统计
        done_anotations = []
        fail_anotations = []

        with tqdm(total=total_jobs, desc="Annotations", dynamic_ncols=True, leave=False) as pbar:
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
                    fail_anotations.append(annotation)
                else:
                    done_anotations.append(annotation)

        # 阶段性打印
        if suite_task_num > 0:
            print(f"\n=============== total success rate : {suite_task_success / suite_task_num:.3f}")

        # 组织并写盘
        current_task_result = {
            "image_url": image_url,
            "done_anotations": done_anotations,   # 保持原字段名（拼写）
            "fail_anotations": fail_anotations,   # 保持原字段名（拼写）
            "task": instruction,
        }
        results.append(current_task_result)
        os.makedirs(os.path.dirname(cfg.output_path), exist_ok=True)
        with open(cfg.output_path, "w") as fout:
            json.dump(results, fout, indent=4)

    # 结束并总结
    for _ in workers:
        job_q.put(None)
    for p in workers:
        p.join()

    all_success_rate = (suite_task_success / suite_task_num) if suite_task_num > 0 else 0.0
    print(f"Overall success rate on {suite_task_num} episodes: {all_success_rate:.3f}")

    curr_prefer_prompt_key_result = {
        cfg.task_suite_name: {cfg.prefer_prompt_key: all_success_rate}
    }
    os.makedirs(os.path.dirname(cfg.whole_acc_log_path), exist_ok=True)
    with open(cfg.whole_acc_log_path, "a") as fout:
        json.dump(curr_prefer_prompt_key_result, fout, indent=4)

    print(f"Done. Results saved to {cfg.output_path}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    run_ert_loop()
