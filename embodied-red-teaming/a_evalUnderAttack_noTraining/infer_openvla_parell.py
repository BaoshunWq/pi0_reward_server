#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_worker.py
OS-level parallel shard: evaluate a subset of task indices on ONE physical GPU.

- No mp.Manager / cross-process Queues.
- Per rollout uses a short-lived spawn child (robust wrt EGL/GL/Vulkan contexts).
- Hard-disables Flash-Attn / xFormers and forces SDPA.
- Strictly delay heavy imports until after we pin the GPU + EGL env.
"""

import os, re, json, time, gc, argparse
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, "/home/baoshuntong/code/saftyEmbodyAI/embodied-red-teaming/LIBERO")

NOW = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# -------------------- Env helpers (no heavy imports here) --------------------
def init_device_env(physical_gpu_id: int):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(physical_gpu_id)
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["PYOPENGL_PLATFORM"] = "egl"
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(physical_gpu_id)
    os.environ["EGL_DEVICE_ID"] = "0"
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

def hard_disable_flash_attn_before_import():
    os.environ["FLASH_ATTENTION"] = "0"
    os.environ["OPENVLA_FLASH_ATTENTION"] = "0"
    os.environ["ATTN_IMPL"] = "sdpa"
    os.environ["XFORMERS_DISABLED"] = "1"
    try:
        import importlib
        iu = importlib.import_module("transformers.utils.import_utils")
        def _false(*args, **kwargs): return False
        iu.is_flash_attn_2_available = _false
    except Exception:
        pass

def parse_range(spec: str) -> List[int]:
    """e.g., '0-3,5,7-9' -> [0,1,2,3,5,7,8,9]"""
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part: continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))

def choose_openvla_ckpt(task_suite_name: str) -> str:
    table = {
        "libero_spatial": "openvla/openvla-7b-finetuned-libero-spatial",
        "libero_object":  "openvla/openvla-7b-finetuned-libero-object",
        "libero_goal":    "openvla/openvla-7b-finetuned-libero-goal",
        "libero_10":      "openvla/openvla-7b-finetuned-libero-10",
    }
    if task_suite_name not in table:
        raise ValueError(f"Unknown task_suite_name: {task_suite_name}")
    return table[task_suite_name]

# -------------------- Child rollout (short-lived) --------------------
def eval_once_child_entry(q, cfg, physical_gpu_id: int, job: dict):
    """Top-level target for spawn child; must live at module scope (picklable)."""
    try:
        init_device_env(physical_gpu_id)
        hard_disable_flash_attn_before_import()

        import torch, gc
        torch.cuda.set_device(0)

        # delay imports until GPU pinned
        from libero.libero import benchmark
        from openvla.experiments.robot.openvla_utils import get_processor
        from openvla.experiments.robot.robot_utils import get_model
        from openvla.experiments.robot.libero.libero_utils import get_libero_env
        from utils import eval_libero

        # Minimal config object that matches OpenVLA expectations
        from types import SimpleNamespace
        _cfg = SimpleNamespace(
            # —— OpenVLA 基本配置 —— #
            model_family="openvla",
            pretrained_checkpoint=cfg["pretrained_checkpoint"],
            load_in_8bit=False,
            load_in_4bit=False,
            center_crop=True,
            attn_implementation="sdpa",  # 明示关闭 Flash-Attn
            device="cuda",

            # —— 任务/日志字段（有些 util/日志会访问）—— #
            task_suite_name=job["task_suite_name"],
            is_save_video=False,   # 如需存视频，可改为 cfg 里传入的值
            save_folder="",        # 同上

            # —— 评估相关 —— #
            num_steps_wait=cfg["num_steps_wait"],
            num_trials_per_task=cfg["num_trials_per_task"],
            seed = 0

        )


        # Load once per child (isolates GL/CUDA state)
        processor = get_processor(_cfg)
        openvla_model = get_model(_cfg).eval().to("cuda")

        # Force SDPA where possible
        try:
            lm = getattr(openvla_model, "language_model", None) or getattr(openvla_model, "lm", None)
            if lm and hasattr(lm, "config") and hasattr(lm.config, "attn_implementation"):
                lm.config.attn_implementation = "sdpa"
            if hasattr(openvla_model, "config") and hasattr(openvla_model.config, "attn_implementation"):
                openvla_model.config.attn_implementation = "sdpa"
        except Exception:
            pass

        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()

        # Build env for this task
        benchmark_dict = benchmark.get_benchmark_dict()
        task_suite = benchmark_dict[job["task_suite_name"]]()
        task = task_suite.get_task(job["task_id"])
        init_states = task_suite.get_task_init_states(job["task_id"])
        env, _ = get_libero_env(task, _cfg.model_family, resolution=256)

        # Run evaluation
        episodes, successes = eval_libero(
            _cfg, openvla_model, processor, env, init_states, job["annotation"]
        )

        env.close()
        del env, openvla_model, processor
        torch.cuda.synchronize(); gc.collect(); torch.cuda.empty_cache()

        q.put(("ok", (episodes, successes)))
    except Exception as e:
        import traceback
        q.put(("err", f"{e}\n{traceback.format_exc()}"))

def run_eval_once_in_subprocess(cfg: Dict[str, Any], physical_gpu_id: int, job: dict, timeout_s: int = 300) -> Tuple[int,int,Optional[str]]:
    import torch.multiprocessing as mp
    ctx = mp.get_context("spawn")
    q = ctx.Queue(1)
    p = ctx.Process(target=eval_once_child_entry, args=(q, cfg, physical_gpu_id, job), daemon=False)
    p.start()
    p.join(timeout=timeout_s)
    if p.is_alive():
        p.terminate(); p.join()
        return 0, 0, "timeout"
    if q.empty():
        return 0, 0, "crash"
    tag, payload = q.get()
    if tag == "ok":
        ep, suc = payload
        return ep, suc, None
    else:
        return 0, 0, str(payload)

def safe_eval_job(cfg: Dict[str, Any], physical_gpu_id: int, job: dict, timeout_s: int, retry: int):
    episodes = successes = 0
    err_last = None
    for t in range(retry + 1):
        episodes, successes, err = run_eval_once_in_subprocess(cfg, physical_gpu_id, job, timeout_s)
        if err is None:
            return episodes, successes, None
        err_last = err
        print(f"[Worker] eval crash/timeout ({err}), retry {t+1}/{retry}", flush=True)
    return episodes, successes, err_last

# -------------------- Main worker --------------------
@dataclass
class Args:
    gpu: int
    task_suite_name: str
    task_range: str
    task_to_huglinks_json_path: str
    output_dir: str
    prefer_prompt_key: str
    red_model: str
    qwen_mode: str
    examples_path: str
    num_instructions: int
    select_topk: int
    num_steps_wait: int
    num_trials_per_task: int
    failure_threshold: float
    seed: int
    eval_timeout_s: int
    eval_retry: int
    whole_acc_log_path: str
    is_save_video: bool
    save_folder: str

def build_argparser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--task-suite-name", default="libero_spatial",
                    choices=["libero_spatial","libero_object","libero_goal","libero_10"])
    ap.add_argument("--task-range", required=True, help="e.g. '0-3,5,7-9'")
    ap.add_argument("--task-to-huglinks-json-path",
                    default="/home/baoshuntong/code/saftyEmbodyAI/embodied-red-teaming/libero-init-frames/json_data_for_rl/vlm_initial_state_links_new.json")
    ap.add_argument("--output-dir", default=f"./output/{NOW}")
    ap.add_argument("--prefer-prompt-key", default="PRONOUN")
    ap.add_argument("--red-model", default="Qwen/Qwen3-VL-4B-Instruct")
    ap.add_argument("--qwen-mode", default=os.getenv("QWEN_MODE", "local"))
    ap.add_argument("--examples-path", default="")
    ap.add_argument("--num-instructions", type=int, default=5)
    ap.add_argument("--select-topk", type=int, default=3)
    ap.add_argument("--num-steps-wait", type=int, default=10)
    ap.add_argument("--num-trials-per-task", type=int, default=10)
    ap.add_argument("--failure-threshold", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--eval-timeout-s", type=int, default=300)
    ap.add_argument("--eval-retry", type=int, default=1)
    ap.add_argument("--whole-acc-log-path",
                    default="/home/baoshuntong/code/saftyEmbodyAI/embodied-red-teaming/logs/whole_acc.log")
    ap.add_argument("--is-save-video", action="store_true")
    ap.add_argument("--save-folder", default=f"output_videos/{NOW}/")
    return ap

def main():
    # Parse args
    ap = build_argparser()
    a = ap.parse_args()

    # Bind GPU + turn off FlashAttn BEFORE importing torch/transformers/openvla/etc.
    init_device_env(a.gpu)
    hard_disable_flash_attn_before_import()

    # Lightweight imports okay now
    import random
    import torch
    torch.cuda.set_device(0)
    random.seed(a.seed); torch.manual_seed(a.seed)

    # Heavy imports AFTER env set
    from libero.libero import benchmark
    from generate_intru_smolvlm import build_red_team_generator, CLIPEmbeddingModel

    # Determine OpenVLA checkpoint from suite
    ckpt = choose_openvla_ckpt(a.task_suite_name)

    # Build red-team generator
    clip_embedder = CLIPEmbeddingModel(device="cuda")
    red_team = build_red_team_generator(
        backend="qwenvl",
        embedding_model=clip_embedder,
        mode=a.qwen_mode,
        model=a.red_model,
        device="cuda",
    )

    # Load task links / suite
    with open(a.task_to_huglinks_json_path, "r") as f:
        task_to_links = json.load(f)
    task_suite_dict = task_to_links[a.task_suite_name]
    task_languages = list(task_suite_dict.keys())

    bench_dict = benchmark.get_benchmark_dict()
    suite = bench_dict[a.task_suite_name]()
    n_tasks = len(task_languages)

    sel_indices = parse_range(a.task_range)
    sel_indices = [i for i in sel_indices if 0 <= i < n_tasks]
    if not sel_indices:
        raise SystemExit(f"No valid task indices from '{a.task_range}'. Total tasks: {n_tasks}")

    os.makedirs(a.output_dir, exist_ok=True)
    os.makedirs(Path(a.whole_acc_log_path).parent, exist_ok=True)

    outfile = Path(a.output_dir) / f"redteaming_results.gpu{a.gpu}.tasks[{a.task_range}].json"
    results_all: List[Dict[str, Any]] = []

    suite_ep_sum = 0
    suite_succ_sum = 0

    cfg_eval = {
        "pretrained_checkpoint": ckpt,
        "num_steps_wait": a.num_steps_wait,
        "num_trials_per_task": a.num_trials_per_task,
    }

    print(f"[WORKER] GPU{a.gpu} | suite={a.task_suite_name} | tasks={sel_indices}", flush=True)

    for task_id in sel_indices:
        task_lang = task_languages[task_id]
        instruction = task_lang.replace("_", " ")
        links = task_suite_dict[task_lang]
        import random as _r
        image_url = _r.choice(links)

        print(f"\n[GEN] task {task_id}/{n_tasks-1}: {instruction}", flush=True)
        annotations, annotations_sim = red_team(
            instruction, image_url=image_url, prefer_prompt_key=a.prefer_prompt_key,
            num_instructions=a.num_instructions, select_topk=a.select_topk
        )

        jobs: List[dict] = []
        for ann, simv in zip(annotations, annotations_sim):
            ann_clean = re.sub(r'^\s*\d+\.\s*', '', ann).strip()
            jobs.append({
                "task_id": task_id,
                "task_language": task_lang,
                "annotation": ann_clean,
                "task_suite_name": a.task_suite_name,
                "num_trials_per_task": a.num_trials_per_task,
                "num_steps_wait": a.num_steps_wait,
                "annotation_sim": simv,
            })

        task_ep_sum = 0
        task_succ_sum = 0
        done, fail = [], []

        for job in jobs:
            ep, suc, err = safe_eval_job(
                cfg_eval, a.gpu, job,
                timeout_s=a.eval_timeout_s,
                retry=a.eval_retry
            )
            if err is None and ep > 0:
                sr = suc / ep
                (done if sr > a.failure_threshold else fail).append(job["annotation"])
                task_ep_sum += ep
                task_succ_sum += suc
            else:
                fail.append(job["annotation"])
                if err is not None:
                    print(f"[EVAL] task {task_id} rollout error: {err}", flush=True)

        suite_ep_sum += task_ep_sum
        suite_succ_sum += task_succ_sum
        task_sr = (task_succ_sum / task_ep_sum) if task_ep_sum else 0.0
        print(f"[GEN] task {task_id} success_rate={task_sr:.3f}", flush=True)

        results_all.append({
            "task": instruction,
            "image_url": image_url,
            "done_annotations": done,
            "fail_annotations": fail,
        })

        with open(outfile, "w") as fout:
            json.dump(results_all, fout, indent=4)
        gc.collect(); torch.cuda.empty_cache()

    all_sr = (suite_succ_sum / suite_ep_sum) if suite_ep_sum else 0.0
    print(f"\n[GEN] GPU{a.gpu} Overall success on {suite_ep_sum} episodes: {all_sr:.3f}", flush=True)

    # Append to whole_acc log as a small dict
    curr = {a.task_suite_name: {a.prefer_prompt_key: all_sr, "gpu": a.gpu, "range": a.task_range, "ts": NOW}}
    with open(a.whole_acc_log_path, "a") as fout:
        json.dump(curr, fout, indent=4)
    print(f"[GEN] Done. Results saved to {outfile}", flush=True)

if __name__ == "__main__":
    main()
