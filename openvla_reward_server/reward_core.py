from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

# Use the OpenVLA-specific evaluation loop
from .env_pertask import Args, eval_one_task
from pi0_reward_server.utils import _extract_text_from_vllm

current_time = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())

# Default config tuned for the OpenVLA-OFT policy server.
DEFAULT_LIBERO_CFG: Dict[str, Any] = {
    # Model server parameters
    "host": "0.0.0.0",
    "port": 23451,  # default for openvla-oft policy server
    "resize_size": 224,
    "replan_steps": 8,  # NUM_ACTIONS_CHUNK for LIBERO in openvla-oft
    # LIBERO environment-specific parameters
    "task_suite_name": "libero_spatial",
    "task_id": 0,
    "num_steps_wait": 10,
    "num_trials_per_task": 5,
    "init_state_id": 0,
    # instruction will be overwritten by candidate text if provided
    "instruction": (
        "pick up the black bowl between the plate and the ramekin and place it on the plate"
    ),
    # Utils
    "video_out_path": f"/root/autodl-tmp/output_big_data/libero/{current_time}/videos",
    "save_videos": False,
    "seed": 7,
}


def _merge_cfg(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Merge caller kwargs with defaults; tolerate partial libero_cfg overrides."""
    merged_cfg = dict(DEFAULT_LIBERO_CFG)
    merged_cfg.update(kwargs.get("libero_cfg", {}) or {})

    # Allow top-level overrides for common knobs.
    if "num_trials_per_task" in kwargs:
        merged_cfg["num_trials_per_task"] = int(kwargs["num_trials_per_task"])
    if "num_steps_wait" in kwargs:
        merged_cfg["num_steps_wait"] = int(kwargs["num_steps_wait"])
    if "replan_steps" in kwargs:
        merged_cfg["replan_steps"] = int(kwargs["replan_steps"])
    if "port" in kwargs:
        merged_cfg["port"] = int(kwargs["port"])
    if "host" in kwargs:
        merged_cfg["host"] = str(kwargs["host"])
    return merged_cfg


def compute_score(
    responses: List[Any],
    metas: Optional[List[Dict[str, Any]]] = None,
    **kwargs,
) -> List[float]:
    """
    Evaluate model responses on LIBERO tasks using the OpenVLA-OFT policy server.

    Args:
        responses: list of vLLM-like response objects.
        metas: optional metadata per sample.
        kwargs: reward_function_kwargs with optional libero_cfg overrides.
    """
    merged_cfg = _merge_cfg(kwargs)

    metas = metas or [{}] * len(responses)
    if len(metas) < len(responses):
        metas = list(metas) + [{} for _ in range(len(responses) - len(metas))]

    success_list: List[float] = []
    total_samples = len(responses)
    logging.info(f"[OpenVLA-OFT] Processing {total_samples} samples...")

    for idx, (r, m) in enumerate(zip(responses, metas), 1):
        logging.info(f"[OpenVLA-OFT] [{idx}/{total_samples}] start")
        m = m or {}

        suite = (
            m.get("suite")
            or m.get("task_suite_name")
            or m.get("suite_name")
            or merged_cfg["task_suite_name"]
        )
        task_id = int(m.get("task_id", merged_cfg["task_id"]))
        seed = int(m.get("seed", merged_cfg["seed"]))
        init_state_id = int(m.get("init_state_id", merged_cfg["init_state_id"]))

        candidate = _extract_text_from_vllm(r)
        i0 = str(m.get("original_instruction", "")) if m.get("original_instruction") is not None else ""
        instruction = (candidate or i0 or merged_cfg.get("instruction", "")).strip()

        args = Args(
            host=merged_cfg["host"],
            port=int(merged_cfg["port"]),
            resize_size=int(merged_cfg["resize_size"]),
            replan_steps=int(merged_cfg["replan_steps"]),
            task_suite_name=str(suite),
            task_id=int(task_id),
            init_state_id=int(init_state_id),
            num_steps_wait=int(merged_cfg["num_steps_wait"]),
            num_trials_per_task=int(merged_cfg["num_trials_per_task"]),
            instruction=instruction,
            video_out_path=str(merged_cfg["video_out_path"]),
            save_videos=bool(merged_cfg.get("save_videos", False)),
            seed=int(seed),
        )

        try:
            sr = float(eval_one_task(args))
            logging.info(f"[OpenVLA-OFT] [{idx}/{total_samples}] success_rate={sr:.2%}")
        except Exception as e:  # noqa: BLE001
            logging.error(
                "[OpenVLA-OFT] eval_one_task failed for suite=%s task_id=%s: %s",
                suite,
                task_id,
                e,
            )
            sr = 0.0
            logging.info(f"[OpenVLA-OFT] [{idx}/{total_samples}] failed, using 0.0")

        success_list.append(sr)

    return success_list


