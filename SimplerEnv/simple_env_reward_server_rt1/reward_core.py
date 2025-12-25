from __future__ import annotations
import time
from typing import Any, Dict, List, Optional
from pi0_reward_server.utils import _extract_text_from_vllm
current_time = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())

DEFAULT_SIMPLE_ENV_CFG: Dict[str, Any] = {
    "host": "0.0.0.0",
    "port": 8002,
    "resize_size": 256,
    "replan_steps": 1,
    "robot": "google_robot",
    "env_name": "GoogleRobotOpenDrawer-v0",
    "scene_name": "bridge_table_1_v1",
    "num_steps_wait": 10,
    "num_trials_per_task": 1,
    "init_state_id": 0,
    "instruction": "open the drawer",
    "rgb_overlay_path": None,
    "control_freq": 5,
    "sim_freq": 500,
    "max_episode_steps": 80,
    "additional_env_build_kwargs": {},
    "obs_camera_name": None,
    "action_scale": 1.0,
    "policy_setup": "google_robot",
    "saved_model_path": "rt_1_x_tf_trained_for_002272480_step",
    "video_out_path": f"/root/autodl-tmp/output_big_data/simple_env/{current_time}/videos",
    "save_videos": False,
    "seed": 7,
}


def compute_score(responses: List[Any], metas: Optional[List[Dict[str, Any]]] = None, **kwargs) -> List[float]:
    from .env_pertask import Args, eval_one_task
    merged_cfg = dict(DEFAULT_SIMPLE_ENV_CFG)
    merged_cfg.update(kwargs.get("simple_env_cfg", {}) or {})
    if "num_trials_per_task" in kwargs:
        merged_cfg["num_trials_per_task"] = int(kwargs["num_trials_per_task"])
    if "num_steps_wait" in kwargs:
        merged_cfg["num_steps_wait"] = int(kwargs["num_steps_wait"])
    metas = metas or [{}] * len(responses)
    if len(metas) < len(responses):
        metas = list(metas) + [{} for _ in range(len(responses) - len(metas))]
    success_list: List[float] = []
    for r, m in zip(responses, metas):
        candidate = _extract_text_from_vllm(r)
        i0 = str(m.get("original_instruction", "")) if m.get("original_instruction") is not None else ""
        instruction = (candidate or i0 or merged_cfg.get("instruction", "")).strip()
        args = Args(
            host=merged_cfg["host"],
            port=int(merged_cfg["port"]),
            resize_size=int(merged_cfg["resize_size"]),
            replan_steps=int(merged_cfg["replan_steps"]),
            robot=str(m.get("robot", merged_cfg["robot"])),
            env_name=str(m.get("env_name", merged_cfg["env_name"])),
            scene_name=str(m.get("scene_name", merged_cfg["scene_name"])),
            num_steps_wait=int(merged_cfg["num_steps_wait"]),
            num_trials_per_task=int(merged_cfg["num_trials_per_task"]),
            instruction=instruction,
            init_state_id=int(m.get("init_state_id", merged_cfg["init_state_id"])),
            rgb_overlay_path=m.get("rgb_overlay_path", merged_cfg["rgb_overlay_path"]),
            control_freq=int(merged_cfg["control_freq"]),
            sim_freq=int(merged_cfg["sim_freq"]),
            max_episode_steps=int(merged_cfg["max_episode_steps"]),
            additional_env_build_kwargs=merged_cfg.get("additional_env_build_kwargs") or {},
            obs_camera_name=merged_cfg.get("obs_camera_name"),
            action_scale=float(merged_cfg["action_scale"]),
            policy_setup=str(merged_cfg["policy_setup"]),
            saved_model_path=str(merged_cfg["saved_model_path"]),
            seed=int(m.get("seed", merged_cfg["seed"])),
        )
        try:
            sr = float(eval_one_task(args))
        except Exception:
            sr = 0.0
        success_list.append(sr)
    return success_list
