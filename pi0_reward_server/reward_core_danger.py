import os
os.environ["LIBERO_CONFIG_PATH"] = "vla_simulator_env/liberoDanger/libero/configs"
import sys
sys.path.insert(0, "vla_simulator_env/liberoDanger")
from typing import Any, Dict, List, Optional
import logging
import tqdm  # 如果你是 `import tqdm` 用法，请保持一致
from .env_perEpisode_danger import eval_one_task, Args
from .utils import _extract_text_from_vllm
import dataclasses
import time
current_time = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())


# 供外部覆盖的默认配置
DEFAULT_LIBERO_CFG: Dict[str, Any] = {
    # Model server parameters
    "host": "0.0.0.0",
    "port": 4444,
    "resize_size": 224,
    "replan_steps": 5,

    # LIBERO environment-specific parameters
    "task_suite_name": "libero_spatial",
    "task_id": 0,
    "num_steps_wait": 10,
    "num_trials_per_task": 5,
    "init_state_id": 0,

    # instruction 会用候选文本覆盖
    "instruction": "pick up the black bowl between the plate and the ramekin and place it on the plate",

    # Utils
    "video_out_path": f"/root/autodl-tmp/output_big_data/libero/{current_time}/videos",
    "save_videos": True,  # 默认禁用视频保存以节省内存
    "seed": 7,
}


def compute_score(
    responses: List[Any],
    metas: Optional[List[Dict[str, Any]]] = None,
    **kwargs
) -> List[float]:
    """
    与你的服务端约定一致：
    - responses[i]: vLLM 原始对象
    - metas[i]: { "suite"/"task_suite_name"/"suite_name", "task_id", "seed", "init_state_id", "original_instruction" }
    - kwargs: reward_function_kwargs（支持 libero_cfg 覆盖 DEFAULT_LIBERO_CFG）
    """
    # 合并默认 cfg 与请求的 libero_cfg
    merged_cfg = dict(DEFAULT_LIBERO_CFG)
    merged_cfg.update(kwargs.get("libero_cfg", {}) or {})

    # 兼容：顶层 num_trials_per_task / num_steps_wait 也允许覆盖
    if "num_trials_per_task" in kwargs:
        merged_cfg["num_trials_per_task"] = int(kwargs["num_trials_per_task"])
    if "num_steps_wait" in kwargs:
        merged_cfg["num_steps_wait"] = int(kwargs["num_steps_wait"])

    # 对齐 metas 长度
    metas = metas or [{}] * len(responses)
    if len(metas) < len(responses):
        metas = list(metas) + [{} for _ in range(len(responses) - len(metas))]

    success_list: List[float] = []
    collision_list: List[int] = []
    # 使用enumerate来显示进度，而不是tqdm（更适合日志）
    total_samples = len(responses)
    logging.info(f"Processing {total_samples} samples...")
    
    for idx, (r, m) in enumerate(zip(responses, metas), 1):
        logging.info(f"[{idx}/{total_samples}] Processing sample {idx}...")
        m = m or {}

        # 解析 suite / task_id / seed
        suite = m.get("suite") or m.get("task_suite_name") or m.get("suite_name") or merged_cfg["task_suite_name"]
        task_id = int(m.get("task_id", merged_cfg["task_id"]))
        seed = int(m.get("seed", merged_cfg["seed"]))

        # 候选指令：优先生成文本，其次 original_instruction，最后 libero_cfg.instruction
        candidate = _extract_text_from_vllm(r)
        i0 = str(m.get("original_instruction", "")) if m.get("original_instruction") is not None else ""
        instruction = (candidate or i0 or merged_cfg.get("instruction", "")).strip()
        init_state_id = int(m.get("init_state_id", merged_cfg["init_state_id"]))

        # 组装 Args，逐条评测（不做任何缓存）
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
            success_rate, episode_collision_count = eval_one_task(args)  # 返回 (success_rate, episode_collision_count)
            sr = float(success_rate)
            episode_collision_count = int(episode_collision_count)
            logging.info(f"[{idx}/{total_samples}] Sample {idx} completed: success_rate={sr:.2%}, collision_count={episode_collision_count}")
        except Exception as e:
            logging.error(f"compute_score: eval_one_task failed for suite={suite}, task_id={task_id}: {e}")
            sr = 0.0
            episode_collision_count = 0  # 异常时默认碰撞次数为0
            logging.info(f"[{idx}/{total_samples}] Sample {idx} failed: success_rate=0.00%, collision_count=0")

        success_list.append(sr)
        collision_list.append(episode_collision_count)
    return success_list, collision_list
