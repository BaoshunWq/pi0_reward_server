from typing import Any, Dict, List, Optional
import logging
import tqdm  # 如果你是 `import tqdm` 用法，请保持一致
from env_pertask import eval_one_task
import dataclasses


@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = "libero_spatial"  # libero_spatial, libero_object, libero_goal, libero_10, libero_90
    task_id: int = 0                          # 要评测的任务 ID（基于 task suite 的索引）
    num_steps_wait: int = 10                  # 开场等待步数，等物体稳定
    num_trials_per_task: int = 10             # 同一个任务重复多少次 rollout
    instruction: str = "pick up the black bowl between the plate and the ramekin and place it on the plate"                     # 外部指令（为空则使用任务自带 language）

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "data/libero/videos"  # Path to save videos
    seed: int = 7                                # Random Seed (for reproducibility)


# 供外部覆盖的默认配置
DEFAULT_LIBERO_CFG: Dict[str, Any] = {
    # Model server parameters
    "host": "0.0.0.0",
    "port": 8000,
    "resize_size": 224,
    "replan_steps": 5,

    # LIBERO environment-specific parameters
    "task_suite_name": "libero_spatial",
    "task_id": 0,
    "num_steps_wait": 10,
    "num_trials_per_task": 5,

    # instruction 会用候选文本覆盖
    "instruction": "pick up the black bowl between the plate and the ramekin and place it on the plate",

    # Utils
    "video_out_path": "data/libero/videos",
    "seed": 7,
}

def _extract_text_from_vllm(r: Any) -> str:
    """尽量鲁棒的候选文本抽取。"""
    if r is None:
        return ""
    if isinstance(r, str):
        return r
    if isinstance(r, list):
        return _extract_text_from_vllm(r[0]) if r else ""
    if isinstance(r, dict):
        for k in ("text", "generated_text", "output_text", "response", "content"):
            v = r.get(k)
            if isinstance(v, str):
                return v
        if isinstance(r.get("outputs"), list) and r["outputs"]:
            o0 = r["outputs"][0]
            if isinstance(o0, dict) and isinstance(o0.get("text"), str):
                return o0["text"]
            # openai/chat 风格
        if isinstance(r.get("choices"), list) and r["choices"]:
            ch0 = r["choices"][0]
            if isinstance(ch0, dict):
                if isinstance(ch0.get("text"), str):
                    return ch0["text"]
                msg = ch0.get("message")
                if isinstance(msg, dict) and isinstance(msg.get("content"), str):
                    return msg["content"]
    try:
        return str(r)
    except Exception:
        return ""

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

    for r, m in tqdm.tqdm(zip(responses, metas), total=len(responses)):
        m = m or {}

        # 解析 suite / task_id / seed
        suite = m.get("suite") or m.get("task_suite_name") or m.get("suite_name") or merged_cfg["task_suite_name"]
        task_id = int(m.get("task_id", merged_cfg["task_id"]))
        seed = int(m.get("seed", merged_cfg["seed"]))

        # 候选指令：优先生成文本，其次 original_instruction，最后 libero_cfg.instruction
        candidate = _extract_text_from_vllm(r)
        i0 = str(m.get("original_instruction", "")) if m.get("original_instruction") is not None else ""
        instruction = (candidate or i0 or merged_cfg.get("instruction", "")).strip()

        # 组装 Args，逐条评测（不做任何缓存）
        args = Args(
            host=merged_cfg["host"],
            port=int(merged_cfg["port"]),
            resize_size=int(merged_cfg["resize_size"]),
            replan_steps=int(merged_cfg["replan_steps"]),
            task_suite_name=str(suite),
            task_id=int(task_id),
            num_steps_wait=int(merged_cfg["num_steps_wait"]),
            num_trials_per_task=int(merged_cfg["num_trials_per_task"]),
            instruction=instruction,
            video_out_path=str(merged_cfg["video_out_path"]),
            seed=int(seed),
        )

        try:
            sr = float(eval_one_task(args))  # ← 直接用返回的成功率
        except Exception as e:
            logging.error(f"compute_score: eval_one_task failed for suite={suite}, task_id={task_id}: {e}")
            sr = 0.0

        success_list.append(sr)

    return success_list
