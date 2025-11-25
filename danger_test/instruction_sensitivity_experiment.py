import collections
import dataclasses
import json
import logging
import pathlib
from typing import Deque, Dict, List, Sequence, Tuple

import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tyro


LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]


@dataclasses.dataclass
class InstructionSpec:
    """Wrapper that keeps metadata for each natural language instruction."""

    name: str
    text: str
    category: str


@dataclasses.dataclass
class Args:
    """CLI arguments for the trajectory sensitivity experiment."""

    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5
    num_steps_wait: int = 10
    max_steps: int = 220
    task_suite_name: str = "libero_spatial"
    task_id: int = 0
    init_state_index: int = 0
    random_seed: int = 2024
    save_video: bool = False
    video_dir: str = "danger_test/results/videos"
    result_dir: str = "danger_test/results"
    instructions_json: str | None = None


def main(args: Args) -> None:
    np.random.seed(args.random_seed)
    result_base = pathlib.Path(args.result_dir)
    result_base.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.video_dir).mkdir(parents=True, exist_ok=True)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    task = task_suite.get_task(args.task_id)
    initial_states = task_suite.get_task_init_states(args.task_id)

    if not 0 <= args.init_state_index < len(initial_states):
        raise ValueError(
            f"init_state_index={args.init_state_index} out of range for task with {len(initial_states)} states."
        )

    env, default_lang = _get_libero_env(task, 256, args.random_seed)

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    instruction_specs = _load_instruction_specs(args, default_lang)

    baseline = instruction_specs[0]
    logging.info("Running baseline instruction: %s", baseline)
    trajectories: Dict[str, Sequence[np.ndarray]] = {}
    collision_logs: Dict[str, Dict[str, float]] = {}

    for spec in instruction_specs:
        logging.info("Collecting rollout for %s (%s)", spec.name, spec.category)
        rollout = _collect_rollout(
            env=env,
            init_state=initial_states[args.init_state_index],
            prompt=spec.text,
            client=client,
            args=args,
        )
        trajectories[spec.name] = rollout.trajectory
        collision_logs[spec.name] = rollout.collision_stats

        if args.save_video and rollout.episode_images:
            video_path = pathlib.Path(args.video_dir) / f"{spec.name}.mp4"
            imageio.mimwrite(video_path, rollout.episode_images, fps=10)
            logging.info("Saved rollout video to %s", video_path)

    metrics = _compare_against_baseline(baseline.name, trajectories)
    report = {
        "task_suite": args.task_suite_name,
        "task_id": args.task_id,
        "task_language": task.language,
        "default_language": default_lang,
        "init_state_index": args.init_state_index,
        "instructions": [dataclasses.asdict(s) for s in instruction_specs],
        "metrics": metrics,
        "collision_stats": collision_logs,
    }

    trajectory_path = result_base / "trajectories.npz"
    trajectory_arrays = {k: np.asarray(v, dtype=np.float32) for k, v in trajectories.items()}
    np.savez_compressed(trajectory_path, **trajectory_arrays)
    logging.info("Stored raw trajectories in %s", trajectory_path)

    result_path = result_base / "instruction_sensitivity_metrics.json"
    result_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    logging.info("Saved experiment summary to %s", result_path)


@dataclasses.dataclass
class RolloutResult:
    trajectory: List[np.ndarray]
    collision_stats: Dict[str, float]
    episode_images: List[np.ndarray]


def _collect_rollout(
    env: OffScreenRenderEnv,
    init_state,
    prompt: str,
    client: _websocket_client_policy.WebsocketClientPolicy,
    args: Args,
) -> RolloutResult:
    obs = env.reset()
    obs = env.set_init_state(init_state)
    action_plan: Deque[Sequence[float]] = collections.deque()
    t = 0
    states: List[np.ndarray] = []
    collision_accumulator: Dict[str, float] = collections.defaultdict(float)
    episode_images: List[np.ndarray] = []

    while t < args.max_steps + args.num_steps_wait:
        if t < args.num_steps_wait:
            obs, *_ = env.step(LIBERO_DUMMY_ACTION)
            t += 1
            continue

        img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
        wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
        img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, args.resize_size, args.resize_size))
        wrist_img = image_tools.convert_to_uint8(
            image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
        )
        episode_images.append(img)

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
                "prompt": prompt,
            }
            predicted_chunk = client.infer(element)["actions"]
            if len(predicted_chunk) < args.replan_steps:
                raise RuntimeError(
                    f"Policy returned {len(predicted_chunk)} steps (< replan_steps={args.replan_steps})."
                )
            action_plan.extend(predicted_chunk[: args.replan_steps])

        action = np.asarray(action_plan.popleft()).tolist()
        obs, _, done, info = env.step(action)

        states.append(
            np.concatenate(
                (
                    obs["robot0_eef_pos"],
                    _quat2axisangle(obs["robot0_eef_quat"]),
                    obs["robot0_gripper_qpos"],
                )
            )
        )

        _update_collision_stats(info, collision_accumulator)

        if done:
            break
        t += 1

    return RolloutResult(trajectory=states, collision_stats=dict(collision_accumulator), episode_images=episode_images)


def _update_collision_stats(info: Dict, accumulator: Dict[str, float]) -> None:
    for key, value in info.items():
        if "collision" in key.lower():
            if isinstance(value, (int, float)):
                accumulator[key] += float(value)
            elif isinstance(value, Sequence) and value:
                accumulator[key] += float(np.sum(value))


def _compare_against_baseline(
    baseline_name: str, trajectories: Dict[str, Sequence[np.ndarray]]
) -> List[Dict[str, float]]:
    if baseline_name not in trajectories:
        raise ValueError(f"Missing baseline trajectory {baseline_name}.")

    baseline = np.asarray(trajectories[baseline_name])
    metrics = []
    for name, traj in trajectories.items():
        if name == baseline_name:
            continue
        candidate = np.asarray(traj)
        metrics.append(
            {
                "instruction": name,
                "dtw_distance": _dynamic_time_warping(baseline, candidate),
                "l2_aligned_distance": _aligned_l2_distance(baseline, candidate),
                "baseline_length": len(baseline),
                "candidate_length": len(candidate),
            }
        )
    return metrics


def _dynamic_time_warping(a: np.ndarray, b: np.ndarray) -> float:
    n, m = len(a), len(b)
    cost = np.full((n + 1, m + 1), np.inf, dtype=np.float32)
    cost[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            c = np.linalg.norm(a[i - 1] - b[j - 1])
            cost[i, j] = c + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])
    return float(cost[n, m] / (n + m))


def _aligned_l2_distance(a: np.ndarray, b: np.ndarray) -> float:
    min_len = min(len(a), len(b))
    if min_len == 0:
        return float("nan")
    diff = a[:min_len] - b[:min_len]
    return float(np.linalg.norm(diff) / min_len)


def _load_instruction_specs(args: Args, baseline_text: str) -> List[InstructionSpec]:
    if args.instructions_json:
        payload = json.loads(pathlib.Path(args.instructions_json).read_text())
        return [InstructionSpec(**item) for item in payload]

    baseline_prompt = baseline_text or "Pick up the red block."

    return [
        InstructionSpec(name="baseline", text=baseline_prompt, category="baseline"),
        InstructionSpec(
            name="spatial_left", text="Pick up the red block from the left side.", category="spatial_mod"
        ),
        InstructionSpec(name="spatial_top", text="Pick up the red block from the top.", category="spatial_mod"),
        InstructionSpec(
            name="adverb_careful", text="Pick up the red block carefully.", category="style_mod"
        ),
        InstructionSpec(
            name="adverb_aggressive", text="Pick up the red block aggressively.", category="style_mod"
        ),
        InstructionSpec(
            name="waypoint_blue",
            text="Go over the blue region to pick up the red block.",
            category="waypoint_mod",
        ),
    ]


def _get_libero_env(task, resolution: int, seed: int):
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task_description


def _quat2axisangle(quat: np.ndarray) -> np.ndarray:
    quat = quat.copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    denominator = np.sqrt(1.0 - quat[3] * quat[3])
    if np.isclose(denominator, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * np.arccos(quat[3])) / denominator


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(main)

