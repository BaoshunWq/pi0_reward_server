"""
轨迹收集脚本
收集VLA模型在正常指令下执行任务的成功轨迹的末端执行器坐标
"""
import os
os.environ["LIBERO_CONFIG_PATH"] = "/root/autodl-tmp/code/attackVLA/pi0_newlibero_reward_server/openpi/third_party/libero/libero/configs"

import collections
import dataclasses
import json
import logging
import math
import pathlib
from typing import List, Dict, Tuple

import numpy as np
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm
import tyro

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256


@dataclasses.dataclass
class Args:
    """轨迹收集参数"""
    host: str = "0.0.0.0"
    port: int = 4444
    resize_size: int = 224
    replan_steps: int = 5
    
    task_suite_name: str = "libero_goal"
    num_steps_wait: int = 10
    num_episodes: int = 50  # 收集的episode数量（只保留成功的）
    min_successful_episodes: int = 20  # 最少需要的成功episode数量
    
    seed: int = 7
    output_dir: str = "trajectory_profiling/data"
    task_id: int = 0  # 要收集的任务ID


def _quat2axisangle(quat):
    """四元数转轴角"""
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def _get_libero_env(task, resolution, seed):
    """初始化LIBERO环境"""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task_description


def collect_trajectory(
    env: OffScreenRenderEnv,
    initial_state,
    task_description: str,
    client: _websocket_client_policy.WebsocketClientPolicy,
    args: Args,
    max_steps: int,
) -> Tuple[List[np.ndarray], bool]:
    """
    收集单个episode的轨迹
    
    Returns:
        trajectory: 末端执行器坐标列表 [(x, y, z), ...]
        success: 是否成功
    """
    env.reset()
    obs = env.set_init_state(initial_state)
    action_plan = collections.deque()
    
    trajectory = []
    t = 0
    
    # 等待物体稳定
    for _ in range(args.num_steps_wait):
        obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)
        t += 1
    
    while t < max_steps + args.num_steps_wait:
        try:
            # 获取观察
            img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
            wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
            img = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
            )
            wrist_img = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
            )
            
            if not action_plan:
                # 获取新动作
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
                    "prompt": str(task_description),
                }
                
                action_chunk = client.infer(element)["actions"]
                assert len(action_chunk) >= args.replan_steps
                action_plan.extend(action_chunk[: args.replan_steps])
            
            action = action_plan.popleft()
            
            # 执行动作并记录末端执行器位置
            obs, reward, done, info = env.step(action.tolist())
            
            # 记录末端执行器坐标 (x, y, z)
            eef_pos = obs["robot0_eef_pos"].copy()
            trajectory.append(eef_pos)
            
            if done:
                return trajectory, True
            
            t += 1
            
        except Exception as e:
            logging.error(f"Episode failed at step {t}: {e}")
            return trajectory, False
    
    return trajectory, False


def collect_trajectories(args: Args) -> None:
    """收集轨迹数据"""
    np.random.seed(args.seed)
    
    # 初始化任务套件
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    
    if args.task_id >= task_suite.n_tasks:
        raise ValueError(f"Task ID {args.task_id} >= {task_suite.n_tasks}")
    
    # 获取任务
    task = task_suite.get_task(args.task_id)
    initial_states = task_suite.get_task_init_states(args.task_id)
    
    # 初始化环境
    env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)
    
    # 确定最大步数
    if args.task_suite_name == "libero_spatial":
        max_steps = 220
    elif args.task_suite_name == "libero_object":
        max_steps = 280
    elif args.task_suite_name == "libero_goal":
        max_steps = 300
    elif args.task_suite_name == "libero_10":
        max_steps = 520
    elif args.task_suite_name == "libero_90":
        max_steps = 400
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")
    
    # 初始化策略客户端
    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    
    # 收集轨迹
    all_trajectories = []
    successful_count = 0
    episode_idx = 0
    
    logging.info(f"开始收集任务 {args.task_id} 的轨迹: {task_description}")
    logging.info(f"目标: 收集至少 {args.min_successful_episodes} 条成功轨迹")
    
    pbar = tqdm.tqdm(total=args.num_episodes, desc="收集轨迹")
    
    while successful_count < args.min_successful_episodes and episode_idx < args.num_episodes:
        initial_state = initial_states[episode_idx % len(initial_states)]
        
        trajectory, success = collect_trajectory(
            env, initial_state, task_description, client, args, max_steps
        )
        
        if success:
            all_trajectories.append(trajectory)
            successful_count += 1
            pbar.set_postfix({"成功": successful_count, "总数": episode_idx + 1})
        
        episode_idx += 1
        pbar.update(1)
    
    pbar.close()
    
    if successful_count < args.min_successful_episodes:
        logging.warning(
            f"只收集到 {successful_count} 条成功轨迹，少于要求的 {args.min_successful_episodes} 条"
        )
    
    # 保存轨迹数据
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / f"{args.task_suite_name}_task{args.task_id}_trajectories.npz"
    
    # 转换为numpy数组（填充到相同长度）
    max_len = max(len(traj) for traj in all_trajectories)
    padded_trajectories = []
    for traj in all_trajectories:
        padded = np.zeros((max_len, 3))
        padded[:len(traj)] = np.array(traj)
        padded_trajectories.append(padded)
    
    trajectories_array = np.array(padded_trajectories)  # Shape: (n_episodes, max_len, 3)
    trajectory_lengths = np.array([len(traj) for traj in all_trajectories])
    
    np.savez(
        output_file,
        trajectories=trajectories_array,
        lengths=trajectory_lengths,
        task_description=task_description,
        task_suite=args.task_suite_name,
        task_id=args.task_id,
        num_successful=successful_count,
    )
    
    logging.info(f"已保存 {successful_count} 条成功轨迹到 {output_file}")
    logging.info(f"轨迹形状: {trajectories_array.shape}")
    logging.info(f"平均轨迹长度: {trajectory_lengths.mean():.1f} ± {trajectory_lengths.std():.1f}")
    
    # 同时保存为JSON格式（便于查看）
    json_file = output_dir / f"{args.task_suite_name}_task{args.task_id}_trajectories.json"
    json_data = {
        "task_suite": args.task_suite_name,
        "task_id": args.task_id,
        "task_description": task_description,
        "num_successful_episodes": successful_count,
        "trajectory_lengths": trajectory_lengths.tolist(),
        "mean_length": float(trajectory_lengths.mean()),
        "std_length": float(trajectory_lengths.std()),
    }
    
    with open(json_file, "w") as f:
        json.dump(json_data, f, indent=2)
    
    logging.info(f"元数据已保存到 {json_file}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(collect_trajectories)

