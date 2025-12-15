"""
轨迹分析脚本
计算轨迹的平均路径和标准差
"""
import json
import logging
import pathlib
from typing import Tuple

import numpy as np
import tyro
from dataclasses import dataclass


@dataclass
class Args:
    """轨迹分析参数"""
    input_file: str  # 轨迹数据文件路径
    output_dir: str = "trajectory_profiling/results"
    approach_window: int = 10  # 抓取前N帧作为approach阶段


def load_trajectories(input_file: str) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    加载轨迹数据
    
    Returns:
        trajectories: (n_episodes, max_len, 3) 轨迹数组
        lengths: (n_episodes,) 每条轨迹的实际长度
        metadata: 元数据字典
    """
    data = np.load(input_file, allow_pickle=True)
    trajectories = data["trajectories"]
    lengths = data["lengths"]
    
    metadata = {
        "task_description": str(data.get("task_description", "")),
        "task_suite": str(data.get("task_suite", "")),
        "task_id": int(data.get("task_id", 0)),
        "num_successful": int(data.get("num_successful", len(trajectories))),
    }
    
    return trajectories, lengths, metadata


def compute_statistics(trajectories: np.ndarray, lengths: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算轨迹统计量
    
    Args:
        trajectories: (n_episodes, max_len, 3) 轨迹数组
        lengths: (n_episodes,) 每条轨迹的实际长度
    
    Returns:
        mean_path: (max_len, 3) 平均路径
        std_path: (max_len, 3) 标准差路径
    """
    n_episodes, max_len, _ = trajectories.shape
    
    mean_path = np.zeros((max_len, 3))
    std_path = np.zeros((max_len, 3))
    
    for t in range(max_len):
        # 只考虑在该时间步有效的轨迹
        valid_mask = lengths > t
        if valid_mask.sum() == 0:
            continue
        
        valid_positions = trajectories[valid_mask, t, :]  # (n_valid, 3)
        
        mean_path[t] = valid_positions.mean(axis=0)
        std_path[t] = valid_positions.std(axis=0)
    
    return mean_path, std_path


def find_approach_phase(mean_path: np.ndarray, lengths: np.ndarray, approach_window: int) -> int:
    """
    找到approach阶段的关键时刻
    通常是在轨迹中段，抓取动作发生前
    
    Args:
        mean_path: 平均路径
        lengths: 轨迹长度
        approach_window: approach阶段的窗口大小
    
    Returns:
        approach_t: approach阶段开始的时间步
    """
    # 使用轨迹长度的中位数来确定关键时刻
    median_length = int(np.median(lengths))
    
    # approach阶段通常在轨迹的30%-50%位置
    approach_t = int(median_length * 0.4)
    
    # 确保有足够的窗口
    if approach_t + approach_window > median_length:
        approach_t = median_length - approach_window
    
    if approach_t < 0:
        approach_t = 0
    
    return approach_t


def analyze_trajectories(args: Args) -> None:
    """分析轨迹数据"""
    logging.info(f"加载轨迹数据: {args.input_file}")
    
    trajectories, lengths, metadata = load_trajectories(args.input_file)
    
    logging.info(f"轨迹数量: {len(trajectories)}")
    logging.info(f"轨迹形状: {trajectories.shape}")
    logging.info(f"平均长度: {lengths.mean():.1f} ± {lengths.std():.1f}")
    
    # 计算统计量
    logging.info("计算平均路径和标准差...")
    mean_path, std_path = compute_statistics(trajectories, lengths)
    
    # 找到approach阶段
    approach_t = find_approach_phase(mean_path, lengths, args.approach_window)
    logging.info(f"Approach阶段时间步: {approach_t}")
    
    # 计算approach阶段的平均位置和标准差
    approach_mean = mean_path[approach_t:approach_t + args.approach_window].mean(axis=0)
    approach_std = std_path[approach_t:approach_t + args.approach_window].mean(axis=0)
    
    logging.info(f"Approach阶段平均位置: {approach_mean}")
    logging.info(f"Approach阶段平均标准差: {approach_std}")
    
    # 保存结果
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    task_suite = metadata["task_suite"]
    task_id = metadata["task_id"]
    
    # 保存统计结果
    output_file = output_dir / f"{task_suite}_task{task_id}_statistics.npz"
    np.savez(
        output_file,
        mean_path=mean_path,
        std_path=std_path,
        approach_t=approach_t,
        approach_mean=approach_mean,
        approach_std=approach_std,
        trajectory_lengths=lengths,
        **metadata,
    )
    
    logging.info(f"统计结果已保存到 {output_file}")
    
    # 保存JSON格式的摘要
    json_file = output_dir / f"{task_suite}_task{task_id}_statistics.json"
    json_data = {
        **metadata,
        "approach_t": int(approach_t),
        "approach_window": args.approach_window,
        "approach_mean": approach_mean.tolist(),
        "approach_std": approach_std.tolist(),
        "mean_trajectory_length": float(lengths.mean()),
        "std_trajectory_length": float(lengths.std()),
    }
    
    with open(json_file, "w") as f:
        json.dump(json_data, f, indent=2)
    
    logging.info(f"摘要已保存到 {json_file}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(analyze_trajectories)

