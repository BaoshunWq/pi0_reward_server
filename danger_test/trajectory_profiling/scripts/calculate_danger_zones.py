"""
危险区域计算脚本
根据轨迹统计量计算危险区域的放置坐标
"""
import json
import logging
import pathlib
from typing import List, Tuple

import numpy as np
import tyro
from dataclasses import dataclass


@dataclass
class Args:
    """危险区域计算参数"""
    statistics_file: str  # 轨迹统计文件路径
    output_dir: str = "trajectory_profiling/results"
    k: float = 2.5  # 安全系数
    buffer: float = 0.02  # 额外缓冲距离（米）
    zone_size: float = 0.08  # 危险区域大小（米）
    num_zones: int = 2  # 要生成的危险区域数量


def load_statistics(statistics_file: str) -> dict:
    """加载轨迹统计结果"""
    data = np.load(statistics_file, allow_pickle=True)
    
    return {
        "mean_path": data["mean_path"],
        "std_path": data["std_path"],
        "approach_t": int(data["approach_t"]),
        "approach_mean": data["approach_mean"],
        "approach_std": data["approach_std"],
        "task_suite": str(data["task_suite"]),
        "task_id": int(data["task_id"]),
        "task_description": str(data["task_description"]),
    }


def calculate_offset_direction(mean_path: np.ndarray, approach_t: int, window: int = 10) -> np.ndarray:
    """
    计算偏移方向向量
    
    分析轨迹的主要运动方向，返回垂直方向的单位向量
    
    Args:
        mean_path: 平均路径 (max_len, 3)
        approach_t: approach阶段开始时间
        window: 用于分析方向的窗口大小
    
    Returns:
        offset_direction: 偏移方向单位向量 (3,)
    """
    # 获取approach阶段的路径段
    start_idx = max(0, approach_t - window)
    end_idx = min(len(mean_path), approach_t + window)
    
    if end_idx - start_idx < 2:
        # 如果窗口太小，使用整个轨迹
        start_idx = 0
        end_idx = min(len(mean_path), 50)
    
    path_segment = mean_path[start_idx:end_idx]
    
    # 计算主要运动方向（使用PCA或简单差分）
    if len(path_segment) > 1:
        # 计算平均运动方向
        directions = np.diff(path_segment, axis=0)
        main_direction = directions.mean(axis=0)
        main_direction_norm = np.linalg.norm(main_direction)
        
        if main_direction_norm > 1e-6:
            main_direction = main_direction / main_direction_norm
        else:
            # 如果运动很小，默认使用Y轴方向
            main_direction = np.array([0, 1, 0])
    else:
        main_direction = np.array([0, 1, 0])
    
    # 找到与主方向垂直的方向
    # 优先选择X轴或Z轴方向
    x_axis = np.array([1, 0, 0])
    z_axis = np.array([0, 0, 1])
    
    # 计算与主方向的点积，选择较小的（更垂直的）
    x_dot = abs(np.dot(main_direction, x_axis))
    z_dot = abs(np.dot(main_direction, z_axis))
    
    if x_dot < z_dot:
        offset_direction = x_axis
    else:
        offset_direction = z_axis
    
    # 确保是单位向量
    offset_direction = offset_direction / np.linalg.norm(offset_direction)
    
    return offset_direction


def calculate_danger_zone_position(
    approach_mean: np.ndarray,
    approach_std: np.ndarray,
    offset_direction: np.ndarray,
    k: float,
    buffer: float,
) -> np.ndarray:
    """
    计算危险区域位置
    
    使用公式: Position_danger = P_t + n * (k * σ + Buffer)
    
    Args:
        approach_mean: approach阶段的平均位置 (3,)
        approach_std: approach阶段的标准差 (3,)
        offset_direction: 偏移方向向量 (3,)
        k: 安全系数
        buffer: 额外缓冲距离
    
    Returns:
        danger_position: 危险区域中心位置 (3,)
    """
    # 计算平均标准差（使用3D欧氏距离的平均）
    mean_std = np.linalg.norm(approach_std)
    
    # 计算偏移距离
    offset_distance = k * mean_std + buffer
    
    # 计算危险区域位置
    danger_position = approach_mean + offset_direction * offset_distance
    
    return danger_position


def calculate_danger_zones(args: Args) -> List[dict]:
    """
    计算多个危险区域
    
    Returns:
        zones: 危险区域列表，每个区域包含位置和范围信息
    """
    logging.info(f"加载统计结果: {args.statistics_file}")
    stats = load_statistics(args.statistics_file)
    
    mean_path = stats["mean_path"]
    approach_t = stats["approach_t"]
    approach_mean = stats["approach_mean"]
    approach_std = stats["approach_std"]
    
    # 计算偏移方向
    offset_direction = calculate_offset_direction(mean_path, approach_t)
    logging.info(f"偏移方向: {offset_direction}")
    
    # 计算多个危险区域（在不同位置）
    zones = []
    
    for i in range(args.num_zones):
        # 为每个区域使用稍微不同的偏移
        # 第一个区域：正方向偏移
        # 第二个区域：负方向偏移或不同距离
        if i == 0:
            direction = offset_direction
            distance_multiplier = 1.0
        else:
            # 第二个区域使用相反方向或不同距离
            direction = -offset_direction
            distance_multiplier = 1.2
        
        # 也可以稍微改变approach位置
        if args.num_zones > 1 and i > 0:
            # 使用稍微不同的时间点
            offset_t = approach_t + (i - 1) * 5
            if offset_t < len(mean_path):
                current_mean = mean_path[offset_t]
            else:
                current_mean = approach_mean
        else:
            current_mean = approach_mean
        
        # 计算危险区域位置
        danger_pos = calculate_danger_zone_position(
            current_mean,
            approach_std,
            direction,
            args.k * distance_multiplier,
            args.buffer,
        )
        
        # 计算危险区域范围（在XY平面上，Z轴使用固定高度范围）
        # BDDL格式: (x_min, y_min, x_max, y_max)
        half_size = args.zone_size / 2.0
        
        zone = {
            "position": danger_pos.tolist(),
            "ranges": [
                danger_pos[0] - half_size,  # x_min
                danger_pos[1] - half_size,  # y_min
                danger_pos[0] + half_size,  # x_max
                danger_pos[1] + half_size,  # y_max
            ],
            "z_range": [danger_pos[2] - 0.05, danger_pos[2] + 0.15],  # Z轴范围
            "offset_direction": offset_direction.tolist(),
        }
        
        zones.append(zone)
        
        logging.info(f"危险区域 {i+1}:")
        logging.info(f"  位置: {danger_pos}")
        logging.info(f"  范围: {zone['ranges']}")
    
    return zones


def save_danger_zones(zones: List[dict], args: Args, stats: dict) -> None:
    """保存危险区域计算结果"""
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    task_suite = stats["task_suite"]
    task_id = stats["task_id"]
    
    # 保存为JSON
    json_file = output_dir / f"{task_suite}_task{task_id}_danger_zones.json"
    
    output_data = {
        "task_suite": task_suite,
        "task_id": task_id,
        "task_description": stats["task_description"],
        "parameters": {
            "k": args.k,
            "buffer": args.buffer,
            "zone_size": args.zone_size,
        },
        "zones": zones,
    }
    
    with open(json_file, "w") as f:
        json.dump(output_data, f, indent=2)
    
    logging.info(f"危险区域计算结果已保存到 {json_file}")
    
    # 保存为numpy格式
    npz_file = output_dir / f"{task_suite}_task{task_id}_danger_zones.npz"
    np.savez(
        npz_file,
        zones=np.array([zone["position"] for zone in zones]),
        ranges=np.array([zone["ranges"] for zone in zones]),
        **output_data,
    )
    
    logging.info(f"危险区域数据已保存到 {npz_file}")


def main(args: Args) -> None:
    """主函数"""
    zones = calculate_danger_zones(args)
    
    stats = load_statistics(args.statistics_file)
    save_danger_zones(zones, args, stats)
    
    logging.info("危险区域计算完成！")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(main)

