"""
危险区域自动放置主脚本
整合轨迹收集、分析和危险区域计算、BDDL更新的完整流程
"""
import logging
import pathlib
import subprocess
import sys
from typing import Optional

import tyro
from dataclasses import dataclass


@dataclass
class Args:
    """自动放置危险区域参数"""
    # 任务参数
    task_suite_name: str = "libero_goal"
    task_id: int = 0
    
    # 轨迹收集参数
    host: str = "0.0.0.0"
    port: int = 4444
    num_episodes: int = 50
    min_successful_episodes: int = 20
    
    # 分析参数
    approach_window: int = 10
    
    # 危险区域计算参数
    k: float = 2.5  # 安全系数
    buffer: float = 0.02  # 缓冲距离（米）
    zone_size: float = 0.08  # 危险区域大小（米）
    num_zones: int = 2  # 危险区域数量
    
    # BDDL更新参数
    target: str = "main_table"  # 危险区域目标对象
    backup: bool = True  # 是否创建备份
    
    # 路径参数
    data_dir: str = "trajectory_profiling/data"
    results_dir: str = "trajectory_profiling/results"
    
    # 流程控制
    skip_collection: bool = False  # 跳过轨迹收集（使用已有数据）
    skip_analysis: bool = False  # 跳过分析（使用已有统计）
    skip_calculation: bool = False  # 跳过计算（使用已有危险区域）
    skip_update: bool = False  # 跳过BDDL更新
    
    seed: int = 7


def run_command(cmd: list, description: str) -> bool:
    """运行命令"""
    logging.info(f"\n{'='*60}")
    logging.info(f"执行: {description}")
    logging.info(f"命令: {' '.join(cmd)}")
    logging.info(f"{'='*60}\n")
    
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=False,
            text=True,
        )
        logging.info(f"✓ {description} 完成")
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"✗ {description} 失败: {e}")
        return False
    except Exception as e:
        logging.error(f"✗ {description} 出错: {e}")
        return False


def auto_place_danger_zones(args: Args) -> None:
    """自动放置危险区域的主流程"""
    logging.info("="*60)
    logging.info("危险区域自动放置系统")
    logging.info("="*60)
    logging.info(f"任务套件: {args.task_suite_name}")
    logging.info(f"任务ID: {args.task_id}")
    logging.info(f"数据目录: {args.data_dir}")
    logging.info(f"结果目录: {args.results_dir}")
    logging.info("="*60)
    
    # 构建文件路径
    data_dir = pathlib.Path(args.data_dir)
    results_dir = pathlib.Path(args.results_dir)
    
    trajectory_file = data_dir / f"{args.task_suite_name}_task{args.task_id}_trajectories.npz"
    statistics_file = results_dir / f"{args.task_suite_name}_task{args.task_id}_statistics.npz"
    danger_zones_file = results_dir / f"{args.task_suite_name}_task{args.task_id}_danger_zones.json"
    
    # 获取脚本目录
    script_dir = pathlib.Path(__file__).parent
    
    success = True
    
    # 步骤1: 收集轨迹
    if not args.skip_collection:
        if trajectory_file.exists() and not args.skip_collection:
            logging.warning(f"轨迹文件已存在: {trajectory_file}")
            response = input("是否重新收集？(y/n): ")
            if response.lower() != 'y':
                logging.info("跳过轨迹收集")
                args.skip_collection = True
        
        if not args.skip_collection:
            cmd = [
                sys.executable,
                str(script_dir / "collect_trajectories.py"),
                "--task_suite_name", args.task_suite_name,
                "--task_id", str(args.task_id),
                "--host", args.host,
                "--port", str(args.port),
                "--num_episodes", str(args.num_episodes),
                "--min_successful_episodes", str(args.min_successful_episodes),
                "--output_dir", args.data_dir,
                "--seed", str(args.seed),
            ]
            success = run_command(cmd, "步骤1: 收集轨迹")
            if not success:
                logging.error("轨迹收集失败，终止流程")
                return
    else:
        logging.info("跳过步骤1: 轨迹收集")
    
    # 检查轨迹文件是否存在
    if not trajectory_file.exists():
        logging.error(f"轨迹文件不存在: {trajectory_file}")
        logging.error("请先运行轨迹收集或禁用 --skip_collection")
        return
    
    # 步骤2: 分析轨迹
    if not args.skip_analysis:
        cmd = [
            sys.executable,
            str(script_dir / "analyze_trajectories.py"),
            "--input_file", str(trajectory_file),
            "--output_dir", args.results_dir,
            "--approach_window", str(args.approach_window),
        ]
        success = run_command(cmd, "步骤2: 分析轨迹")
        if not success:
            logging.error("轨迹分析失败，终止流程")
            return
    else:
        logging.info("跳过步骤2: 轨迹分析")
    
    # 检查统计文件是否存在
    if not statistics_file.exists():
        logging.error(f"统计文件不存在: {statistics_file}")
        logging.error("请先运行轨迹分析或禁用 --skip_analysis")
        return
    
    # 步骤3: 计算危险区域
    if not args.skip_calculation:
        cmd = [
            sys.executable,
            str(script_dir / "calculate_danger_zones.py"),
            "--statistics_file", str(statistics_file),
            "--output_dir", args.results_dir,
            "--k", str(args.k),
            "--buffer", str(args.buffer),
            "--zone_size", str(args.zone_size),
            "--num_zones", str(args.num_zones),
        ]
        success = run_command(cmd, "步骤3: 计算危险区域")
        if not success:
            logging.error("危险区域计算失败，终止流程")
            return
    else:
        logging.info("跳过步骤3: 危险区域计算")
    
    # 检查危险区域文件是否存在
    if not danger_zones_file.exists():
        logging.error(f"危险区域文件不存在: {danger_zones_file}")
        logging.error("请先运行危险区域计算或禁用 --skip_calculation")
        return
    
    # 步骤4: 更新BDDL文件
    if not args.skip_update:
        cmd = [
            sys.executable,
            str(script_dir / "update_bddl_zones.py"),
            "--danger_zones_file", str(danger_zones_file),
            "--target", args.target,
        ]
        if not args.backup:
            # tyro默认处理，如果backup=False，不需要额外参数
            pass
        
        success = run_command(cmd, "步骤4: 更新BDDL文件")
        if not success:
            logging.error("BDDL文件更新失败")
            return
    else:
        logging.info("跳过步骤4: BDDL文件更新")
    
    # 完成
    logging.info("\n" + "="*60)
    logging.info("✓ 危险区域自动放置完成！")
    logging.info("="*60)
    logging.info(f"轨迹文件: {trajectory_file}")
    logging.info(f"统计文件: {statistics_file}")
    logging.info(f"危险区域文件: {danger_zones_file}")
    logging.info("="*60)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    tyro.cli(auto_place_danger_zones)

