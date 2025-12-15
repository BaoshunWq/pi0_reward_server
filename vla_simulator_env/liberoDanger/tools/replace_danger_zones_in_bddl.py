#!/usr/bin/env python3
"""
将生成的危险区域替换到原始BDDL文件中

功能：
1. 读取danger_zone_analysis中生成的危险区域
2. 找到对应的BDDL文件
3. 注释掉原有的危险区域定义
4. 添加新生成的危险区域定义
5. 处理target名称映射（如main_table -> living_room_table）
"""

import os
import sys
import re
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from libero.libero import benchmark


# Target名称映射表（根据不同的任务套件和场景）
TARGET_NAME_MAPPING = {
    'libero_10': {
        # libero_10中的场景使用不同的桌子名称
        'main_table': {
            'LIVING_ROOM': 'living_room_table',
            'KITCHEN': 'kitchen_table',
            'STUDY': 'study_table',
        }
    },
    # libero_spatial使用main_table
    'libero_spatial': {},
    # libero_object使用floor（不是table！）
    'libero_object': {
        'main_table': 'floor'  # 物体放在地板上，不是桌子上
    },
    # libero_goal使用main_table
    'libero_goal': {},
}


def extract_scene_from_task_name(task_name: str) -> str:
    """从任务名称中提取场景名称"""
    if 'LIVING_ROOM' in task_name:
        return 'LIVING_ROOM'
    elif 'KITCHEN' in task_name:
        return 'KITCHEN'
    elif 'STUDY' in task_name:
        return 'STUDY'
    return None


def get_target_mapping(suite_name: str, task_name: str) -> Dict[str, str]:
    """获取target名称映射"""
    mapping = {}
    
    if suite_name in TARGET_NAME_MAPPING:
        suite_mapping = TARGET_NAME_MAPPING[suite_name]
        
        # 对于libero_10，需要根据场景确定映射
        if suite_name == 'libero_10':
            scene = extract_scene_from_task_name(task_name)
            if scene and 'main_table' in suite_mapping:
                scene_mapping = suite_mapping['main_table']
                if scene in scene_mapping:
                    mapping['main_table'] = scene_mapping[scene]
        
        # 对于libero_object，直接使用映射表
        elif suite_name == 'libero_object':
            mapping = suite_mapping.copy()
    
    return mapping


def read_generated_zones(zone_file: str) -> Tuple[str, List[str]]:
    """读取生成的危险区域文件
    
    Returns:
        (task_name, zone_definitions)
    """
    with open(zone_file, 'r') as f:
        content = f.read()
    
    # 提取任务名称
    task_match = re.search(r'; Task: (.+)', content)
    task_name = task_match.group(1) if task_match else None
    
    # 提取所有危险区域定义
    zone_pattern = r'\(auto_danger_zone_\d+.*?\n\)'
    zones = re.findall(zone_pattern, content, re.DOTALL)
    
    return task_name, zones


def apply_target_mapping(zone_def: str, mapping: Dict[str, str]) -> str:
    """应用target名称映射"""
    for old_target, new_target in mapping.items():
        zone_def = zone_def.replace(f'(:target {old_target})', f'(:target {new_target})')
    
    return zone_def


def extract_ranges_from_zone(zone_def: str) -> str:
    """从危险区域定义中提取ranges参数
    
    Returns:
        ranges字符串，例如: "(0.2782 -0.1764 0.3582 -0.0964)"
    """
    # 匹配 (:ranges ( ... ))
    ranges_match = re.search(r'\(:ranges\s*\(\s*\n?\s*\(([^)]+)\)', zone_def, re.DOTALL)
    if ranges_match:
        return ranges_match.group(1).strip()
    return None


def find_danger_zones_in_bddl(bddl_content: str) -> List[Dict]:
    """在BDDL内容中查找所有危险区域定义
    
    Returns:
        List of {name, start_pos, end_pos, full_text, ranges}
    """
    zones = []
    
    # 查找所有danger_zone定义（更精确的模式）
    pattern = r'\((\w*danger_zone_\d+)\s*\n.*?\(:ranges\s*\(\s*\n?\s*\(([^)]+)\).*?\n\s*\)\s*\n\s*\)'
    
    for match in re.finditer(pattern, bddl_content, re.DOTALL):
        zone_name = match.group(1)
        ranges = match.group(2).strip()
        start_pos = match.start()
        end_pos = match.end()
        full_text = match.group(0)
        
        zones.append({
            'name': zone_name,
            'start_pos': start_pos,
            'end_pos': end_pos,
            'full_text': full_text,
            'ranges': ranges
        })
    
    return zones


def replace_zone_ranges(bddl_content: str, old_zones: List[Dict], new_ranges: List[str]) -> str:
    """替换危险区域的ranges参数，保持其他部分不变
    
    Args:
        bddl_content: BDDL文件内容
        old_zones: 原有的危险区域信息列表
        new_ranges: 新的ranges参数列表
    
    Returns:
        替换后的BDDL内容
    """
    if not old_zones or not new_ranges:
        return bddl_content
    
    # 从后往前处理，避免位置偏移
    zones_sorted = sorted(old_zones, key=lambda x: x['start_pos'], reverse=True)
    
    for i, zone_info in enumerate(zones_sorted):
        if i >= len(new_ranges):
            break
        
        old_ranges = zone_info['ranges']
        new_range = new_ranges[len(new_ranges) - 1 - i]  # 倒序对应
        
        # 在原定义中替换ranges
        old_text = zone_info['full_text']
        new_text = old_text.replace(old_ranges, new_range)
        
        # 替换内容
        start_pos = zone_info['start_pos']
        end_pos = zone_info['end_pos']
        bddl_content = bddl_content[:start_pos] + new_text + bddl_content[end_pos:]
        
        print(f"  ✅ 替换 {zone_info['name']}:")
        print(f"     旧: ({old_ranges})")
        print(f"     新: ({new_range})")
    
    return bddl_content


def extract_new_ranges_from_generated(zone_defs: List[str]) -> List[str]:
    """从生成的危险区域定义中提取ranges参数
    
    Args:
        zone_defs: 生成的危险区域定义列表（完整的BDDL格式）
    
    Returns:
        ranges参数列表，例如: ["0.2782 -0.1764 0.3582 -0.0964", ...]
    """
    ranges_list = []
    
    for zone_def in zone_defs:
        # 匹配 (:ranges ( (x1 y1 x2 y2) ))
        ranges_match = re.search(r'\(:ranges\s*\(\s*\n?\s*\(([^)]+)\)', zone_def, re.DOTALL)
        if ranges_match:
            ranges = ranges_match.group(1).strip()
            ranges_list.append(ranges)
    
    return ranges_list


def process_single_task(suite_name: str, task_id: int, 
                       zone_file: str, bddl_file: str,
                       dry_run: bool = False) -> bool:
    """处理单个任务的BDDL文件
    
    Args:
        suite_name: 任务套件名称
        task_id: 任务ID
        zone_file: 生成的危险区域文件路径
        bddl_file: 原始BDDL文件路径
        dry_run: 如果为True，只打印不实际修改
    
    Returns:
        是否成功
    """
    print(f"\n{'='*80}")
    print(f"处理: {suite_name} - Task {task_id}")
    print(f"{'='*80}")
    
    # 检查文件是否存在
    if not os.path.exists(zone_file):
        print(f"❌ 危险区域文件不存在: {zone_file}")
        return False
    
    if not os.path.exists(bddl_file):
        print(f"❌ BDDL文件不存在: {bddl_file}")
        return False
    
    # 读取生成的危险区域
    task_name, new_zone_defs = read_generated_zones(zone_file)
    print(f"任务名称: {task_name}")
    print(f"新危险区域数量: {len(new_zone_defs)}")
    
    # 从生成的定义中提取ranges参数
    new_ranges = extract_new_ranges_from_generated(new_zone_defs)
    print(f"提取的新ranges: {new_ranges}")
    
    # 读取原始BDDL文件
    with open(bddl_file, 'r') as f:
        bddl_content = f.read()
    
    # 查找旧的危险区域
    old_zones = find_danger_zones_in_bddl(bddl_content)
    print(f"原有危险区域数量: {len(old_zones)}")
    if old_zones:
        print(f"原有危险区域: {[z['name'] for z in old_zones]}")
    
    # 检查数量是否匹配
    if len(old_zones) != len(new_ranges):
        print(f"⚠️  警告: 原有危险区域数量({len(old_zones)}) 与 新ranges数量({len(new_ranges)}) 不匹配")
    
    # 替换危险区域的ranges参数（保持名称和结构不变）
    bddl_content = replace_zone_ranges(bddl_content, old_zones, new_ranges)
    
    if dry_run:
        print("\n[DRY RUN] 修改后的内容预览:")
        print("-" * 80)
        # 只显示危险区域相关的部分
        lines = bddl_content.split('\n')
        for i, line in enumerate(lines):
            if 'danger_zone' in line.lower() or 'OLD - Commented' in line:
                start = max(0, i - 2)
                end = min(len(lines), i + 10)
                print('\n'.join(f"{j:4d}: {lines[j]}" for j in range(start, end)))
                break
        print("-" * 80)
        print("✅ [DRY RUN] 预览完成，未实际修改文件")
    else:
        # 备份原文件
        backup_file = bddl_file + '.backup'
        with open(backup_file, 'w') as f:
            with open(bddl_file, 'r') as orig:
                f.write(orig.read())
        print(f"✅ 已备份原文件到: {backup_file}")
        
        # 写入修改后的内容
        with open(bddl_file, 'w') as f:
            f.write(bddl_content)
        print(f"✅ 已更新BDDL文件: {bddl_file}")
    
    return True


def process_all_suites(analysis_dir: str = "./danger_zone_analysis",
                       dry_run: bool = False):
    """处理所有任务套件"""
    
    analysis_path = Path(analysis_dir)
    if not analysis_path.exists():
        print(f"❌ 分析目录不存在: {analysis_dir}")
        return
    
    benchmark_dict = benchmark.get_benchmark_dict()
    suite_names = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
    
    stats = {
        'total': 0,
        'success': 0,
        'failed': 0,
        'skipped': 0
    }
    
    for suite_name in suite_names:
        print(f"\n{'#'*80}")
        print(f"# 处理套件: {suite_name}")
        print(f"{'#'*80}")
        
        try:
            task_suite = benchmark_dict[suite_name]()
            
            for task_id in range(task_suite.n_tasks):
                stats['total'] += 1
                
                # 生成的危险区域文件
                zone_file = analysis_path / f"{suite_name}_task{task_id}_zones.txt"
                
                if not zone_file.exists():
                    print(f"\n⚠️  跳过 Task {task_id}: 危险区域文件不存在")
                    stats['skipped'] += 1
                    continue
                
                # 获取任务信息
                task = task_suite.get_task(task_id)
                
                # 原始BDDL文件
                libero_root = os.path.join(os.path.dirname(__file__), '..', 'libero', 'libero')
                bddl_file = os.path.join(libero_root, 'bddl_files', 
                                        task.problem_folder, task.bddl_file)
                bddl_file = os.path.abspath(bddl_file)
                
                # 处理单个任务
                success = process_single_task(
                    suite_name, task_id, 
                    str(zone_file), bddl_file,
                    dry_run=dry_run
                )
                
                if success:
                    stats['success'] += 1
                else:
                    stats['failed'] += 1
        
        except Exception as e:
            print(f"❌ 处理套件 {suite_name} 时出错: {e}")
            import traceback
            traceback.print_exc()
    
    # 打印统计信息
    print(f"\n{'='*80}")
    print("处理完成统计")
    print(f"{'='*80}")
    print(f"总任务数: {stats['total']}")
    print(f"成功: {stats['success']}")
    print(f"失败: {stats['failed']}")
    print(f"跳过: {stats['skipped']}")
    print(f"{'='*80}")
    
    if not dry_run:
        print("\n⚠️  注意: 原始BDDL文件已备份为 .backup 文件")
        print("如需恢复，可以使用备份文件")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="将生成的危险区域替换到原始BDDL文件中"
    )
    parser.add_argument(
        "--analysis_dir", 
        default="./danger_zone_analysis",
        help="危险区域分析结果目录"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="预览模式，不实际修改文件"
    )
    
    args = parser.parse_args()
    
    if args.dry_run:
        print("=" * 80)
        print("预览模式 (DRY RUN) - 不会实际修改文件")
        print("=" * 80)
    
    process_all_suites(args.analysis_dir, dry_run=args.dry_run)
