"""
BDDL文件更新脚本
自动更新BDDL文件中的危险区域定义
"""
import json
import logging
import pathlib
import re
from typing import List, Dict

import tyro
from dataclasses import dataclass
from libero.libero import get_libero_path


@dataclass
class Args:
    """BDDL更新参数"""
    danger_zones_file: str  # 危险区域JSON文件路径
    backup: bool = True  # 是否创建备份
    target: str = "main_table"  # 危险区域的目标对象（通常是main_table）


def load_danger_zones(danger_zones_file: str) -> Dict:
    """加载危险区域数据"""
    with open(danger_zones_file, "r") as f:
        return json.load(f)


def find_bddl_file(task_suite: str, task_id: int) -> pathlib.Path:
    """找到对应的BDDL文件"""
    bddl_dir = pathlib.Path(get_libero_path("bddl_files"))
    
    # 获取任务套件目录
    suite_dir = bddl_dir / task_suite
    
    if not suite_dir.exists():
        raise FileNotFoundError(f"任务套件目录不存在: {suite_dir}")
    
    # 列出所有BDDL文件
    bddl_files = sorted(suite_dir.glob("*.bddl"))
    
    # 排除backup文件
    bddl_files = [f for f in bddl_files if not f.name.endswith(".backup")]
    
    if task_id >= len(bddl_files):
        raise ValueError(f"任务ID {task_id} 超出范围 (0-{len(bddl_files)-1})")
    
    return bddl_files[task_id]


def parse_bddl_file(bddl_file: pathlib.Path) -> tuple:
    """
    解析BDDL文件
    
    Returns:
        (content, regions_section_start, regions_section_end, regions_content)
    """
    with open(bddl_file, "r") as f:
        content = f.read()
    
    # 找到regions部分
    # 匹配 (:regions ... ) 部分，需要匹配嵌套的括号
    # 使用更精确的匹配，找到 (:regions 到对应的闭合括号
    pattern = r"\(:regions\s*\n"
    match = re.search(pattern, content)
    
    if not match:
        raise ValueError(f"无法找到regions部分: {bddl_file}")
    
    regions_start = match.start()
    # 从 (:regions 之后开始查找对应的闭合括号
    pos = match.end()
    depth = 1  # 已经有一个开括号
    regions_content_start = pos
    
    while pos < len(content) and depth > 0:
        if content[pos] == '(':
            depth += 1
        elif content[pos] == ')':
            depth -= 1
        pos += 1
    
    if depth != 0:
        raise ValueError(f"regions部分括号不匹配: {bddl_file}")
    
    regions_end = pos
    regions_content = content[regions_content_start:regions_end-1]  # 排除最后的闭合括号
    
    return content, regions_start, regions_end, regions_content


def format_zone_ranges(ranges: List[float]) -> str:
    """
    格式化危险区域范围
    
    Args:
        ranges: [x_min, y_min, x_max, y_max]
    
    Returns:
        格式化的字符串
    """
    return f"({ranges[0]:.4f} {ranges[1]:.4f} {ranges[2]:.4f} {ranges[3]:.4f})"


def generate_danger_zone_definition(zone_id: int, target: str, ranges: List[float]) -> str:
    """
    生成危险区域定义
    
    Args:
        zone_id: 区域ID (1, 2, ...)
        target: 目标对象
        ranges: [x_min, y_min, x_max, y_max]
    
    Returns:
        BDDL格式的危险区域定义
    """
    zone_name = f"{target}_danger_zone_{zone_id}"
    ranges_str = format_zone_ranges(ranges)
    
    definition = f"""      ({zone_name}
          (:target {target})
          (:ranges (
              {ranges_str}
            )
          )
      )"""
    
    return definition


def remove_existing_danger_zones(regions_content: str, target: str) -> str:
    """移除现有的危险区域定义"""
    # 匹配所有危险区域定义，需要匹配嵌套的括号
    # 格式: (target_danger_zone_N ... )
    pattern = rf"\({target}_danger_zone_\d+\s*\([^)]*\)[^)]*\)"
    
    # 更精确的匹配：找到完整的危险区域定义（包括嵌套括号）
    lines = regions_content.split('\n')
    cleaned_lines = []
    skip_until_depth = 0
    current_depth = 0
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # 检查是否是危险区域定义的开始
        if re.search(rf"\({target}_danger_zone_\d+", line):
            # 计算这一行的开括号和闭括号
            open_count = line.count('(')
            close_count = line.count(')')
            current_depth = open_count - close_count
            
            # 跳过这个区域定义
            skip_until_depth = current_depth
            i += 1
            
            # 继续跳过直到深度回到0
            while i < len(lines) and skip_until_depth > 0:
                line = lines[i]
                open_count = line.count('(')
                close_count = line.count(')')
                skip_until_depth += open_count - close_count
                i += 1
            continue
        
        cleaned_lines.append(line)
        i += 1
    
    cleaned = '\n'.join(cleaned_lines)
    
    # 清理多余的空行（保留单个空行）
    cleaned = re.sub(r"\n\s*\n\s*\n+", "\n\n", cleaned)
    
    return cleaned


def update_bddl_file(bddl_file: pathlib.Path, zones: List[Dict], target: str, backup: bool = True) -> None:
    """更新BDDL文件中的危险区域"""
    if backup:
        backup_file = bddl_file.with_suffix(".bddl.backup")
        if not backup_file.exists():
            logging.info(f"创建备份文件: {backup_file}")
            with open(bddl_file, "r") as src, open(backup_file, "w") as dst:
                dst.write(src.read())
        else:
            logging.info(f"备份文件已存在: {backup_file}")
    
    # 解析文件
    content, regions_start, regions_end, regions_content = parse_bddl_file(bddl_file)
    
    # 移除现有危险区域
    cleaned_regions = remove_existing_danger_zones(regions_content, target)
    
    # 生成新的危险区域定义
    new_zone_definitions = []
    for i, zone in enumerate(zones, start=1):
        zone_def = generate_danger_zone_definition(i, target, zone["ranges"])
        new_zone_definitions.append(zone_def)
    
    # 组合新的regions内容
    # 在regions末尾添加危险区域
    new_regions_content = cleaned_regions.rstrip()
    if new_regions_content and not new_regions_content.endswith("\n"):
        new_regions_content += "\n"
    
    new_regions_content += "\n".join(new_zone_definitions)
    new_regions_content += "\n"
    
    # 重建文件内容
    new_content = (
        content[:regions_start + len("(:regions\n")]
        + new_regions_content
        + content[regions_end - len(")"):]
    )
    
    # 写入文件
    with open(bddl_file, "w") as f:
        f.write(new_content)
    
    logging.info(f"已更新BDDL文件: {bddl_file}")
    logging.info(f"添加了 {len(zones)} 个危险区域")


def main(args: Args) -> None:
    """主函数"""
    # 加载危险区域数据
    logging.info(f"加载危险区域数据: {args.danger_zones_file}")
    danger_zones_data = load_danger_zones(args.danger_zones_file)
    
    task_suite = danger_zones_data["task_suite"]
    task_id = danger_zones_data["task_id"]
    zones = danger_zones_data["zones"]
    
    logging.info(f"任务套件: {task_suite}, 任务ID: {task_id}")
    logging.info(f"危险区域数量: {len(zones)}")
    
    # 找到BDDL文件
    bddl_file = find_bddl_file(task_suite, task_id)
    logging.info(f"找到BDDL文件: {bddl_file}")
    
    # 更新BDDL文件
    update_bddl_file(bddl_file, zones, args.target, backup=args.backup)
    
    logging.info("BDDL文件更新完成！")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(main)

