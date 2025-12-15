#!/usr/bin/env python3
"""测试危险区域自动放置工具"""

import os
import sys

# 确保正确的路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from tools.danger_zone_auto_placement import DangerZoneAutoPlacement
from libero.libero import benchmark

print("="*60)
print("测试危险区域自动放置工具")
print("="*60)

# 获取任务
benchmark_dict = benchmark.get_benchmark_dict()
task_suite = benchmark_dict["libero_spatial"]()
task = task_suite.get_task(0)

print(f"\n任务名称: {task.name}")
print(f"任务文件夹: {task.problem_folder}")
print(f"BDDL文件名: {task.bddl_file}")

# 构建完整的BDDL文件路径
# BDDL文件位于: libero/libero/bddl_files/{problem_folder}/{bddl_file}
libero_root = os.path.join(current_dir, 'libero', 'libero')
bddl_path = os.path.join(libero_root, 'bddl_files', task.problem_folder, task.bddl_file)

print(f"完整BDDL路径: {bddl_path}")

# 检查文件是否存在
if not os.path.exists(bddl_path):
    print(f"❌ 错误: BDDL文件不存在: {bddl_path}")
    sys.exit(1)

print(f"✅ BDDL文件存在")

# 创建分析器
print("\n创建分析器...")
analyzer = DangerZoneAutoPlacement(bddl_path)

# 分析布局
print("\n步骤1: 分析物体布局...")
try:
    layout_info = analyzer.analyze_task_layout(num_samples=5)  # 少量采样以加快速度
    print(f"✅ 分析完成，检测到 {len(layout_info['objects'])} 个物体")
except Exception as e:
    print(f"❌ 分析失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 推荐危险区域
print("\n步骤2: 推荐危险区域...")
try:
    zones = analyzer.recommend_danger_zones(num_zones=2)
    print(f"✅ 推荐了 {len(zones)} 个危险区域")
except Exception as e:
    print(f"❌ 推荐失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 可视化
print("\n步骤3: 生成可视化...")
try:
    output_path = os.path.join(current_dir, "danger_zone_result.png")
    analyzer.visualize_recommendations(zones, save_path=output_path)
    print(f"✅ 可视化已保存到: {output_path}")
except Exception as e:
    print(f"❌ 可视化失败: {e}")
    import traceback
    traceback.print_exc()

# 导出BDDL格式
print("\n步骤4: 导出BDDL格式...")
try:
    bddl_format = analyzer.export_to_bddl_format(zones)
    print(bddl_format)
    
    # 保存到文件
    output_bddl = os.path.join(current_dir, "danger_zone_result.txt")
    with open(output_bddl, 'w') as f:
        f.write(f"; Task: {task.name}\n")
        f.write(bddl_format)
    print(f"\n✅ BDDL格式已保存到: {output_bddl}")
except Exception as e:
    print(f"❌ 导出失败: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("✅ 测试完成！")
print("="*60)