#!/usr/bin/env python3
"""调试路径问题"""

import os
import sys

print("="*60)
print("调试路径信息")
print("="*60)

# 当前脚本路径
current_file = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file)

print(f"\n当前脚本: {current_file}")
print(f"当前目录: {current_dir}")

# 添加路径
sys.path.insert(0, current_dir)

# 导入libero
from libero.libero import benchmark

libero_module_path = os.path.dirname(os.path.abspath(benchmark.__file__))
print(f"\nLIBERO模块路径: {libero_module_path}")

# 获取任务
benchmark_dict = benchmark.get_benchmark_dict()
task_suite = benchmark_dict["libero_spatial"]()
task = task_suite.get_task(0)

print(f"\n任务信息:")
print(f"  名称: {task.name}")
print(f"  problem_folder: {task.problem_folder}")
print(f"  problem_folder类型: {type(task.problem_folder)}")
print(f"  problem_folder是否存在: {os.path.exists(task.problem_folder)}")

# 如果不存在，尝试找到正确路径
if not os.path.exists(task.problem_folder):
    print(f"\n❌ 路径不存在，尝试查找...")
    
    # 尝试不同的路径组合
    possible_paths = [
        task.problem_folder,
        os.path.join(current_dir, task.problem_folder),
        os.path.join(libero_module_path, task.problem_folder),
        os.path.join(current_dir, 'libero', task.problem_folder),
    ]
    
    for i, path in enumerate(possible_paths):
        abs_path = os.path.abspath(path)
        exists = os.path.exists(abs_path)
        print(f"\n尝试 {i+1}: {abs_path}")
        print(f"  存在: {exists}")
        if exists:
            print(f"  ✅ 找到正确路径！")
            break
else:
    print(f"\n✅ 路径存在")
    abs_path = os.path.abspath(task.problem_folder)
    print(f"  绝对路径: {abs_path}")

print("\n" + "="*60)
