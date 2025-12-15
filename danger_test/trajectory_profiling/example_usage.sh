#!/bin/bash
# 危险区域自动放置系统使用示例

# 设置工作目录
cd /root/autodl-tmp/code/attackVLA/pi0_newlibero_reward_server

# 示例1: 完整流程（推荐）
# 为 libero_goal 任务套件的第0个任务自动放置危险区域
python trajectory_profiling/scripts/auto_place_danger_zones.py \
    --task_suite_name libero_goal \
    --task_id 0 \
    --host 0.0.0.0 \
    --port 4444 \
    --num_episodes 50 \
    --min_successful_episodes 20 \
    --k 2.5 \
    --buffer 0.02 \
    --zone_size 0.08 \
    --num_zones 2

# 示例2: 只更新BDDL文件（使用已有数据）
# python trajectory_profiling/scripts/auto_place_danger_zones.py \
#     --task_suite_name libero_goal \
#     --task_id 0 \
#     --skip_collection \
#     --skip_analysis \
#     --skip_calculation

# 示例3: Baseline指令（不易碰撞）- 使用较大的k值
# python trajectory_profiling/scripts/auto_place_danger_zones.py \
#     --task_suite_name libero_goal \
#     --task_id 0 \
#     --k 3.0 \
#     --buffer 0.03

# 示例4: 改写指令（易碰撞）- 使用较小的k值
# python trajectory_profiling/scripts/auto_place_danger_zones.py \
#     --task_suite_name libero_goal \
#     --task_id 0 \
#     --k 1.5 \
#     --buffer 0.01

