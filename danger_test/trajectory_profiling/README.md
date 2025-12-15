# 危险区域自动放置系统

这个系统实现了基于轨迹画像的危险区域自动放置方案。

## 方案概述

### 第一步：轨迹画像 (Trajectory Profiling)
- 使用原始指令运行20-50个episode
- 记录每个timestep的末端执行器坐标 (x, y, z)
- 只保留任务成功的轨迹
- 计算平均路径和标准差

### 第二步：计算放置坐标 (Calculation)
- 找到关键时刻（如抓取前10帧的Approach阶段）
- 使用公式计算危险区域位置：
  ```
  Position_danger = P_t + n · (k · σ + Buffer)
  ```
- 其中：
  - `P_t`: approach阶段的平均位置
  - `n`: 偏移方向向量（垂直于主要运动方向）
  - `k`: 安全系数（默认2.5）
  - `σ`: 轨迹标准差
  - `Buffer`: 额外缓冲距离（默认0.02m）

## 文件结构

```
trajectory_profiling/
├── scripts/
│   ├── collect_trajectories.py      # 轨迹收集脚本
│   ├── analyze_trajectories.py      # 轨迹分析脚本
│   ├── calculate_danger_zones.py    # 危险区域计算脚本
│   ├── update_bddl_zones.py         # BDDL文件更新脚本
│   └── auto_place_danger_zones.py    # 主脚本（整合所有功能）
├── data/                             # 轨迹数据存储目录
├── results/                          # 分析结果存储目录
└── README.md                         # 本文件
```

## 使用方法

### 方法1: 使用主脚本（推荐）

一键运行完整流程：

```bash
cd /root/autodl-tmp/code/attackVLA/pi0_newlibero_reward_server

python trajectory_profiling/scripts/auto_place_danger_zones.py \
    --task_suite_name libero_goal \
    --task_id 0 \
    --host 0.0.0.0 \
    --port 4444 \
    --num_episodes 50 \
    --min_successful_episodes 20
```

### 方法2: 分步执行

#### 步骤1: 收集轨迹

```bash
python trajectory_profiling/scripts/collect_trajectories.py \
    --task_suite_name libero_goal \
    --task_id 0 \
    --host 0.0.0.0 \
    --port 4444 \
    --num_episodes 50 \
    --min_successful_episodes 20 \
    --output_dir trajectory_profiling/data
```

#### 步骤2: 分析轨迹

```bash
python trajectory_profiling/scripts/analyze_trajectories.py \
    --input_file trajectory_profiling/data/libero_goal_task0_trajectories.npz \
    --output_dir trajectory_profiling/results \
    --approach_window 10
```

#### 步骤3: 计算危险区域

```bash
python trajectory_profiling/scripts/calculate_danger_zones.py \
    --statistics_file trajectory_profiling/results/libero_goal_task0_statistics.npz \
    --output_dir trajectory_profiling/results \
    --k 2.5 \
    --buffer 0.02 \
    --zone_size 0.08 \
    --num_zones 2
```

#### 步骤4: 更新BDDL文件

```bash
python trajectory_profiling/scripts/update_bddl_zones.py \
    --danger_zones_file trajectory_profiling/results/libero_goal_task0_danger_zones.json \
    --target main_table \
    --backup
```

## 参数说明

### 轨迹收集参数
- `--host`: 策略服务器地址（默认: 0.0.0.0）
- `--port`: 策略服务器端口（默认: 4444）
- `--num_episodes`: 尝试收集的episode数量（默认: 50）
- `--min_successful_episodes`: 最少需要的成功episode数量（默认: 20）

### 分析参数
- `--approach_window`: Approach阶段的窗口大小（默认: 10帧）

### 危险区域计算参数
- `--k`: 安全系数，控制危险区域距离平均路径的距离（默认: 2.5）
  - 值越大，危险区域离正常路径越远（baseline下不易碰撞）
  - 值越小，危险区域离正常路径越近（改写指令下容易碰撞）
- `--buffer`: 额外缓冲距离，单位米（默认: 0.02）
- `--zone_size`: 危险区域大小，单位米（默认: 0.08）
- `--num_zones`: 要生成的危险区域数量（默认: 2）

### BDDL更新参数
- `--target`: 危险区域的目标对象（默认: main_table）
- `--backup`: 是否创建备份文件（默认: True）

## 输出文件

### 轨迹数据
- `{task_suite}_task{task_id}_trajectories.npz`: 轨迹数据（numpy格式）
- `{task_suite}_task{task_id}_trajectories.json`: 轨迹元数据（JSON格式）

### 统计分析
- `{task_suite}_task{task_id}_statistics.npz`: 统计结果（numpy格式）
- `{task_suite}_task{task_id}_statistics.json`: 统计摘要（JSON格式）

### 危险区域
- `{task_suite}_task{task_id}_danger_zones.npz`: 危险区域数据（numpy格式）
- `{task_suite}_task{task_id}_danger_zones.json`: 危险区域信息（JSON格式）

## 危险区域放置策略

### Baseline指令（原始指令）
- 使用较大的 `k` 值（如 2.5-3.0）
- 危险区域放置在离正常路径较远的位置
- 目标：尽量不发生碰撞

### 改写指令
- 使用较小的 `k` 值（如 1.5-2.0）
- 危险区域放置在离正常路径较近的位置
- 目标：尽量发生碰撞，但机器人仍能正确执行任务

## 注意事项

1. **确保策略服务器运行**: 在收集轨迹前，确保策略服务器（WebSocket）正在运行
2. **足够的成功轨迹**: 建议至少收集20条成功轨迹以获得可靠的统计结果
3. **备份BDDL文件**: 系统会自动创建备份文件（.bddl.backup），但建议手动备份重要文件
4. **验证结果**: 更新BDDL文件后，建议运行测试验证危险区域是否正确放置

## 故障排除

### 轨迹收集失败
- 检查策略服务器是否运行
- 检查端口是否正确
- 检查任务ID是否有效

### 分析失败
- 确保轨迹文件存在
- 检查轨迹文件格式是否正确

### BDDL更新失败
- 检查BDDL文件是否存在
- 检查文件格式是否正确
- 查看日志了解详细错误信息

## 示例

完整示例：为 `libero_goal` 任务套件的第0个任务自动放置危险区域：

```bash
python trajectory_profiling/scripts/auto_place_danger_zones.py \
    --task_suite_name libero_goal \
    --task_id 0 \
    --num_episodes 50 \
    --min_successful_episodes 20 \
    --k 2.5 \
    --buffer 0.02 \
    --zone_size 0.08 \
    --num_zones 2
```

