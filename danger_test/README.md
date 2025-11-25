## 指令敏感性危险区实验

本实验用于验证 VLA 模型（OpenVLA / RT-2 等）对“轨迹修饰词”是否敏感，并结合我们在 Libero 中加入的红色危险区域与碰撞代价，评估潜在失效风险。

### 依赖准备
- 已安装 `openpi` 及其所有 Libero 依赖（参见 `openpi/README.md`）。
- 已启动对应的策略服务（例如 websocket 版 OpenVLA），默认监听 `0.0.0.0:8000`。
- 运行脚本前请在仓库根目录下执行，以便 `PYTHONPATH` 能找到 `openpi` 与 `libero`。

### 运行步骤
1. 固定随机种子并启动策略服务。
2. 在仓库根目录执行：
   ```
   python danger_test/instruction_sensitivity_experiment.py \
     --task_suite_name libero_spatial \
     --task_id 0 \
     --init_state_index 0 \
     --max_steps 220 \
     --num_steps_wait 10
   ```
   可使用 `--instructions_json path/to/custom.json` 覆盖默认指令组。
3. 结果存于 `danger_test/results/`：
   - `instruction_sensitivity_metrics.json`: 逐条指令与基线轨迹之间的 DTW / 对齐 L2 距离及碰撞累计。
   - `trajectories.npz`: 每条指令对应的末端执行器轨迹张量。
   - 若加上 `--save_video True`，每条指令的可视化视频保存在 `danger_test/results/videos/`。

### 评价指标
- `dtw_distance`：轨迹形状差异，约等于 0 表示“耳聋”。
- `l2_aligned_distance`：对齐到最短长度的逐步欧氏距离均值。
- `collision_stats`：从环境 `info` 中自动抓取键名包含 `collision` 的统计量，便于核对红区碰撞 cost。

### 自定义
- 通过 JSON（字段 `name/text/category`）指定更多修饰指令。
- 修改 `--task_suite_name` / `--task_id` 可切换至其他 Pick & Place 任务。
- 结果文件可直接供后续 RL 微调或分析脚本读取，进一步评估“最大化碰撞 cost”与任务成功率之间的权衡。

