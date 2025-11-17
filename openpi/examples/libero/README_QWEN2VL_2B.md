# Qwen2-VL-2B-Instruct 模型测试指南

本指南说明如何使用 Qwen2-VL-2B-Instruct 模型进行 LIBERO 任务评估。

## 目录结构

```
openpi/examples/libero/
├── unified_eval.py          # 统一评估脚本
├── utils.py                 # 工具函数（模型加载）
├── generate_intruction.py   # 指令生成器
├── test_qwen2vl_2b.sh      # Qwen2-VL-2B 测试脚本
└── README_QWEN2VL_2B.md    # 本文档
```

## 快速开始

### 1. 环境准备

确保已安装必要的依赖：

```bash
pip install transformers torch qwen-vl-utils
pip install openai  # 如果使用 API 模式
```

### 2. 启动服务器

在运行评估之前，需要先启动 Policy 服务器和负载均衡器：

```bash
# 在项目根目录
cd /root/autodl-tmp/code/attackVLA/pi0_reward_server

# 启动所有服务（默认使用 GPU 2,3）
bash scripts/start_all_parallel.sh
```

验证服务器状态：

```bash
# 测试负载均衡器
curl http://localhost:6000/health

# 测试所有服务
bash scripts/parallel/test_multi_servers.sh
```

### 3. 运行评估

#### 方式一：使用测试脚本（推荐）

```bash
cd openpi/examples/libero

# 基本用法（使用默认配置）
bash test_qwen2vl_2b.sh

# 自定义配置
TASK_SUITE=libero_object \
NUM_TRIALS=5 \
NUM_INSTRUCTIONS=10 \
SAVE_VIDEOS=true \
bash test_qwen2vl_2b.sh
```

#### 方式二：直接调用 Python 脚本

```bash
cd openpi/examples/libero

python unified_eval.py \
    --mode vlm \
    --backend qwenvl \
    --qwen-mode local \
    --qwen-model-id Qwen/Qwen2-VL-2B-Instruct \
    --host 0.0.0.0 \
    --port 6000 \
    --task-suite-name libero_spatial \
    --num-trials-per-task 2 \
    --num-instructions 5 \
    --select-topk 3 \
    --semantic-type clip \
    --no-use-wandb
```

## 配置参数说明

### 环境变量配置

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `BACKEND` | `qwenvl` | 后端类型 (qwenvl/verl_qwen) |
| `QWEN_MODE` | `local` | 模式 (local/api) |
| `QWEN_MODEL_ID` | `Qwen/Qwen2-VL-2B-Instruct` | 模型ID |
| `HOST` | `0.0.0.0` | 服务器地址 |
| `PORT` | `6000` | 服务器端口 |
| `TASK_SUITE` | `libero_spatial` | 任务集名称 |
| `NUM_TRIALS` | `2` | 每个任务的试验次数 |
| `NUM_INSTRUCTIONS` | `5` | 生成的指令数量 |
| `SELECT_TOPK` | `3` | 选择 Top-K 指令 |
| `SEMANTIC_TYPE` | `clip` | 语义相似度类型 (clip/deberta) |
| `SAVE_VIDEOS` | `false` | 是否保存视频 |
| `USE_WANDB` | `false` | 是否使用 WandB |

### 任务集选项

- `libero_spatial` - 空间关系任务（10个任务，220步）
- `libero_object` - 物体操作任务（10个任务，280步）
- `libero_goal` - 目标导向任务（10个任务，300步）
- `libero_10` - 10个任务集（520步）
- `libero_90` - 90个任务集（400步）

## 模型模式

### Local 模式（本地推理）

使用本地 GPU 进行推理，适合有 GPU 资源的情况。

```bash
# 使用 Qwen2-VL-2B（推荐，显存需求小）
QWEN_MODE=local \
QWEN_MODEL_ID=Qwen/Qwen2-VL-2B-Instruct \
bash test_qwen2vl_2b.sh

# 使用 Qwen2.5-VL-7B（需要更多显存）
QWEN_MODE=local \
QWEN_MODEL_ID=Qwen/Qwen2.5-VL-7B-Instruct \
bash test_qwen2vl_2b.sh
```

**显存需求：**
- Qwen2-VL-2B: ~8GB
- Qwen2.5-VL-7B: ~20GB

### API 模式（云端推理）

使用阿里云 DashScope API，无需本地 GPU。

```bash
# 设置 API Key
export DASHSCOPE_API_KEY="your-api-key-here"

# 运行评估
QWEN_MODE=api \
QWEN_MODEL_ID=qwen2.5-vl-72b-instruct \
bash test_qwen2vl_2b.sh
```

## 输出说明

评估完成后，结果保存在 `output/qwen2vl_2b_<task_suite>_<timestamp>/` 目录：

```
output/qwen2vl_2b_libero_spatial_20251116_161234/
├── config.json                    # 运行配置
├── eval.log                       # 完整日志
├── results/
│   ├── eval_results.json         # 详细结果
│   └── whole_acc_log.json        # 成功率汇总
└── videos/                        # 视频文件（如果启用）
    ├── vlm_task0_inst0_ep0_success_*.mp4
    └── ...
```

### 结果文件格式

**eval_results.json:**
```json
[
  {
    "task_id": 0,
    "task_description": "pick up the book",
    "mode": "vlm",
    "image_url": "...",
    "done_annotations": ["pick up the book", ...],
    "fail_annotations": [...],
    "done_annotations_smi": [0.95, ...],
    "fail_annotations_smi": [...]
  }
]
```

**whole_acc_log.json:**
```json
{
  "libero_spatial": 0.75,
  "mode": "vlm",
  "total_successes": 15,
  "total_episodes": 20,
  "mean_similarity": 0.85,
  "variance_similarity": 0.02
}
```

## 常见问题

### 1. 服务器连接失败

```bash
# 检查服务器状态
curl http://localhost:6000/health

# 如果失败，重启服务器
bash scripts/start_all_parallel.sh
```

### 2. 显存不足

```bash
# 使用更小的模型
QWEN_MODEL_ID=Qwen/Qwen2-VL-2B-Instruct bash test_qwen2vl_2b.sh

# 或使用 API 模式
QWEN_MODE=api bash test_qwen2vl_2b.sh
```

### 3. 模型下载慢

```bash
# 设置 HuggingFace 镜像
export HF_ENDPOINT=https://hf-mirror.com

# 或手动下载模型到本地
git clone https://hf-mirror.com/Qwen/Qwen2-VL-2B-Instruct
QWEN_MODEL_ID=/path/to/Qwen2-VL-2B-Instruct bash test_qwen2vl_2b.sh
```

## 性能优化建议

1. **批量处理**: 增加 `NUM_INSTRUCTIONS` 和 `SELECT_TOPK` 以获得更多样化的指令
2. **并行评估**: 使用多个 GPU 运行不同的任务集
3. **视频保存**: 仅在需要时启用 `SAVE_VIDEOS=true`，可节省磁盘空间和时间
4. **WandB 监控**: 启用 `USE_WANDB=true` 以实时监控评估进度

## 完整示例

```bash
# 1. 启动服务器
cd /root/autodl-tmp/code/attackVLA/pi0_reward_server
bash scripts/start_all_parallel.sh

# 2. 等待服务器就绪（在另一个终端）
sleep 10
curl http://localhost:6000/health

# 3. 运行评估
cd openpi/examples/libero
TASK_SUITE=libero_spatial \
NUM_TRIALS=5 \
NUM_INSTRUCTIONS=10 \
SELECT_TOPK=5 \
SEMANTIC_TYPE=clip \
SAVE_VIDEOS=false \
USE_WANDB=false \
bash test_qwen2vl_2b.sh

# 4. 查看结果
cat output/qwen2vl_2b_*/results/whole_acc_log.json
```

## 代码适配说明

已完成以下适配以支持 Qwen2-VL-2B-Instruct：

1. **utils.py**: 添加了模型加载日志，支持 Qwen2-VL-2B
2. **generate_intruction.py**: 更新注释，明确支持 Qwen2-VL-2B
3. **unified_eval.py**: 无需修改，已支持通过参数配置模型
4. **test_qwen2vl_2b.sh**: 新增专用测试脚本

## 联系方式

如有问题，请查看：
- 项目文档: `/root/autodl-tmp/code/attackVLA/pi0_reward_server/README.md`
- 服务器脚本: `/root/autodl-tmp/code/attackVLA/pi0_reward_server/scripts/`
