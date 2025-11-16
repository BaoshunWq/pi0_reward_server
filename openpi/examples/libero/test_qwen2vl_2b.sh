#!/usr/bin/env bash
# 测试 Qwen2-VL-2B-Instruct 模型的评估脚本

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# ==========================================
# 配置参数
# ==========================================

# 模型配置
export BACKEND="${BACKEND:-qwenvl}"
export QWEN_MODE="${QWEN_MODE:-local}"  # local 或 api
export QWEN_MODEL_ID="${QWEN_MODEL_ID:-Qwen/Qwen2-VL-2B-Instruct}"

# 服务器配置
export HOST="${HOST:-0.0.0.0}"
export PORT="${PORT:-6000}"  # 负载均衡器端口

# 任务配置
export TASK_SUITE="${TASK_SUITE:-libero_spatial}"
export NUM_TRIALS="${NUM_TRIALS:-2}"
export NUM_INSTRUCTIONS="${NUM_INSTRUCTIONS:-5}"
export SELECT_TOPK="${SELECT_TOPK:-3}"

# 语义相似度配置
export SEMANTIC_TYPE="${SEMANTIC_TYPE:-clip}"

# 输出配置
export SAVE_VIDEOS="${SAVE_VIDEOS:-false}"
export USE_WANDB="${USE_WANDB:-false}"

# GPU配置
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# 时间戳
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
export OUTPUT_DIR="output/qwen2vl_2b_${TASK_SUITE}_${TIMESTAMP}"

echo "=========================================="
echo "Qwen2-VL-2B-Instruct 模型测试"
echo "=========================================="
echo "后端: ${BACKEND}"
echo "模式: ${QWEN_MODE}"
echo "模型: ${QWEN_MODEL_ID}"
echo "服务器: ${HOST}:${PORT}"
echo "任务集: ${TASK_SUITE}"
echo "试验次数: ${NUM_TRIALS}"
echo "生成指令数: ${NUM_INSTRUCTIONS}"
echo "Top-K选择: ${SELECT_TOPK}"
echo "语义类型: ${SEMANTIC_TYPE}"
echo "保存视频: ${SAVE_VIDEOS}"
echo "使用WandB: ${USE_WANDB}"
echo "输出目录: ${OUTPUT_DIR}"
echo "=========================================="

# ==========================================
# 检查依赖
# ==========================================
echo ""
echo "检查依赖..."

# 检查Python环境
if ! command -v python &> /dev/null; then
    echo "错误: 未找到 python" >&2
    exit 1
fi

echo "Python版本: $(python --version)"

# 检查必要的Python包
python -c "import transformers; print(f'transformers: {transformers.__version__}')" || {
    echo "错误: 未安装 transformers" >&2
    exit 1
}

python -c "import torch; print(f'torch: {torch.__version__}')" || {
    echo "错误: 未安装 torch" >&2
    exit 1
}

# 检查服务器是否运行
echo ""
echo "检查服务器状态..."
if curl -s "http://${HOST}:${PORT}/health" > /dev/null 2>&1; then
    echo "✓ 服务器运行正常 (${HOST}:${PORT})"
else
    echo "警告: 服务器未响应 (${HOST}:${PORT})"
    echo "请先启动服务器: bash scripts/start_all_parallel.sh"
    read -p "是否继续? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# ==========================================
# 准备输出目录
# ==========================================
mkdir -p "${OUTPUT_DIR}/videos"
mkdir -p "${OUTPUT_DIR}/results"

# ==========================================
# 构建命令参数
# ==========================================
CMD_ARGS=(
    --mode vlm
    --backend "${BACKEND}"
    --qwen-mode "${QWEN_MODE}"
    --qwen-model-id "${QWEN_MODEL_ID}"
    --host "${HOST}"
    --port "${PORT}"
    --task-suite-name "${TASK_SUITE}"
    --num-trials-per-task "${NUM_TRIALS}"
    --num-instructions "${NUM_INSTRUCTIONS}"
    --select-topk "${SELECT_TOPK}"
    --semantic-type "${SEMANTIC_TYPE}"
    --output-path "${OUTPUT_DIR}/results/eval_results.json"
    --whole-acc-log-path "${OUTPUT_DIR}/results/whole_acc_log.json"
)

# 添加可选参数
if [[ "${SAVE_VIDEOS}" == "true" ]]; then
    CMD_ARGS+=(--save-videos)
    CMD_ARGS+=(--video-out-path "${OUTPUT_DIR}/videos")
fi

if [[ "${USE_WANDB}" == "true" ]]; then
    CMD_ARGS+=(--use-wandb)
    CMD_ARGS+=(--wandb-project "qwen2vl_2b_libero_eval")
else
    CMD_ARGS+=(--no-use-wandb)
fi

# ==========================================
# 运行评估
# ==========================================
echo ""
echo "=========================================="
echo "开始评估..."
echo "=========================================="
echo ""

# 保存配置
cat > "${OUTPUT_DIR}/config.json" <<EOF
{
    "backend": "${BACKEND}",
    "qwen_mode": "${QWEN_MODE}",
    "qwen_model_id": "${QWEN_MODEL_ID}",
    "host": "${HOST}",
    "port": ${PORT},
    "task_suite": "${TASK_SUITE}",
    "num_trials": ${NUM_TRIALS},
    "num_instructions": ${NUM_INSTRUCTIONS},
    "select_topk": ${SELECT_TOPK},
    "semantic_type": "${SEMANTIC_TYPE}",
    "save_videos": ${SAVE_VIDEOS},
    "use_wandb": ${USE_WANDB},
    "timestamp": "${TIMESTAMP}"
}
EOF

echo "配置已保存到: ${OUTPUT_DIR}/config.json"
echo ""

# 运行评估（捕获日志）
python unified_eval.py "${CMD_ARGS[@]}" 2>&1 | tee "${OUTPUT_DIR}/eval.log"

EXIT_CODE=${PIPESTATUS[0]}

# ==========================================
# 结果汇总
# ==========================================
echo ""
echo "=========================================="
if [[ ${EXIT_CODE} -eq 0 ]]; then
    echo "✓ 评估完成"
else
    echo "✗ 评估失败 (退出码: ${EXIT_CODE})"
fi
echo "=========================================="
echo ""
echo "输出文件:"
echo "  配置: ${OUTPUT_DIR}/config.json"
echo "  日志: ${OUTPUT_DIR}/eval.log"
echo "  结果: ${OUTPUT_DIR}/results/eval_results.json"
echo "  汇总: ${OUTPUT_DIR}/results/whole_acc_log.json"
if [[ "${SAVE_VIDEOS}" == "true" ]]; then
    echo "  视频: ${OUTPUT_DIR}/videos/"
fi
echo ""

# 显示成功率（如果结果文件存在）
if [[ -f "${OUTPUT_DIR}/results/whole_acc_log.json" ]]; then
    echo "成功率统计:"
    python -c "
import json
try:
    with open('${OUTPUT_DIR}/results/whole_acc_log.json', 'r') as f:
        data = json.load(f)
    if isinstance(data, list) and len(data) > 0:
        data = data[-1]  # 取最后一条记录
    print(f\"  任务集: {data.get('mode', 'N/A')} - {list(data.keys())[0] if data else 'N/A'}\")
    for key, value in data.items():
        if key not in ['mode', 'total_successes', 'total_episodes', 'mean_similarity', 'variance_similarity']:
            print(f\"  成功率: {value:.2%}\")
        elif key == 'total_episodes':
            print(f\"  总试验数: {value}\")
        elif key == 'total_successes':
            print(f\"  成功次数: {value}\")
except Exception as e:
    print(f\"  无法读取结果: {e}\")
" || echo "  无法解析结果文件"
fi

echo ""
echo "=========================================="

exit ${EXIT_CODE}
