#!/usr/bin/env bash
# 测试 Qwen2-VL-2B-Instruct 模型的评估脚本
# 支持运行多个 task suite: libero_spatial, libero_object, libero_goal, libero_10

set -euo pipefail

# export PYTHONPATH=$PYTHONPATH:$PWD/openpi/third_party/libero
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
export PORT="${PORT:-4444}"  # 负载均衡器端口

# 任务配置
# 支持多个 task suite，用空格分隔
export TASK_SUITES="${TASK_SUITES:-libero_object libero_goal libero_10 libero_spatial}"
export NUM_TRIALS="${NUM_TRIALS:-50}"
export NUM_INSTRUCTIONS="${NUM_INSTRUCTIONS:-3}"
export SELECT_TOPK="${SELECT_TOPK:-1}"

# 语义相似度配置
export SEMANTIC_TYPE="${SEMANTIC_TYPE:-clip}"

# 输出配置
export SAVE_VIDEOS="${SAVE_VIDEOS:-false}"
export USE_WANDB="${USE_WANDB:-false}"

# GPU配置
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# 时间戳（全局）
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BASE_OUTPUT_DIR="output/qwen2vl_2b_all_suites_${TIMESTAMP}"

echo "=========================================="
echo "Qwen2-VL-2B-Instruct 模型测试"
echo "=========================================="
echo "后端: ${BACKEND}"
echo "模式: ${QWEN_MODE}"
echo "模型: ${QWEN_MODEL_ID}"
echo "服务器: ${HOST}:${PORT}"
echo "任务集: ${TASK_SUITES}"
echo "试验次数: ${NUM_TRIALS}"
echo "生成指令数: ${NUM_INSTRUCTIONS}"
echo "Top-K选择: ${SELECT_TOPK}"
echo "语义类型: ${SEMANTIC_TYPE}"
echo "保存视频: ${SAVE_VIDEOS}"
echo "使用WandB: ${USE_WANDB}"
echo "基础输出目录: ${BASE_OUTPUT_DIR}"
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
# 准备基础输出目录
# ==========================================
mkdir -p "${BASE_OUTPUT_DIR}"

# 保存全局配置
cat > "${BASE_OUTPUT_DIR}/global_config.json" <<EOF
{
    "backend": "${BACKEND}",
    "qwen_mode": "${QWEN_MODE}",
    "qwen_model_id": "${QWEN_MODEL_ID}",
    "host": "${HOST}",
    "port": ${PORT},
    "task_suites": "${TASK_SUITES}",
    "num_trials": ${NUM_TRIALS},
    "num_instructions": ${NUM_INSTRUCTIONS},
    "select_topk": ${SELECT_TOPK},
    "semantic_type": "${SEMANTIC_TYPE}",
    "save_videos": ${SAVE_VIDEOS},
    "use_wandb": ${USE_WANDB},
    "timestamp": "${TIMESTAMP}"
}
EOF

echo "全局配置已保存到: ${BASE_OUTPUT_DIR}/global_config.json"
echo ""

# ==========================================
# 循环运行所有 task suite
# ==========================================
TOTAL_SUITES=0
SUCCESSFUL_SUITES=0
FAILED_SUITES=()

for TASK_SUITE in ${TASK_SUITES}; do
    TOTAL_SUITES=$((TOTAL_SUITES + 1))
    
    echo ""
    echo "=========================================="
    echo "运行 Task Suite ${TOTAL_SUITES}: ${TASK_SUITE}"
    echo "=========================================="
    echo ""
    
    # 为每个 suite 创建单独的输出目录
    OUTPUT_DIR="${BASE_OUTPUT_DIR}/${TASK_SUITE}"
    mkdir -p "${OUTPUT_DIR}/videos"
    mkdir -p "${OUTPUT_DIR}/results"
    
    # 构建命令参数
    CMD_ARGS=(
        --mode vlm
        --BACKEND "${BACKEND}"
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
    
    # 保存当前 suite 的配置
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
    echo "开始评估 ${TASK_SUITE}..."
    if python openpi/examples/libero/unified_eval.py "${CMD_ARGS[@]}" 2>&1 | tee "${OUTPUT_DIR}/eval.log"; then
        EXIT_CODE=0
        SUCCESSFUL_SUITES=$((SUCCESSFUL_SUITES + 1))
        echo ""
        echo "✓ ${TASK_SUITE} 评估完成"
    else
        EXIT_CODE=$?
        FAILED_SUITES+=("${TASK_SUITE}")
        echo ""
        echo "✗ ${TASK_SUITE} 评估失败 (退出码: ${EXIT_CODE})"
    fi
    
    # 显示当前 suite 的成功率
    if [[ -f "${OUTPUT_DIR}/results/whole_acc_log.json" ]]; then
        echo ""
        echo "${TASK_SUITE} 成功率统计:"
        python -c "
import json
try:
    with open('${OUTPUT_DIR}/results/whole_acc_log.json', 'r') as f:
        content = f.read()
        # 尝试解析多个JSON对象
        import re
        json_objects = re.findall(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', content)
        if json_objects:
            data = json.loads(json_objects[-1])  # 取最后一个JSON对象
        else:
            data = json.loads(content)
    
    if isinstance(data, list) and len(data) > 0:
        data = data[-1]
    
    print(f\"  模式: {data.get('mode', 'N/A')}\")
    for key, value in data.items():
        if key not in ['mode', 'total_successes', 'total_episodes', 'mean_similarity', 'variance_similarity']:
            if isinstance(value, (int, float)):
                print(f\"  {key} 成功率: {value:.2%}\")
        elif key == 'total_episodes':
            print(f\"  总试验数: {value}\")
        elif key == 'total_successes':
            print(f\"  成功次数: {value}\")
        elif key == 'mean_similarity':
            print(f\"  平均相似度: {value:.4f}\")
except Exception as e:
    print(f\"  无法读取结果: {e}\")
" || echo "  无法解析结果文件"
    fi
    
    echo ""
    echo "=========================================="
    
    # 短暂休息，避免资源冲突
    sleep 2
done

# ==========================================
# 最终汇总
# ==========================================
echo ""
echo "=========================================="
echo "所有 Task Suite 评估完成"
echo "=========================================="
echo ""
echo "总计: ${TOTAL_SUITES} 个 task suite"
echo "成功: ${SUCCESSFUL_SUITES} 个"
echo "失败: $((TOTAL_SUITES - SUCCESSFUL_SUITES)) 个"

if [[ ${#FAILED_SUITES[@]} -gt 0 ]]; then
    echo ""
    echo "失败的 task suite:"
    for suite in "${FAILED_SUITES[@]}"; do
        echo "  - ${suite}"
    done
fi

echo ""
echo "所有结果保存在: ${BASE_OUTPUT_DIR}"
echo ""

# 生成汇总报告
SUMMARY_FILE="${BASE_OUTPUT_DIR}/summary_report.txt"
{
    echo "Qwen2-VL-2B-Instruct 评估汇总报告"
    echo "=================================="
    echo ""
    echo "时间戳: ${TIMESTAMP}"
    echo "模型: ${QWEN_MODEL_ID}"
    echo "后端: ${BACKEND}"
    echo "模式: ${QWEN_MODE}"
    echo ""
    echo "评估结果:"
    echo "--------"
    
    for TASK_SUITE in ${TASK_SUITES}; do
        echo ""
        echo "Task Suite: ${TASK_SUITE}"
        
        LOG_FILE="${BASE_OUTPUT_DIR}/${TASK_SUITE}/results/whole_acc_log.json"
        if [[ -f "${LOG_FILE}" ]]; then
            python -c "
import json
try:
    with open('${LOG_FILE}', 'r') as f:
        content = f.read()
        import re
        json_objects = re.findall(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', content)
        if json_objects:
            data = json.loads(json_objects[-1])
        else:
            data = json.loads(content)
    
    if isinstance(data, list) and len(data) > 0:
        data = data[-1]
    
    for key, value in data.items():
        if key not in ['mode', 'total_successes', 'total_episodes', 'mean_similarity', 'variance_similarity']:
            if isinstance(value, (int, float)):
                print(f\"  成功率: {value:.2%}\")
        elif key == 'total_episodes':
            print(f\"  总试验数: {value}\")
        elif key == 'total_successes':
            print(f\"  成功次数: {value}\")
except Exception as e:
    print(f\"  错误: {e}\")
" 2>/dev/null || echo "  无法读取结果"
        else
            echo "  状态: 失败或未完成"
        fi
    done
} > "${SUMMARY_FILE}"

echo "汇总报告已保存到: ${SUMMARY_FILE}"
cat "${SUMMARY_FILE}"

echo ""
echo "=========================================="

# 如果有失败的 suite，返回非零退出码
if [[ ${#FAILED_SUITES[@]} -gt 0 ]]; then
    exit 1
else
    exit 0
fi
