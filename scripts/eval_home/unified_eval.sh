#!/bin/bash
# 统一评估测试脚本
# 使用方法: ./test_unified_eval.sh [llm|vlm] [verl_model_path]

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Unified Evaluation"
echo -e "${GREEN}========================================${NC}"
echo ""

# 获取模式参数
MODE=${1:-"vlm"}
# VERL_MODEL_PATH=${2:-"/root/autodl-tmp/trained_models/Qwen2.5-1.5B-Instruct_full_train_step200"}
VERL_MODEL_PATH=${2:-"/root/autodl-tmp/code/saftyVLA/rover_verl/checkpoints/custom_rover_qwen2_5_vl_lora/global_step_180"}

# 验证模式
if [[ "$MODE" != "llm" && "$MODE" != "vlm" ]]; then
    echo -e "${RED}错误: 无效的模式 '$MODE'${NC}"
    echo "用法: $0 [llm|vlm] [verl_model_path]"
    exit 1
fi

# 配置参数
HOST=${HOST:-"0.0.0.0"}
PORT=${PORT:-8000}
TASK_SUITE=${TASK_SUITE:-"libero_spatial"}
NUM_TRIALS=${NUM_TRIALS:-50}
NUM_INSTRUCTIONS=${NUM_INSTRUCTIONS:-1}
SELECT_TOPK=${SELECT_TOPK:-1}
BACKEND=${BACKEND:-"verl_qwen"}
SEMANTIC_TYPE=${SEMANTIC_TYPE:-"clip"}
WANDB_PROJECT=${WANDB_PROJECT:-"unified_eval_test"}
WANDB_ENTITY=${WANDB_ENTITY:-"tongbs-sysu"}

echo -e "${YELLOW}测试配置:${NC}"
echo "  - Mode: ${MODE^^}"
echo "  - VERL Model Path: $VERL_MODEL_PATH"
echo "  - Pi0 Server: $HOST:$PORT"
echo "  - Task Suite: $TASK_SUITE"
echo "  - Num Trials per Task: $NUM_TRIALS"
echo "  - Num Instructions: $NUM_INSTRUCTIONS"
echo "  - Select Top K: $SELECT_TOPK"
echo "  - Save Videos: $SAVE_VIDEOS"
echo "  - Backend: $BACKEND"
echo "  - Semantic Type: $SEMANTIC_TYPE"
echo "  - Use WandB: $USE_WANDB"
if [ "$USE_WANDB" = "true" ]; then
    echo "  - WandB Project: $WANDB_PROJECT"
    if [ -n "$WANDB_ENTITY" ]; then
        echo "  - WandB Entity: $WANDB_ENTITY"
    fi
fi

# 构建命令
python openpi/examples/libero/unified_eval.py \
    --mode $MODE \
    --host $HOST \
    --port $PORT \
    --task-suite-name $TASK_SUITE \
    --num-trials-per-task $NUM_TRIALS \
    --num-instructions $NUM_INSTRUCTIONS \
    --select-topk $SELECT_TOPK \
    --save-videos $SAVE_VIDEOS \
    --BACKEND $BACKEND \
    --semantic-type $SEMANTIC_TYPE \
    --verl-model-path $VERL_MODEL_PATH \
    --output-path $OUTPUT_DIR/results/unified_results.json \
    --whole-acc-log-path $OUTPUT_DIR/results/whole_acc_log.json \
    --no-use-wandb
    # --use-wandb $USE_WANDB

