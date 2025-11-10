#!/bin/bash
# 测试our_trained_LLM_test.py脚本
# 使用方法: ./test_trained_llm.sh [verl_model_path]

set -e

# 颜色输出
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Trained LLM Test Script${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# 获取verl模型路径参数
VERL_MODEL_PATH=${1:-"/root/autodl-tmp/trained_models/rover-coundown/countdown-rover/actor/global_step_200"}

# 配置参数
HOST=${HOST:-"0.0.0.0"}
PORT=${PORT:-8000}
TASK_SUITE=${TASK_SUITE:-"libero_spatial"}
NUM_TRIALS=${NUM_TRIALS:-2}  # 测试用，只跑2次
NUM_INSTRUCTIONS=${NUM_INSTRUCTIONS:-5}  # 测试用，生成5条指令
SELECT_TOPK=${SELECT_TOPK:-3}  # 测试用，选择top 3
SAVE_VIDEOS=${SAVE_VIDEOS:-false}  # 默认不保存视频以节省空间
BACKEND=${BACKEND:-"verl_qwen"}
SEMANTIC_TYPE=${SEMANTIC_TYPE:-"clip"}
WANDB_PROJECT=${WANDB_PROJECT:-"test_trained_llm"}
WANDB_ENTITY=${WANDB_ENTITY:-""}

echo -e "${YELLOW}测试配置:${NC}"
echo "  - VERL Model Path: $VERL_MODEL_PATH"
echo "  - Pi0 Server: $HOST:$PORT"
echo "  - Task Suite: $TASK_SUITE"
echo "  - Num Trials per Task: $NUM_TRIALS"
echo "  - Num Instructions: $NUM_INSTRUCTIONS"
echo "  - Select Top K: $SELECT_TOPK"
echo "  - Save Videos: $SAVE_VIDEOS"
echo "  - Backend: $BACKEND"
echo "  - Semantic Type: $SEMANTIC_TYPE"
echo "  - WandB Project: $WANDB_PROJECT"
if [ -n "$WANDB_ENTITY" ]; then
    echo "  - WandB Entity: $WANDB_ENTITY"
fi
echo ""

# 检查模型路径
if [ ! -d "$VERL_MODEL_PATH" ]; then
    echo -e "${RED}错误: VERL模型路径不存在: $VERL_MODEL_PATH${NC}"
    echo -e "${YELLOW}请提供正确的模型路径作为第一个参数${NC}"
    echo "使用方法: $0 /path/to/verl/model"
    exit 1
fi

echo -e "${GREEN}✓ VERL模型路径存在${NC}"
echo ""

# 检查Pi0服务器
echo -e "${YELLOW}检查Pi0服务器 $HOST:$PORT ...${NC}"
if timeout 5 bash -c "echo > /dev/tcp/$HOST/$PORT" 2>/dev/null; then
    echo -e "${GREEN}✓ Pi0服务器运行正常${NC}"
else
    echo -e "${RED}✗ 警告: 无法连接到Pi0服务器 $HOST:$PORT${NC}"
    echo -e "${YELLOW}请确保Pi0服务已启动${NC}"
    read -p "是否继续？[y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi
echo ""

# 检查GPU
echo -e "${YELLOW}检查GPU状态...${NC}"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader
    echo -e "${GREEN}✓ GPU可用${NC}"
else
    echo -e "${YELLOW}○ 未检测到GPU${NC}"
fi
echo ""

# 检查WandB登录状态
echo -e "${YELLOW}检查WandB状态...${NC}"
if command -v wandb &> /dev/null; then
    if wandb verify 2>&1 | grep -q "logged in"; then
        echo -e "${GREEN}✓ WandB已登录${NC}"
    else
        echo -e "${YELLOW}○ WandB未登录，将使用匿名模式${NC}"
        echo -e "${YELLOW}  提示: 运行 'wandb login' 登录WandB${NC}"
    fi
else
    echo -e "${RED}✗ 未安装wandb${NC}"
    echo -e "${YELLOW}  请运行: pip install wandb${NC}"
    exit 1
fi
echo ""

# 创建输出目录
TIMESTAMP=$(date +%Y-%m-%d_%H:%M:%S)
OUTPUT_DIR="data/test_trained_llm_${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR/results"
if [ "$SAVE_VIDEOS" = "true" ]; then
    mkdir -p "$OUTPUT_DIR/videos"
fi

echo -e "${GREEN}输出目录: $OUTPUT_DIR${NC}"
echo ""

# 构建命令
CMD="python openpi/examples/libero/our_trained_LLM_test.py \
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
    --output-path $OUTPUT_DIR/results/trained_llm_results.json \
    --whole-acc-log-path $OUTPUT_DIR/results/whole_acc_log.json \
    --wandb-project $WANDB_PROJECT"

if [ -n "$WANDB_ENTITY" ]; then
    CMD="$CMD --wandb-entity $WANDB_ENTITY"
fi

if [ "$SAVE_VIDEOS" = "true" ]; then
    CMD="$CMD --video-out-path $OUTPUT_DIR/videos"
fi

echo -e "${YELLOW}执行命令:${NC}"
echo "$CMD"
echo ""

read -p "开始测试？[Y/n] " -n 1 -r
echo
if [[ $REPLY =~ ^[Nn]$ ]]; then
    echo "取消测试"
    exit 0
fi

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  开始测试...${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# 记录开始时间
START_TIME=$(date +%s)

# 运行测试（将输出同时保存到日志文件）
LOG_FILE="$OUTPUT_DIR/test.log"
$CMD 2>&1 | tee "$LOG_FILE"

# 记录结束时间
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  测试完成！${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "${YELLOW}测试统计:${NC}"
echo "  - 总耗时: ${ELAPSED}秒 ($((ELAPSED/60))分钟)"
echo "  - 输出目录: $OUTPUT_DIR"
echo "  - 日志文件: $LOG_FILE"
echo ""

# 检查结果文件
if [ -f "$OUTPUT_DIR/results/trained_llm_results.json" ]; then
    echo -e "${GREEN}✓ 结果文件生成成功${NC}"
    echo "  - 结果: $OUTPUT_DIR/results/trained_llm_results.json"
    
    # 显示部分结果
    echo ""
    echo -e "${YELLOW}结果摘要:${NC}"
    python3 << EOF
import json
try:
    with open('$OUTPUT_DIR/results/trained_llm_results.json', 'r') as f:
        results = json.load(f)
    print(f"  - 总任务数: {len(results)}")
    for i, result in enumerate(results[:3]):  # 只显示前3个
        print(f"  - 任务{i}: {result.get('task', 'N/A')[:50]}...")
        print(f"    成功指令: {len(result.get('done_anotations', []))}")
        print(f"    失败指令: {len(result.get('fail_anotations', []))}")
except Exception as e:
    print(f"  无法解析结果: {e}")
EOF
else
    echo -e "${RED}✗ 结果文件未生成${NC}"
fi

if [ -f "$OUTPUT_DIR/results/whole_acc_log.json" ]; then
    echo ""
    echo -e "${GREEN}✓ 准确率日志生成成功${NC}"
    cat "$OUTPUT_DIR/results/whole_acc_log.json" | python3 -m json.tool
else
    echo -e "${YELLOW}○ 准确率日志未生成${NC}"
fi

echo ""
echo -e "${YELLOW}WandB链接:${NC}"
echo "  - 查看WandB Dashboard了解详细指标"
echo "  - 项目: $WANDB_PROJECT"
echo ""

echo -e "${YELLOW}下一步:${NC}"
echo "  - 查看详细结果: cat $OUTPUT_DIR/results/trained_llm_results.json"
echo "  - 查看日志: cat $LOG_FILE"
if [ "$SAVE_VIDEOS" = "true" ]; then
    echo "  - 查看视频: ls $OUTPUT_DIR/videos/"
fi
echo "  - 查看WandB: 访问 https://wandb.ai/ 查看实验结果"
echo ""

