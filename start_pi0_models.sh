#!/bin/bash
# 在两张GPU上分别启动Pi0模型服务
# 使用方法: ./start_pi0_models.sh

set -e

# 颜色输出
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  启动多GPU Pi0模型服务${NC}"
echo -e "${GREEN}========================================${NC}"

# 配置
GPU0_PORT=${GPU0_PORT:-8000}
GPU1_PORT=${GPU1_PORT:-8001}
PI0_MODEL_PATH=${PI0_MODEL_PATH:-"/path/to/pi0_model"}
PI0_SERVER_SCRIPT=${PI0_SERVER_SCRIPT:-"openpi/packages/openpi-server/src/openpi_server/serve.py"}

echo -e "${YELLOW}配置信息:${NC}"
echo "  - GPU 0 端口: $GPU0_PORT"
echo "  - GPU 1 端口: $GPU1_PORT"
echo "  - 模型路径: $PI0_MODEL_PATH"
echo "  - 服务脚本: $PI0_SERVER_SCRIPT"
echo ""

# 检查模型路径
if [ ! -f "$PI0_SERVER_SCRIPT" ]; then
    echo -e "${RED}错误: 找不到Pi0服务脚本: $PI0_SERVER_SCRIPT${NC}"
    echo -e "${YELLOW}请设置环境变量 PI0_SERVER_SCRIPT 指向正确的路径${NC}"
    exit 1
fi

# 创建日志目录
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

# 启动GPU 0上的模型
echo -e "${GREEN}启动GPU 0上的Pi0模型服务 (端口 $GPU0_PORT)...${NC}"
CUDA_VISIBLE_DEVICES=0 python "$PI0_SERVER_SCRIPT" \
    --model_path "$PI0_MODEL_PATH" \
    --port "$GPU0_PORT" \
    > "$LOG_DIR/pi0_gpu0.log" 2>&1 &
GPU0_PID=$!
echo "  - GPU 0 进程PID: $GPU0_PID"
echo "$GPU0_PID" > "$LOG_DIR/pi0_gpu0.pid"

# 等待一下确保第一个服务启动
sleep 5

# 启动GPU 1上的模型
echo -e "${GREEN}启动GPU 1上的Pi0模型服务 (端口 $GPU1_PORT)...${NC}"
CUDA_VISIBLE_DEVICES=1 python "$PI0_SERVER_SCRIPT" \
    --model_path "$PI0_MODEL_PATH" \
    --port "$GPU1_PORT" \
    > "$LOG_DIR/pi0_gpu1.log" 2>&1 &
GPU1_PID=$!
echo "  - GPU 1 进程PID: $GPU1_PID"
echo "$GPU1_PID" > "$LOG_DIR/pi0_gpu1.pid"

echo ""
echo -e "${GREEN}✓ 两个Pi0模型服务已启动${NC}"
echo ""
echo -e "${YELLOW}查看日志:${NC}"
echo "  - GPU 0: tail -f $LOG_DIR/pi0_gpu0.log"
echo "  - GPU 1: tail -f $LOG_DIR/pi0_gpu1.log"
echo ""
echo -e "${YELLOW}停止服务:${NC}"
echo "  - 运行: ./stop_pi0_models.sh"
echo "  - 或手动: kill $GPU0_PID $GPU1_PID"
echo ""

# 等待几秒检查进程是否正常运行
sleep 3
if ps -p $GPU0_PID > /dev/null 2>&1; then
    echo -e "${GREEN}✓ GPU 0 服务运行正常${NC}"
else
    echo -e "${RED}✗ GPU 0 服务启动失败，请检查日志${NC}"
fi

if ps -p $GPU1_PID > /dev/null 2>&1; then
    echo -e "${GREEN}✓ GPU 1 服务运行正常${NC}"
else
    echo -e "${RED}✗ GPU 1 服务启动失败，请检查日志${NC}"
fi

