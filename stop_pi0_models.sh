#!/bin/bash
# 停止Pi0模型服务
# 使用方法: ./stop_pi0_models.sh

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${YELLOW}停止Pi0模型服务...${NC}"

LOG_DIR="logs"

# 停止GPU 0服务
if [ -f "$LOG_DIR/pi0_gpu0.pid" ]; then
    PID=$(cat "$LOG_DIR/pi0_gpu0.pid")
    if ps -p $PID > /dev/null 2>&1; then
        echo "停止GPU 0服务 (PID: $PID)"
        kill $PID
        rm "$LOG_DIR/pi0_gpu0.pid"
        echo -e "${GREEN}✓ GPU 0服务已停止${NC}"
    else
        echo -e "${YELLOW}GPU 0服务未运行${NC}"
        rm "$LOG_DIR/pi0_gpu0.pid"
    fi
else
    echo -e "${YELLOW}未找到GPU 0服务的PID文件${NC}"
fi

# 停止GPU 1服务
if [ -f "$LOG_DIR/pi0_gpu1.pid" ]; then
    PID=$(cat "$LOG_DIR/pi0_gpu1.pid")
    if ps -p $PID > /dev/null 2>&1; then
        echo "停止GPU 1服务 (PID: $PID)"
        kill $PID
        rm "$LOG_DIR/pi0_gpu1.pid"
        echo -e "${GREEN}✓ GPU 1服务已停止${NC}"
    else
        echo -e "${YELLOW}GPU 1服务未运行${NC}"
        rm "$LOG_DIR/pi0_gpu1.pid"
    fi
else
    echo -e "${YELLOW}未找到GPU 1服务的PID文件${NC}"
fi

echo -e "${GREEN}完成${NC}"

