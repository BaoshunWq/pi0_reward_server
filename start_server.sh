#!/bin/bash
# 启动并行版本的Pi0 Reward Server
# 使用方法: ./start_server.sh

set -e  # 遇到错误立即退出

# 颜色输出
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Pi0 Reward Server - 并行版本启动器${NC}"
echo -e "${GREEN}========================================${NC}"

# 配置参数
export PORT=${PORT:-6006}
export GPU_HOST=${GPU_HOST:-0.0.0.0}
export GPU_PORTS=${GPU_PORTS:-"8000,8001"}  # 两张显卡的端口，逗号分隔
export GUNICORN_WORKERS=${GUNICORN_WORKERS:-2}  # Gunicorn worker数量
export GUNICORN_TIMEOUT=${GUNICORN_TIMEOUT:-1800}  # 30分钟超时
export LOG_LEVEL=${LOG_LEVEL:-info}

echo -e "${YELLOW}配置信息:${NC}"
echo "  - Server Port: $PORT"
echo "  - GPU Host: $GPU_HOST"
echo "  - GPU Ports: $GPU_PORTS"
echo "  - Gunicorn Workers: $GUNICORN_WORKERS"
echo "  - Timeout: $GUNICORN_TIMEOUT seconds"
echo ""

# 检查Pi0模型服务是否运行
IFS=',' read -ra PORTS <<< "$GPU_PORTS"
for port in "${PORTS[@]}"; do
    port=$(echo $port | xargs)  # 去除空格
    echo -e "${YELLOW}检查GPU模型服务 $GPU_HOST:$port ...${NC}"
    if timeout 5 bash -c "echo > /dev/tcp/$GPU_HOST/$port" 2>/dev/null; then
        echo -e "${GREEN}✓ GPU服务 $GPU_HOST:$port 正在运行${NC}"
    else
        echo -e "${RED}✗ 警告: GPU服务 $GPU_HOST:$port 无法连接！${NC}"
        echo -e "${RED}  请先启动Pi0模型服务在端口 $port${NC}"
    fi
done
echo ""

# 进入项目目录
cd "$(dirname "$0")/pi0_reward_server"

echo -e "${YELLOW}启动方式选择:${NC}"
echo "  1. 开发模式 (Flask内置服务器, 单进程)"
echo "  2. 生产模式 (Gunicorn, 多进程)"
echo ""
read -p "请选择 [1/2] (默认: 2): " MODE
MODE=${MODE:-2}

if [ "$MODE" = "1" ]; then
    echo -e "${GREEN}使用开发模式启动...${NC}"
    python app_pi0_libero_parallel.py
elif [ "$MODE" = "2" ]; then
    echo -e "${GREEN}使用生产模式启动 (Gunicorn)...${NC}"
    
    # 检查是否安装了gunicorn
    if ! command -v gunicorn &> /dev/null; then
        echo -e "${RED}错误: 未安装 gunicorn${NC}"
        echo -e "${YELLOW}请运行: pip install gunicorn${NC}"
        exit 1
    fi
    
    # 启动Gunicorn
    gunicorn \
        --config ../gunicorn_config.py \
        --bind "0.0.0.0:$PORT" \
        --workers "$GUNICORN_WORKERS" \
        --timeout "$GUNICORN_TIMEOUT" \
        --log-level "$LOG_LEVEL" \
        "app_pi0_libero_parallel:create_app()"
else
    echo -e "${RED}无效选择，退出${NC}"
    exit 1
fi

