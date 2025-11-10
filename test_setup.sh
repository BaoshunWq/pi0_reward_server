#!/bin/bash
# 测试并行版本的Pi0 Reward Server环境配置
# 使用方法: ./test_setup.sh

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Pi0 Reward Server 环境检查${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# 检查Python版本
echo -e "${YELLOW}检查Python版本...${NC}"
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version)
    echo -e "${GREEN}✓ $PYTHON_VERSION${NC}"
else
    echo -e "${RED}✗ 未找到Python3${NC}"
    exit 1
fi
echo ""

# 检查Python包
echo -e "${YELLOW}检查Python依赖...${NC}"
REQUIRED_PACKAGES=("flask" "numpy" "imageio" "tqdm")
for pkg in "${REQUIRED_PACKAGES[@]}"; do
    if python3 -c "import $pkg" 2>/dev/null; then
        echo -e "${GREEN}✓ $pkg${NC}"
    else
        echo -e "${RED}✗ $pkg 未安装${NC}"
        echo -e "${YELLOW}  请运行: pip install $pkg${NC}"
    fi
done
echo ""

# 检查可选包
echo -e "${YELLOW}检查可选依赖...${NC}"
if python3 -c "import gunicorn" 2>/dev/null; then
    echo -e "${GREEN}✓ gunicorn (生产环境推荐)${NC}"
else
    echo -e "${YELLOW}○ gunicorn 未安装 (可选)${NC}"
    echo -e "${YELLOW}  生产环境建议安装: pip install gunicorn${NC}"
fi
echo ""

# 检查GPU
echo -e "${YELLOW}检查GPU...${NC}"
if command -v nvidia-smi &> /dev/null; then
    GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    echo -e "${GREEN}✓ 检测到 $GPU_COUNT 张GPU${NC}"
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | while IFS=, read -r idx name memory; do
        echo "  GPU $idx: $name ($memory)"
    done
else
    echo -e "${RED}✗ 未检测到GPU或nvidia-smi不可用${NC}"
fi
echo ""

# 检查端口占用
echo -e "${YELLOW}检查端口占用...${NC}"
PORTS=(6006 8000 8001)
for port in "${PORTS[@]}"; do
    if netstat -tuln 2>/dev/null | grep -q ":$port "; then
        echo -e "${YELLOW}○ 端口 $port 已被占用${NC}"
    else
        echo -e "${GREEN}✓ 端口 $port 可用${NC}"
    fi
done
echo ""

# 检查文件
echo -e "${YELLOW}检查必需文件...${NC}"
REQUIRED_FILES=(
    "pi0_reward_server/app_pi0_libero_parallel.py"
    "pi0_reward_server/reward_core_parallel.py"
    "pi0_reward_server/env_pertask_parallel.py"
    "pi0_reward_server/client_pool.py"
    "gunicorn_config.py"
    "start_server.sh"
    "start_pi0_models.sh"
)

for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo -e "${GREEN}✓ $file${NC}"
    else
        echo -e "${RED}✗ $file 不存在${NC}"
    fi
done
echo ""

# 检查脚本权限
echo -e "${YELLOW}检查脚本权限...${NC}"
SCRIPTS=("start_server.sh" "start_pi0_models.sh" "stop_pi0_models.sh" "example_client.py")
for script in "${SCRIPTS[@]}"; do
    if [ -f "$script" ]; then
        if [ -x "$script" ]; then
            echo -e "${GREEN}✓ $script 可执行${NC}"
        else
            echo -e "${YELLOW}○ $script 不可执行${NC}"
            echo -e "${YELLOW}  运行: chmod +x $script${NC}"
        fi
    fi
done
echo ""

# 环境变量检查
echo -e "${YELLOW}检查环境变量...${NC}"
if [ -z "$GPU_PORTS" ]; then
    echo -e "${YELLOW}○ GPU_PORTS 未设置 (将使用默认值: 8000,8001)${NC}"
else
    echo -e "${GREEN}✓ GPU_PORTS=$GPU_PORTS${NC}"
fi

if [ -z "$PORT" ]; then
    echo -e "${YELLOW}○ PORT 未设置 (将使用默认值: 6006)${NC}"
else
    echo -e "${GREEN}✓ PORT=$PORT${NC}"
fi
echo ""

# 测试本地连接
echo -e "${YELLOW}测试本地网络...${NC}"
if timeout 2 bash -c "echo > /dev/tcp/127.0.0.1/22" 2>/dev/null; then
    echo -e "${GREEN}✓ 本地网络正常${NC}"
else
    echo -e "${YELLOW}○ 无法测试本地网络${NC}"
fi
echo ""

# 总结
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  环境检查完成${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "${YELLOW}下一步:${NC}"
echo "  1. 启动Pi0模型服务: ./start_pi0_models.sh"
echo "  2. 启动Reward Server: ./start_server.sh"
echo "  3. 测试服务: python example_client.py"
echo ""
echo -e "${YELLOW}文档:${NC}"
echo "  - 快速开始: cat QUICKSTART.md"
echo "  - 详细文档: cat README_PARALLEL.md"
echo ""

