#!/usr/bin/env bash
# 测试多服务器模式

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

# 配置（与unify_multi_servers.sh保持一致）
LB_PORT="${LB_PORT:-6000}"
BASE_REWARD_PORT="${BASE_REWARD_PORT:-6001}"
BASE_POLICY_PORT="${BASE_POLICY_PORT:-8000}"
NUM_SERVERS="${NUM_SERVERS:-2}"

echo "=========================================="
echo "多服务器模式测试"
echo "=========================================="
echo "负载均衡器: http://localhost:${LB_PORT}"
echo "Reward服务器: ${BASE_REWARD_PORT}-$((BASE_REWARD_PORT + NUM_SERVERS - 1))"
echo "Policy服务器: ${BASE_POLICY_PORT}-$((BASE_POLICY_PORT + NUM_SERVERS - 1))"
echo "=========================================="

# 激活环境
if command -v conda &> /dev/null; then
    echo "激活 openpi-libero_peft 环境..."
    eval "$(conda shell.bash hook)"
    conda activate openpi-libero_peft || conda activate openpi-libero || true
fi

echo ""
echo "[1/4] 测试各个Reward服务器..."
ALL_OK=true
for i in $(seq 0 $((NUM_SERVERS - 1))); do
    PORT=$((BASE_REWARD_PORT + i))
    echo -n "  Reward服务器 $i (port ${PORT}): "
    if curl -s http://localhost:${PORT}/health > /dev/null 2>&1; then
        echo "✅ 正常"
    else
        echo "❌ 无响应"
        ALL_OK=false
    fi
done

echo ""
echo "[2/4] 测试负载均衡器..."
echo -n "  负载均衡器 (port ${LB_PORT}): "
if curl -s http://localhost:${LB_PORT}/health > /dev/null 2>&1; then
    echo "✅ 正常"
else
    echo "❌ 无响应"
    echo ""
    echo "提示: 请先启动负载均衡器"
    echo "  NUM_SERVERS=${NUM_SERVERS} bash scripts/parallel/launch_load_balancer.sh"
    ALL_OK=false
fi

if [ "$ALL_OK" = false ]; then
    echo ""
    echo "❌ 健康检查失败"
    exit 1
fi

echo ""
echo "[3/4] 测试单个请求..."
python scripts/parallel/test_dual_env.py \
    --reward_url "http://localhost:${LB_PORT}/score" \
    --policy_base_port "${BASE_POLICY_PORT}" \
    --num_gpus "${NUM_SERVERS}" \
    --num_samples 1 \
    --skip_health

echo ""
echo "[4/4] 测试多个请求（验证负载均衡）..."
python scripts/parallel/test_dual_env.py \
    --reward_url "http://localhost:${LB_PORT}/score" \
    --policy_base_port "${BASE_POLICY_PORT}" \
    --num_gpus "${NUM_SERVERS}" \
    --num_samples 5 \
    --skip_health

echo ""
echo "=========================================="
echo "✅ 多服务器模式测试完成！"
echo "=========================================="
