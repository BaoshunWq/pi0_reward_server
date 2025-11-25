#!/usr/bin/env bash
set -euo pipefail

# 一键启动所有服务器和负载均衡器
# 
# 环境变量配置:
#   GPUS="0,1,2,3"              - 使用的GPU列表
#   BASE_POLICY_PORT=8000       - Policy服务器起始端口
#   BASE_REWARD_PORT=6001       - Reward服务器起始端口
#   LB_PORT=6000                - 负载均衡器端口
#   POLICY_PY=/path/to/python   - Policy服务器Python路径
#   REWARD_PY=/path/to/python   - Reward服务器Python路径
#   OPENPI_ENV=PI0_LIBERO       - OpenPI环境名称

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="${SCRIPT_DIR}/.."
cd "$PROJ_ROOT"

# 默认配置
export GPUS="${GPUS:-4,5}"
export BASE_POLICY_PORT="${BASE_POLICY_PORT:-8000}"
export BASE_REWARD_PORT="${BASE_REWARD_PORT:-6001}"
export LB_PORT="${LB_PORT:-6000}"
export POLICY_PY="${POLICY_PY:-/root/autodl-tmp/conda/envs/openpi/bin/python}"
export REWARD_PY="${REWARD_PY:-/root/autodl-tmp/conda/envs/openpi-libero/bin/python}"
export OPENPI_ENV="${OPENPI_ENV:-PI0_LIBERO}"

# 验证Python
if [[ ! -x "${POLICY_PY}" ]]; then
    echo "[fatal] POLICY_PY not executable: ${POLICY_PY}" >&2
    exit 2
fi

if [[ ! -x "${REWARD_PY}" ]]; then
    echo "[fatal] REWARD_PY not executable: ${REWARD_PY}" >&2
    exit 2
fi

# 解析GPU列表
IFS=',' read -ra GPU_ARRAY <<< "$GPUS"
NUM_GPUS=${#GPU_ARRAY[@]}
export NUM_SERVERS="${NUM_GPUS}"

echo "=========================================="
echo "启动完整并行服务架构"
echo "=========================================="
echo "  GPUs: ${GPUS} (${NUM_GPUS}个)"
echo "  Policy端口: ${BASE_POLICY_PORT}-$((BASE_POLICY_PORT + NUM_GPUS - 1))"
echo "  Reward端口: ${BASE_REWARD_PORT}-$((BASE_REWARD_PORT + NUM_GPUS - 1))"
echo "  负载均衡器: ${LB_PORT}"
echo "  Policy Python: ${POLICY_PY}"
echo "  Reward Python: ${REWARD_PY}"
echo "=========================================="

POLICY_PIDS=()
REWARD_PIDS=()
LB_PID=""

# 清理函数
cleanup() {
    echo ""
    echo "=========================================="
    echo "停止所有服务..."
    echo "=========================================="
    
    if [[ -n "${LB_PID}" ]] && kill -0 "${LB_PID}" 2>/dev/null; then
        echo "停止负载均衡器 (PID: ${LB_PID})..."
        kill "${LB_PID}" 2>/dev/null || true
    fi
    
    if [[ ${#REWARD_PIDS[@]} -gt 0 ]]; then
        echo "停止Reward服务器 (PIDs: ${REWARD_PIDS[*]})..."
        kill "${REWARD_PIDS[@]}" 2>/dev/null || true
    fi
    
    if [[ ${#POLICY_PIDS[@]} -gt 0 ]]; then
        echo "停止Policy服务器 (PIDs: ${POLICY_PIDS[*]})..."
        kill "${POLICY_PIDS[@]}" 2>/dev/null || true
    fi
    
    wait 2>/dev/null || true
    echo "所有服务已停止"
    exit 0
}

# 注册清理函数
trap cleanup INT TERM EXIT

# ==========================================
# 第一步：启动所有服务器
# ==========================================
echo ""
echo "第一步：启动所有GPU服务器..."
echo "=========================================="

for i in "${!GPU_ARRAY[@]}"; do
    GPU_ID="${GPU_ARRAY[$i]}"
    POLICY_PORT=$((BASE_POLICY_PORT + i))
    REWARD_PORT=$((BASE_REWARD_PORT + i))
    
    echo ""
    echo "[GPU ${GPU_ID}] 启动服务器..."
    echo "  Policy端口: ${POLICY_PORT}"
    echo "  Reward端口: ${REWARD_PORT}"
    
    # 启动Policy服务器
    CUDA_VISIBLE_DEVICES="${GPU_ID}" \
    "${POLICY_PY}" openpi/scripts/serve_policy.py \
        --env "${OPENPI_ENV}" \
        --port "${POLICY_PORT}" \
        > "server_policy_gpu${GPU_ID}_port${POLICY_PORT}.log" 2>&1 &
    
    POLICY_PID=$!
    POLICY_PIDS+=($POLICY_PID)
    echo "  Policy PID: ${POLICY_PID}"
    
    # 等待policy服务器启动
    sleep 2
    
    # 启动Reward服务器（单GPU，禁用GPU池）
    export PYTHONPATH="${PROJ_ROOT}:${PYTHONPATH:-}"
    
    CUDA_VISIBLE_DEVICES="${GPU_ID}" \
    PORT="${REWARD_PORT}" \
    USE_GPU_POOL=0 \
    POLICY_PORT="${POLICY_PORT}" \
    "${REWARD_PY}" -m waitress \
        --host=0.0.0.0 \
        --port="${REWARD_PORT}" \
        --threads=4 \
        --channel-timeout=600 \
        --call \
        pi0_reward_server.app_pi0_libero:create_app \
        > "server_reward_gpu${GPU_ID}_port${REWARD_PORT}.log" 2>&1 &
    
    REWARD_PID=$!
    REWARD_PIDS+=($REWARD_PID)
    echo "  Reward PID: ${REWARD_PID}"
done

echo ""
echo "所有服务器已启动，等待5秒确保服务就绪..."
sleep 5

# ==========================================
# 第二步：启动负载均衡器
# ==========================================
echo ""
echo "第二步：启动负载均衡器..."
echo "=========================================="
echo "  监听端口: ${LB_PORT}"
echo "  后端服务器: ${BASE_REWARD_PORT} - $((BASE_REWARD_PORT + NUM_SERVERS - 1))"
echo "  策略: 最少连接数 (Least Connections)"
echo "=========================================="

python scripts/parallel/load_balancer.py \
    --listen_port "${LB_PORT}" \
    --base_port "${BASE_REWARD_PORT}" \
    --num_servers "${NUM_SERVERS}" \
    > "load_balancer_port${LB_PORT}.log" 2>&1 &

LB_PID=$!
echo "负载均衡器 PID: ${LB_PID}"

echo ""
echo "=========================================="
echo "✓ 所有服务已成功启动"
echo "=========================================="
echo ""
echo "服务信息:"
echo "  负载均衡器: http://localhost:${LB_PORT}"
echo "  健康检查: curl http://localhost:${LB_PORT}/health"
echo ""
echo "后端Reward服务器:"
for i in "${!GPU_ARRAY[@]}"; do
    REWARD_PORT=$((BASE_REWARD_PORT + i))
    echo "  GPU ${GPU_ARRAY[$i]}: http://localhost:${REWARD_PORT}"
done
echo ""
echo "Policy服务器:"
for i in "${!GPU_ARRAY[@]}"; do
    POLICY_PORT=$((BASE_POLICY_PORT + i))
    echo "  GPU ${GPU_ARRAY[$i]}: http://localhost:${POLICY_PORT}"
done
echo ""
echo "日志文件:"
echo "  负载均衡器: load_balancer_port${LB_PORT}.log"
for i in "${!GPU_ARRAY[@]}"; do
    GPU_ID="${GPU_ARRAY[$i]}"
    POLICY_PORT=$((BASE_POLICY_PORT + i))
    REWARD_PORT=$((BASE_REWARD_PORT + i))
    echo "  GPU ${GPU_ID} Policy: server_policy_gpu${GPU_ID}_port${POLICY_PORT}.log"
    echo "  GPU ${GPU_ID} Reward: server_reward_gpu${GPU_ID}_port${REWARD_PORT}.log"
done
echo ""
echo "按 Ctrl+C 停止所有服务"
echo "=========================================="

# 等待所有进程
wait
