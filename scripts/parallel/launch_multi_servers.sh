#!/usr/bin/env bash
set -euo pipefail

# 多服务器模式：每个GPU运行独立的policy和reward服务器
# 这样可以完全隔离EGL上下文，避免冲突
#
# 架构：
#   GPU 0: Policy(8000) + Reward(6001)
#   GPU 1: Policy(8001) + Reward(6002)
#   GPU 2: Policy(8002) + Reward(6003)
#   ...
#   负载均衡器: port 6000 → 分发到各个reward服务器
#
# Env overrides:
#   GPUS="0,1,2,3"
#   BASE_POLICY_PORT=8000
#   BASE_REWARD_PORT=6001
#   LB_PORT=6000
#   POLICY_PY=/path/to/openpi/bin/python
#   REWARD_PY=/path/to/openpi-libero/bin/python

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="${SCRIPT_DIR}/../.."
cd "$PROJ_ROOT"

export GPUS="${GPUS:-0,1}"
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

echo "=========================================="
echo "多服务器模式启动"
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

# 为每个GPU启动policy和reward服务器
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
    # 设置PYTHONPATH确保可以导入模块
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
echo "=========================================="
echo "所有服务器已启动"
echo "=========================================="
echo "Policy PIDs: ${POLICY_PIDS[*]}"
echo "Reward PIDs: ${REWARD_PIDS[*]}"
echo ""
echo "Reward服务器列表:"
for i in "${!GPU_ARRAY[@]}"; do
    REWARD_PORT=$((BASE_REWARD_PORT + i))
    echo "  GPU ${GPU_ARRAY[$i]}: http://localhost:${REWARD_PORT}"
done
echo ""
echo "下一步:"
echo "  1. 启动负载均衡器: bash scripts/parallel/launch_load_balancer.sh"
echo "  2. 或直接访问某个服务器: curl http://localhost:${BASE_REWARD_PORT}/health"
echo ""
echo "按 Ctrl+C 停止所有服务器"
echo "=========================================="

# Trap信号
trap 'echo ""; echo "停止所有服务器..."; kill ${POLICY_PIDS[@]} ${REWARD_PIDS[@]} 2>/dev/null; wait; echo "所有服务器已停止"; exit' INT TERM

# 等待所有进程
wait
