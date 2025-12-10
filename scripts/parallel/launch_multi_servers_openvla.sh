#!/usr/bin/env bash
set -euo pipefail

# Multi-server launcher for OpenVLA-OFT:
#   - Spawns one policy server + one reward server per GPU
#   - Uses the OpenVLA-OFT websocket policy server and the new reward server
#
# Env overrides:
#   GPUS="0,1"                 GPUs to use
#   BASE_POLICY_PORT=23451     First policy port (one per GPU)
#   BASE_REWARD_PORT=6101      First reward port (one per GPU)
#   LB_PORT=6100               Optional load balancer port (for reference)
#   POLICY_PY=python           Python for policy server
#   REWARD_PY=python           Python for reward server
#   TASK_SUITE=libero_spatial  LIBERO suite name
#   PRETRAINED_CKPT=...        HuggingFace checkpoint id/path
#   NUM_ACTIONS_CHUNK=8        Open-loop steps

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="${SCRIPT_DIR}/../.."
cd "${PROJ_ROOT}"

mkdir -p logs/openvla_parallel

export GPUS="${GPUS:-0}"
export BASE_POLICY_PORT="${BASE_POLICY_PORT:-23451}"
export BASE_REWARD_PORT="${BASE_REWARD_PORT:-6101}"
export LB_PORT="${LB_PORT:-6100}"
export POLICY_PY="${POLICY_PY:-python}"
export REWARD_PY="${REWARD_PY:-python}"
export TASK_SUITE="${TASK_SUITE:-libero_spatial}"
export PRETRAINED_CKPT="${PRETRAINED_CKPT:-}"
export NUM_ACTIONS_CHUNK="${NUM_ACTIONS_CHUNK:-8}"

# Ensure imports resolve for both OpenVLA-OFT and reward server
export PYTHONPATH="${PROJ_ROOT}:${PROJ_ROOT}/openvla-oft:${PYTHONPATH:-}"

IFS=',' read -ra GPU_ARRAY <<< "${GPUS}"
NUM_GPUS=${#GPU_ARRAY[@]}

echo "=========================================="
echo "[OpenVLA-OFT] 多服务器模式启动"
echo "=========================================="
echo "  GPUs: ${GPUS} (${NUM_GPUS}个)"
echo "  Policy端口: ${BASE_POLICY_PORT}-$((BASE_POLICY_PORT + NUM_GPUS - 1))"
echo "  Reward端口: ${BASE_REWARD_PORT}-$((BASE_REWARD_PORT + NUM_GPUS - 1))"
echo "  负载均衡器参考端口: ${LB_PORT}"
echo "  Policy Python: ${POLICY_PY}"
echo "  Reward Python: ${REWARD_PY}"
echo "  LIBERO suite: ${TASK_SUITE}"
echo "=========================================="

POLICY_PIDS=()
REWARD_PIDS=()

for i in "${!GPU_ARRAY[@]}"; do
    GPU_ID="${GPU_ARRAY[$i]}"
    POLICY_PORT=$((BASE_POLICY_PORT + i))
    REWARD_PORT=$((BASE_REWARD_PORT + i))

    LOG_PREFIX="logs/openvla_parallel/gpu${GPU_ID}"

    echo ""
    echo "[GPU ${GPU_ID}] 启动服务器..."
    echo "  Policy端口: ${POLICY_PORT}"
    echo "  Reward端口: ${REWARD_PORT}"

    # Policy server
    CUDA_VISIBLE_DEVICES="${GPU_ID}" \
    "${POLICY_PY}" openvla-oft/scripts/serve_policy.py \
        --policy_server_port "${POLICY_PORT}" \
        --task_suite_name "${TASK_SUITE}" \
        --num_open_loop_steps "${NUM_ACTIONS_CHUNK}" \
        ${PRETRAINED_CKPT:+--pretrained_checkpoint "${PRETRAINED_CKPT}"} \
        > "${LOG_PREFIX}_policy_port${POLICY_PORT}.log" 2>&1 &

    POLICY_PID=$!
    POLICY_PIDS+=("${POLICY_PID}")
    echo "  Policy PID: ${POLICY_PID}"

    sleep 2

    # Reward server (Waitress)
    # 设置PYTHONPATH确保可以导入模块
    export PYTHONPATH="${PROJ_ROOT}:${PROJ_ROOT}/openvla-oft:${PYTHONPATH:-}"
    
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
        openvla_reward_server.app_openvla:create_app \
        > "${LOG_PREFIX}_reward_port${REWARD_PORT}.log" 2>&1 &

    REWARD_PID=$!
    REWARD_PIDS+=("${REWARD_PID}")
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
echo "  1. 启动负载均衡器: bash scripts/parallel/launch_load_balancer_openvla.sh"
echo "  2. 或直接访问某个服务器: curl http://localhost:${BASE_REWARD_PORT}/health"
echo ""
echo "按 Ctrl+C 停止所有服务器"
echo "=========================================="

trap 'echo ""; echo "停止所有服务器..."; kill ${POLICY_PIDS[@]} ${REWARD_PIDS[@]} 2>/dev/null; wait; echo "所有服务器已停止"; exit' INT TERM

wait

