#!/usr/bin/env bash
set -euo pipefail

# 一键启动 OpenVLA-OFT: 多GPU Policy + Reward + 负载均衡器
#
# 可覆盖的环境变量：
#   GPUS="0,1"                    - GPU 列表
#   BASE_POLICY_PORT=23451        - Policy 起始端口
#   BASE_REWARD_PORT=6101         - Reward 起始端口
#   LB_PORT=6100                  - 负载均衡器端口
#   POLICY_PY=python              - Policy Python 解释器
#   REWARD_PY=python              - Reward Python 解释器
#   PRETRAINED_CKPT=""            - HF checkpoint 路径/ID
#   NUM_ACTIONS_CHUNK=8           - open-loop 步数
#
# 示例：
#   GPUS=0,1 PRETRAINED_CKPT=moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10 \
#   bash scripts/openvla-oft/start_all_parallel_openvla.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="${SCRIPT_DIR}/../.."
cd "${PROJ_ROOT}"

mkdir -p logs/openvla_parallel

export GPUS="${GPUS:-0,1}"
export BASE_POLICY_PORT="${BASE_POLICY_PORT:-23451}"
export BASE_REWARD_PORT="${BASE_REWARD_PORT:-6101}"
export LB_PORT="${LB_PORT:-6100}"
export POLICY_PY="${POLICY_PY:-/root/autodl-tmp/conda/envs/openvla-oft/bin/python}"
export REWARD_PY="${REWARD_PY:-/root/autodl-tmp/conda/envs/openpi-libero_3_10/bin/python}"
export PRETRAINED_CKPT="${PRETRAINED_CKPT:-}"
export NUM_ACTIONS_CHUNK="${NUM_ACTIONS_CHUNK:-8}"

# 确保 import 路径
export PYTHONPATH="${PROJ_ROOT}:${PROJ_ROOT}/openvla-oft:${PYTHONPATH:-}"

# 检查 Python 解释器
if [[ ! -x "$(command -v "${POLICY_PY}")" ]]; then
    echo "[fatal] POLICY_PY 不可执行: ${POLICY_PY}" >&2
    exit 2
fi
if [[ ! -x "$(command -v "${REWARD_PY}")" ]]; then
    echo "[fatal] REWARD_PY 不可执行: ${REWARD_PY}" >&2
    exit 2
fi

# 解析 GPU 列表
IFS=',' read -ra GPU_ARRAY <<< "${GPUS}"
NUM_GPUS=${#GPU_ARRAY[@]}
export NUM_SERVERS="${NUM_GPUS}"

# 端口清理函数
free_port() {
    local port="$1"
    local pids
    pids="$(lsof -ti TCP:"${port}" 2>/dev/null || true)"
    if [[ -n "${pids}" ]]; then
        echo "  -> 端口 ${port} 被进程 ${pids} 占用，尝试释放..."
        kill ${pids} 2>/dev/null || true
        sleep 1
        if lsof -ti TCP:"${port}" >/dev/null 2>&1; then
            echo "  -> 端口 ${port} 仍被占用，执行强制终止..."
            kill -9 ${pids} 2>/dev/null || true
        fi
    fi
}

echo ""
echo "清理历史端口占用..."
echo "=========================================="
for i in "${!GPU_ARRAY[@]}"; do
    POLICY_PORT=$((BASE_POLICY_PORT + i))
    REWARD_PORT=$((BASE_REWARD_PORT + i))
    free_port "${POLICY_PORT}"
    free_port "${REWARD_PORT}"
done
free_port "${LB_PORT}"
echo "端口清理完成"
echo "=========================================="

echo "=========================================="
echo "启动 OpenVLA-OFT 并行服务架构"
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

trap cleanup INT TERM EXIT

echo ""
echo "第一步：并行启动所有Policy服务器..."
echo "=========================================="

# 先并行启动所有 Policy 服务器
for i in "${!GPU_ARRAY[@]}"; do
    GPU_ID="${GPU_ARRAY[$i]}"
    POLICY_PORT=$((BASE_POLICY_PORT + i))

    echo ""
    echo "[GPU ${GPU_ID}] 启动 Policy 服务器..."
    echo "  Policy端口: ${POLICY_PORT}"

    # Policy server
    CUDA_VISIBLE_DEVICES="${GPU_ID}" \
    "${POLICY_PY}" openvla-oft/scripts/serve_policy.py \
        --policy_server_port "${POLICY_PORT}" \
        --num_open_loop_steps "${NUM_ACTIONS_CHUNK}" \
        ${PRETRAINED_CKPT:+--pretrained_checkpoint "${PRETRAINED_CKPT}"} \
        > "logs/openvla_parallel/server_policy_gpu${GPU_ID}_port${POLICY_PORT}.log" 2>&1 &

    POLICY_PID=$!
    POLICY_PIDS+=("${POLICY_PID}")
    echo "  Policy PID: ${POLICY_PID}"
done

# 统一等待所有 Policy 服务器启动
echo ""
echo "等待所有 Policy 服务器启动（最多等待 5 分钟）..."
MAX_WAIT=300  # 5分钟
WAIT_INTERVAL=5  # 每5秒检查一次
ELAPSED=0
POLICY_PORTS=()
for i in "${!GPU_ARRAY[@]}"; do
    POLICY_PORTS+=($((BASE_POLICY_PORT + i)))
done

while [ ${ELAPSED} -lt ${MAX_WAIT} ]; do
    ALL_READY=true
    for POLICY_PORT in "${POLICY_PORTS[@]}"; do
        if ! lsof -ti TCP:"${POLICY_PORT}" >/dev/null 2>&1; then
            ALL_READY=false
            break
        fi
    done
    
    if [ "${ALL_READY}" = true ]; then
        echo "✓ 所有 Policy 服务器已启动"
        break
    fi
    
    sleep ${WAIT_INTERVAL}
    ELAPSED=$((ELAPSED + WAIT_INTERVAL))
    if [ $((ELAPSED % 30)) -eq 0 ]; then
        echo "  仍在等待 Policy 服务器启动... (已等待 ${ELAPSED} 秒)"
        for POLICY_PORT in "${POLICY_PORTS[@]}"; do
            if lsof -ti TCP:"${POLICY_PORT}" >/dev/null 2>&1; then
                echo "    ✓ 端口 ${POLICY_PORT} 已就绪"
            else
                echo "    ⏳ 端口 ${POLICY_PORT} 未就绪"
            fi
        done
    fi
done

# 检查是否有未启动的 Policy 服务器
for i in "${!GPU_ARRAY[@]}"; do
    POLICY_PORT=$((BASE_POLICY_PORT + i))
    if ! lsof -ti TCP:"${POLICY_PORT}" >/dev/null 2>&1; then
        echo "  ⚠ 警告: GPU ${GPU_ARRAY[$i]} 的 Policy 服务器在 ${MAX_WAIT} 秒内未启动"
    fi
done

echo ""
echo "第二步：并行启动所有Reward服务器..."
echo "=========================================="

# 设置PYTHONPATH确保可以导入模块
export PYTHONPATH="${PROJ_ROOT}:${PROJ_ROOT}/openvla-oft:${PYTHONPATH:-}"

# 并行启动所有 Reward 服务器
for i in "${!GPU_ARRAY[@]}"; do
    GPU_ID="${GPU_ARRAY[$i]}"
    POLICY_PORT=$((BASE_POLICY_PORT + i))
    REWARD_PORT=$((BASE_REWARD_PORT + i))

    echo ""
    echo "[GPU ${GPU_ID}] 启动 Reward 服务器..."
    echo "  Reward端口: ${REWARD_PORT}"
    echo "  关联Policy端口: ${POLICY_PORT}"
    
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
        > "logs/openvla_parallel/server_reward_gpu${GPU_ID}_port${REWARD_PORT}.log" 2>&1 &

    REWARD_PID=$!
    REWARD_PIDS+=("${REWARD_PID}")
    echo "  Reward PID: ${REWARD_PID}"
done

# 统一等待所有 Reward 服务器启动
echo ""
echo "等待所有 Reward 服务器启动（最多等待 2 分钟）..."
MAX_WAIT_REWARD=120  # 2分钟
ELAPSED=0
REWARD_PORTS=()
for i in "${!GPU_ARRAY[@]}"; do
    REWARD_PORTS+=($((BASE_REWARD_PORT + i)))
done

while [ ${ELAPSED} -lt ${MAX_WAIT_REWARD} ]; do
    ALL_READY=true
    for REWARD_PORT in "${REWARD_PORTS[@]}"; do
        if ! lsof -ti TCP:"${REWARD_PORT}" >/dev/null 2>&1; then
            ALL_READY=false
            break
        fi
    done
    
    if [ "${ALL_READY}" = true ]; then
        echo "✓ 所有 Reward 服务器已启动"
        break
    fi
    
    sleep ${WAIT_INTERVAL}
    ELAPSED=$((ELAPSED + WAIT_INTERVAL))
    if [ $((ELAPSED % 30)) -eq 0 ]; then
        echo "  仍在等待 Reward 服务器启动... (已等待 ${ELAPSED} 秒)"
    fi
done

# 检查是否有未启动的 Reward 服务器
for i in "${!GPU_ARRAY[@]}"; do
    REWARD_PORT=$((BASE_REWARD_PORT + i))
    if ! lsof -ti TCP:"${REWARD_PORT}" >/dev/null 2>&1; then
        echo "  ⚠ 警告: GPU ${GPU_ARRAY[$i]} 的 Reward 服务器在 ${MAX_WAIT_REWARD} 秒内未启动"
    fi
done

echo ""
echo "第三步：启动负载均衡器..."
echo "=========================================="
echo "  监听端口: ${LB_PORT}"
echo "  后端服务器: ${BASE_REWARD_PORT} - $((BASE_REWARD_PORT + NUM_SERVERS - 1))"
echo "  策略: 最少连接数 (Least Connections)"
echo "=========================================="

python openvla_reward_server/load_balancer_openvla.py \
    --listen_port "${LB_PORT}" \
    --base_port "${BASE_REWARD_PORT}" \
    --num_servers "${NUM_SERVERS}" \
    > "logs/openvla_parallel/load_balancer_port${LB_PORT}.log" 2>&1 &

LB_PID=$!
echo "负载均衡器 PID: ${LB_PID}"

echo ""
echo "=========================================="
echo "✓ OpenVLA-OFT 全部服务已成功启动"
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
echo "日志文件路径: logs/openvla_parallel/"
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

wait

