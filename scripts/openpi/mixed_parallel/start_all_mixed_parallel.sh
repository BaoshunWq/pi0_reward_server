#!/usr/bin/env bash
set -euo pipefail

# 混合并行启动：支持多卡，每张卡上可启动多个 Policy + Reward 实例，并附带负载均衡器
#
# 可覆盖的环境变量：
#   GPUS="0,1"                   - GPU 列表（逗号分隔）
#   INSTANCES_PER_GPU=2          - 每张 GPU 启动的实例数量（Policy/Reward 对）
#   BASE_POLICY_PORT=8000        - Policy 起始端口（全局连贯递增）
#   BASE_REWARD_PORT=6001        - Reward 起始端口（全局连贯递增）
#   LB_PORT=6000                 - 负载均衡器端口
#   POLICY_PY=/path/to/python    - Policy Python 解释器
#   REWARD_PY=/path/to/python    - Reward Python 解释器
#   OPENPI_ENV=PI0_LIBERO        - OpenPI 环境名称（如 PI0_LIBERO / PI05_LIBERO）
#
# 使用示例：
#   GPUS=0,1 INSTANCES_PER_GPU=3 BASE_POLICY_PORT=8000 BASE_REWARD_PORT=6001 LB_PORT=6000 \
#   POLICY_PY=/root/autodl-tmp/conda/envs/openpi/bin/python \
#   REWARD_PY=/root/autodl-tmp/conda/envs/openpi-libero_3_10/bin/python \
#   OPENPI_ENV=PI0_LIBERO \
#   bash scripts/openpi/mixed_parallel/start_all_mixed_parallel.sh
#
# 说明：不修改原有代码与脚本，仅新增本脚本以支持混合并行模式

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="${SCRIPT_DIR}/../../.."
cd "${PROJ_ROOT}"

mkdir -p logs/mixed_parallel

export GPUS="${GPUS:-0,1}"
export INSTANCES_PER_GPU="${INSTANCES_PER_GPU:-2}"
export BASE_POLICY_PORT="${BASE_POLICY_PORT:-8000}"
export BASE_REWARD_PORT="${BASE_REWARD_PORT:-6001}"
export LB_PORT="${LB_PORT:-6000}"
export POLICY_PY="${POLICY_PY:-/root/autodl-tmp/conda/envs/openpi/bin/python}"
export REWARD_PY="${REWARD_PY:-/root/autodl-tmp/conda/envs/openpi-libero_3_10/bin/python}"
export OPENPI_ENV="${OPENPI_ENV:-PI0_LIBERO}"

# 依赖检查
if [[ ! -x "${POLICY_PY}" ]]; then
  echo "[fatal] POLICY_PY not executable: ${POLICY_PY}" >&2
  exit 2
fi
if [[ ! -x "${REWARD_PY}" ]]; then
  echo "[fatal] REWARD_PY not executable: ${REWARD_PY}" >&2
  exit 2
fi

# 解析 GPU 列表
IFS=',' read -ra GPU_ARRAY <<< "${GPUS}"
NUM_GPUS=${#GPU_ARRAY[@]}
NUM_SERVERS=$((NUM_GPUS * INSTANCES_PER_GPU))
export NUM_SERVERS

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
for ((idx=0; idx<NUM_SERVERS; idx++)); do
  POLICY_PORT=$((BASE_POLICY_PORT + idx))
  REWARD_PORT=$((BASE_REWARD_PORT + idx))
  free_port "${POLICY_PORT}"
  free_port "${REWARD_PORT}"
done
free_port "${LB_PORT}"
echo "端口清理完成"
echo "=========================================="

echo "=========================================="
echo "启动混合并行服务架构"
echo "=========================================="
echo "  GPUs: ${GPUS}（${NUM_GPUS} 张）"
echo "  每 GPU 实例数: ${INSTANCES_PER_GPU}"
echo "  总实例数（Policy/Reward 对）: ${NUM_SERVERS}"
echo "  Policy端口: ${BASE_POLICY_PORT} - $((BASE_POLICY_PORT + NUM_SERVERS - 1))"
echo "  Reward端口: ${BASE_REWARD_PORT} - $((BASE_REWARD_PORT + NUM_SERVERS - 1))"
echo "  负载均衡器: ${LB_PORT}"
echo "  Policy Python: ${POLICY_PY}"
echo "  Reward Python: ${REWARD_PY}"
echo "  OpenPI 环境: ${OPENPI_ENV}"
echo "=========================================="

POLICY_PIDS=()
REWARD_PIDS=()

# 启动各 GPU 的多个实例
for i in "${!GPU_ARRAY[@]}"; do
  GPU_ID="${GPU_ARRAY[$i]}"
  for ((j=0; j<INSTANCES_PER_GPU; j++)); do
    GLOBAL_IDX=$((i * INSTANCES_PER_GPU + j))
    POLICY_PORT=$((BASE_POLICY_PORT + GLOBAL_IDX))
    REWARD_PORT=$((BASE_REWARD_PORT + GLOBAL_IDX))

    echo ""
    echo "[GPU ${GPU_ID}] 启动实例 ${j}..."
    echo "  Policy端口: ${POLICY_PORT}"
    echo "  Reward端口: ${REWARD_PORT}"

    # Policy server
    CUDA_VISIBLE_DEVICES="${GPU_ID}" \
    "${POLICY_PY}" openpi/scripts/serve_policy.py \
        --env "${OPENPI_ENV}" \
        --port "${POLICY_PORT}" \
        > "logs/mixed_parallel/server_policy_gpu${GPU_ID}_inst${j}_port${POLICY_PORT}.log" 2>&1 &

    POLICY_PID=$!
    POLICY_PIDS+=("${POLICY_PID}")
    echo "  Policy PID: ${POLICY_PID}"

    # 等待policy服务器启动
    sleep 2

    # Reward server（Waitress）
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
        > "logs/mixed_parallel/server_reward_gpu${GPU_ID}_inst${j}_port${REWARD_PORT}.log" 2>&1 &

    REWARD_PID=$!
    REWARD_PIDS+=("${REWARD_PID}")
    echo "  Reward PID: ${REWARD_PID}"
  done
done

echo ""
echo "所有服务器已启动，等待5秒确保服务就绪..."
sleep 5

# 启动负载均衡器
echo ""
echo "第二步：启动负载均衡器..."
echo "=========================================="
echo "  监听端口: ${LB_PORT}"
echo "  后端服务器: ${BASE_REWARD_PORT} - $((BASE_REWARD_PORT + NUM_SERVERS - 1))"
echo "  策略: 最少连接数 (Least Connections)"
echo "=========================================="

python scripts/openpi/parallel/load_balancer.py \
    --listen_port "${LB_PORT}" \
    --base_port "${BASE_REWARD_PORT}" \
    --num_servers "${NUM_SERVERS}" \
    > "logs/mixed_parallel/load_balancer_port${LB_PORT}.log" 2>&1 &

LB_PID=$!
echo "负载均衡器 PID: ${LB_PID}"

echo ""
echo "=========================================="
echo "✓ 混合并行服务已成功启动"
echo "=========================================="
echo ""
echo "服务信息:"
echo "  负载均衡器: http://localhost:${LB_PORT}"
echo "  健康检查: curl http://localhost:${LB_PORT}/health"
echo ""
echo "后端 Reward 服务器:"
for ((idx=0; idx<NUM_SERVERS; idx++)); do
  REWARD_PORT=$((BASE_REWARD_PORT + idx))
  echo "  实例 ${idx}: http://localhost:${REWARD_PORT}"
done
echo ""
echo "Policy 服务器:"
for ((idx=0; idx<NUM_SERVERS; idx++)); do
  POLICY_PORT=$((BASE_POLICY_PORT + idx))
  echo "  实例 ${idx}: http://localhost:${POLICY_PORT}"
done
echo ""
echo "日志目录: logs/mixed_parallel"
echo "  负载均衡器: load_balancer_port${LB_PORT}.log"
for i in "${!GPU_ARRAY[@]}"; do
  GPU_ID="${GPU_ARRAY[$i]}"
  for ((j=0; j<INSTANCES_PER_GPU; j++)); do
    GLOBAL_IDX=$((i * INSTANCES_PER_GPU + j))
    POLICY_PORT=$((BASE_POLICY_PORT + GLOBAL_IDX))
    REWARD_PORT=$((BASE_REWARD_PORT + GLOBAL_IDX))
    echo "  GPU ${GPU_ID} 实例 ${j} Policy: server_policy_gpu${GPU_ID}_inst${j}_port${POLICY_PORT}.log"
    echo "  GPU ${GPU_ID} 实例 ${j} Reward: server_reward_gpu${GPU_ID}_inst${j}_port${REWARD_PORT}.log"
  done
done
echo ""
echo "按 Ctrl+C 停止所有服务"
echo "=========================================="

trap 'echo ""; echo "停止所有服务..."; kill ${POLICY_PIDS[@]} ${REWARD_PIDS[@]} ${LB_PID} 2>/dev/null || true; wait || true; echo "所有服务已停止"; exit' INT TERM

wait

