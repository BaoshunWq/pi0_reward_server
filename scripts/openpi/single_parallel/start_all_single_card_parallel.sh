#!/usr/bin/env bash
set -euo pipefail

# 单卡多开：在同一块 GPU 上并行启动多个 Policy + Reward 实例，并附带负载均衡器
#
# 可覆盖的环境变量：
#   GPU_ID=0                    - 目标 GPU（单卡）
#   NUM_INSTANCES=2             - 启动的实例数量（Policy/Reward 对的数量）
#   BASE_POLICY_PORT=8000       - Policy 起始端口（按实例递增）
#   BASE_REWARD_PORT=6001       - Reward 起始端口（按实例递增）
#   LB_PORT=6000                - 负载均衡器端口
#   POLICY_PY=/path/to/python   - Policy 服务器 Python 解释器
#   REWARD_PY=/path/to/python   - Reward 服务器 Python 解释器
#   OPENPI_ENV=PI0_LIBERO       - OpenPI 环境名称（示例：PI0_LIBERO / PI05_LIBERO）
#
# 使用示例：
#   GPU_ID=0 NUM_INSTANCES=4 BASE_POLICY_PORT=8000 BASE_REWARD_PORT=6001 LB_PORT=6000 \
#   POLICY_PY=/root/autodl-tmp/conda/envs/openpi/bin/python \
#   REWARD_PY=/root/autodl-tmp/conda/envs/openpi-libero_3_10/bin/python \
#   OPENPI_ENV=PI0_LIBERO \
#   bash scripts/openpi/single_parallel/start_all_single_card_parallel.sh
#
# 注意：不修改任何现有脚本与代码，仅在新建文件夹中提供单卡多开方案

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="${SCRIPT_DIR}/../../.."
cd "${PROJ_ROOT}"

mkdir -p logs/single_card_parallel

export GPU_ID="${GPU_ID:-0}"
export NUM_INSTANCES="${NUM_INSTANCES:-2}"
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
for ((i=0; i<NUM_INSTANCES; i++)); do
  POLICY_PORT=$((BASE_POLICY_PORT + i))
  REWARD_PORT=$((BASE_REWARD_PORT + i))
  free_port "${POLICY_PORT}"
  free_port "${REWARD_PORT}"
done
free_port "${LB_PORT}"
echo "端口清理完成"
echo "=========================================="

echo "=========================================="
echo "启动单卡并行服务架构"
echo "=========================================="
echo "  GPU: ${GPU_ID}"
echo "  实例数量: ${NUM_INSTANCES}"
echo "  Policy端口: ${BASE_POLICY_PORT} - $((BASE_POLICY_PORT + NUM_INSTANCES - 1))"
echo "  Reward端口: ${BASE_REWARD_PORT} - $((BASE_REWARD_PORT + NUM_INSTANCES - 1))"
echo "  负载均衡器: ${LB_PORT}"
echo "  Policy Python: ${POLICY_PY}"
echo "  Reward Python: ${REWARD_PY}"
echo "=========================================="

POLICY_PIDS=()
REWARD_PIDS=()

for ((i=0; i<NUM_INSTANCES; i++)); do
  POLICY_PORT=$((BASE_POLICY_PORT + i))
  REWARD_PORT=$((BASE_REWARD_PORT + i))

  echo ""
  echo "[GPU ${GPU_ID}] 启动实例 ${i}..."
  echo "  Policy端口: ${POLICY_PORT}"
  echo "  Reward端口: ${REWARD_PORT}"

  # 启动 Policy 服务器（绑定同一 GPU）
  CUDA_VISIBLE_DEVICES="${GPU_ID}" \
  "${POLICY_PY}" openpi/scripts/serve_policy.py \
      --env "${OPENPI_ENV}" \
      --port "${POLICY_PORT}" \
      > "logs/single_card_parallel/server_policy_gpu${GPU_ID}_instance${i}_port${POLICY_PORT}.log" 2>&1 &

  POLICY_PID=$!
  POLICY_PIDS+=("${POLICY_PID}")
  echo "  Policy PID: ${POLICY_PID}"

  # 等待 policy 启动
  sleep 2

  # 启动 Reward 服务器（禁用 GPU 池，明确绑定同一 GPU）
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
      > "logs/single_card_parallel/server_reward_gpu${GPU_ID}_instance${i}_port${REWARD_PORT}.log" 2>&1 &

  REWARD_PID=$!
  REWARD_PIDS+=("${REWARD_PID}")
  echo "  Reward PID: ${REWARD_PID}"
done

echo ""
echo "所有服务器已启动，等待5秒确保服务就绪..."
sleep 5

echo ""
echo "第二步：启动负载均衡器..."
echo "=========================================="
echo "  监听端口: ${LB_PORT}"
echo "  后端服务器: ${BASE_REWARD_PORT} - $((BASE_REWARD_PORT + NUM_INSTANCES - 1))"
echo "  策略: 最少连接数 (Least Connections)"
echo "=========================================="

python scripts/openpi/parallel/load_balancer.py \
    --listen_port "${LB_PORT}" \
    --base_port "${BASE_REWARD_PORT}" \
    --num_servers "${NUM_INSTANCES}" \
    > "logs/single_card_parallel/load_balancer_port${LB_PORT}.log" 2>&1 &

LB_PID=$!
echo "负载均衡器 PID: ${LB_PID}"

echo ""
echo "=========================================="
echo "✓ 单卡并行服务已成功启动"
echo "=========================================="
echo ""
echo "服务信息:"
echo "  负载均衡器: http://localhost:${LB_PORT}"
echo "  健康检查: curl http://localhost:${LB_PORT}/health"
echo ""
echo "后端Reward服务器:"
for ((i=0; i<NUM_INSTANCES; i++)); do
  REWARD_PORT=$((BASE_REWARD_PORT + i))
  echo "  实例 ${i}: http://localhost:${REWARD_PORT}"
done
echo ""
echo "Policy服务器:"
for ((i=0; i<NUM_INSTANCES; i++)); do
  POLICY_PORT=$((BASE_POLICY_PORT + i))
  echo "  实例 ${i}: http://localhost:${POLICY_PORT}"
done
echo ""
echo "日志文件目录: logs/single_card_parallel"
echo "  负载均衡器: load_balancer_port${LB_PORT}.log"
for ((i=0; i<NUM_INSTANCES; i++)); do
  POLICY_PORT=$((BASE_POLICY_PORT + i))
  REWARD_PORT=$((BASE_REWARD_PORT + i))
  echo "  实例 ${i} Policy: server_policy_gpu${GPU_ID}_instance${i}_port${POLICY_PORT}.log"
  echo "  实例 ${i} Reward: server_reward_gpu${GPU_ID}_instance${i}_port${REWARD_PORT}.log"
done
echo ""
echo "按 Ctrl+C 停止所有服务"
echo "=========================================="

trap 'echo ""; echo "停止所有服务..."; kill ${POLICY_PIDS[@]} ${REWARD_PIDS[@]} ${LB_PID} 2>/dev/null || true; wait || true; echo "所有服务已停止"; exit' INT TERM

wait

