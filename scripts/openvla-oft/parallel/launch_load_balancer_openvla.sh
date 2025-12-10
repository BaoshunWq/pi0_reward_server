#!/usr/bin/env bash
set -euo pipefail

# Load balancer wrapper for OpenVLA-OFT reward servers
# Env overrides:
#   LB_PORT=6100          Load balancer port
#   BASE_REWARD_PORT=6101 First reward server port
#   NUM_SERVERS=1         Number of backend reward servers

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/../.."

export LB_PORT="${LB_PORT:-6100}"
export BASE_REWARD_PORT="${BASE_REWARD_PORT:-6101}"
export NUM_SERVERS="${NUM_SERVERS:-1}"

echo "=========================================="
echo "[OpenVLA-OFT] 启动负载均衡器"
echo "=========================================="
echo "  Load Balancer Port: ${LB_PORT}"
echo "  Backend Servers: ${BASE_REWARD_PORT} - $((BASE_REWARD_PORT + NUM_SERVERS - 1))"
echo "  策略: 最少连接数 (Least Connections)"
echo "=========================================="

python openvla_reward_server/load_balancer_openvla.py \
    --listen_port "${LB_PORT}" \
    --base_port "${BASE_REWARD_PORT}" \
    --num_servers "${NUM_SERVERS}"

