#!/usr/bin/env bash
set -euo pipefail

# Launch load balancer for multiple reward servers
# 
# Env overrides:
#   LB_PORT=6000          - Load balancer port
#   BASE_REWARD_PORT=6001 - First reward server port
#   NUM_SERVERS=2         - Number of reward servers (default: 2)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/../.."

export LB_PORT="${LB_PORT:-6000}"
export BASE_REWARD_PORT="${BASE_REWARD_PORT:-6001}"
export NUM_SERVERS="${NUM_SERVERS:-2}"

echo "=========================================="
echo "Starting Load Balancer"
echo "=========================================="
echo "  Load Balancer Port: ${LB_PORT}"
echo "  Backend Servers: ${BASE_REWARD_PORT} - $((BASE_REWARD_PORT + NUM_SERVERS - 1))"
echo "  策略: 最少连接数 (Least Connections)"
echo "=========================================="

python scripts/openpi/parallel/load_balancer.py \
    --listen_port "${LB_PORT}" \
    --base_port "${BASE_REWARD_PORT}" \
    --num_servers "${NUM_SERVERS}"
