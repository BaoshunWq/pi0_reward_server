#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="/root/autodl-tmp/code/attackVLA/pi0_reward_server"
cd "${PROJ_ROOT}"
export PYTHONPATH="${PROJ_ROOT}:${PYTHONPATH:-}"
: "${PORT:=6102}"
: "${POLICY_PORT:=8002}"
echo "Starting SimpleEnv RT1 reward server on port ${PORT}, policy port ${POLICY_PORT}"
waitress-serve \
  --host=0.0.0.0 \
  --port="${PORT}" \
  --threads=4 \
  --channel-timeout=600 \
  --call \
  SimplerEnv.simple_env_reward_server_rt1.app_simple_env:create_app

