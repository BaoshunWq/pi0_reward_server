#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="/root/autodl-tmp/code/attackVLA/pi0_reward_server"
cd "${PROJ_ROOT}"
export PYTHONPATH="${PROJ_ROOT}:${PYTHONPATH:-}"
: "${PORT:=8001}"
echo "Starting Octo policy server on port ${PORT}"
waitress-serve \
  --host=0.0.0.0 \
  --port="${PORT}" \
  --threads=4 \
  --channel-timeout=600 \
  --call \
  SimplerEnv.simple_env_policy_server_octo.app_policy:create_app

