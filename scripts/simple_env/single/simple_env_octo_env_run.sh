#!/usr/bin/env bash
set -e
TIMESTAMP="$(date +%Y-%m-%d_%H-%M-%S)"
TYPE="SIMPLE_ENV_OCTO_REWARD"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../../.. && pwd)"
LOG_DIR="${ROOT_DIR}/logs/${TYPE}"
mkdir -p "${LOG_DIR}"
PORT="${PORT:-6101}"
POLICY_PORT="${POLICY_PORT:-8001}"
GPU="${GPU:-0}"
cd "${ROOT_DIR}"
nohup bash -c "CUDA_VISIBLE_DEVICES=${GPU} PORT=${PORT} POLICY_PORT=${POLICY_PORT} python SimplerEnv/simple_env_reward_server_octo/app_simple_env.py" > "${LOG_DIR}/reward_log_${TIMESTAMP}.txt" 2>&1 &
echo "$!"
