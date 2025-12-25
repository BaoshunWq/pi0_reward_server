#!/usr/bin/env bash
set -e
TIMESTAMP="$(date +%Y-%m-%d_%H-%M-%S)"
TYPE="SIMPLE_ENV_RT1_REWARD"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../../.. && pwd)"
LOG_DIR="${ROOT_DIR}/logs/${TYPE}"
mkdir -p "${LOG_DIR}"
PORT="${PORT:-6102}"
POLICY_PORT="${POLICY_PORT:-8002}"
GPU="${GPU:-0}"
cd "${ROOT_DIR}"
nohup bash -c "CUDA_VISIBLE_DEVICES=${GPU} PORT=${PORT} POLICY_PORT=${POLICY_PORT} python SimplerEnv/simple_env_reward_server_rt1/app_simple_env.py" > "${LOG_DIR}/reward_log_${TIMESTAMP}.txt" 2>&1 &
echo "$!"
