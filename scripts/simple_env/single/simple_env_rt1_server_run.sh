#!/usr/bin/env bash
set -e
TIMESTAMP="$(date +%Y-%m-%d_%H-%M-%S)"
TYPE="SIMPLE_ENV_RT1_POLICY"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../../.. && pwd)"
LOG_DIR="${ROOT_DIR}/logs/${TYPE}"
mkdir -p "${LOG_DIR}"
PORT="${PORT:-8002}"
GPU="${GPU:-0}"
cd "${ROOT_DIR}"
nohup bash -c "CUDA_VISIBLE_DEVICES=${GPU} PORT=${PORT} python SimplerEnv/simple_env_policy_server_rt1/app_policy.py" > "${LOG_DIR}/server_log_${TIMESTAMP}.txt" 2>&1 &
echo "$!"
