#!/usr/bin/env bash
set -e
TIMESTAMP="$(date +%Y-%m-%d_%H-%M-%S)"
TYPE="SIMPLE_ENV_OCTO_POLICY"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../../.. && pwd)"
LOG_DIR="${ROOT_DIR}/logs/${TYPE}"
mkdir -p "${LOG_DIR}"
PORT="${PORT:-8001}"
GPU="${GPU:-0}"
cd "${ROOT_DIR}"

bash -c "CUDA_VISIBLE_DEVICES=${GPU} PORT=${PORT} python SimplerEnv/simple_env_policy_server_octo/app_policy.py" > "${LOG_DIR}/server_log_${TIMESTAMP}.txt" 2>&1 &

# nohup bash -c "CUDA_VISIBLE_DEVICES=${GPU} PORT=${PORT} python SimplerEnv/simple_env_policy_server_octo/app_policy.py" > "${LOG_DIR}/server_log_${TIMESTAMP}.txt" 2>&1 &
echo "$!"
