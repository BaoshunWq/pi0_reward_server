#!/usr/bin/env bash

OPENPI_TYPE=PI0_LIBERO
TIMESTAMP="$(date +%Y-%m-%d_%H-%M-%S)"
LOG_DIR="logs"
LOG_FILE="${LOG_DIR}/${OPENPI_TYPE}/server_log_${TIMESTAMP}.txt"

mkdir -p "${LOG_DIR}"

# CUDA_VISIBLE_DEVICES=5,6,7 nohup uv run openpi/scripts/serve_policy.py --env LIBERO > "server_log_${TIMESTAMP}.txt" 2>&1 &
# CUDA_VISIBLE_DEVICES=1 nohup python openpi/scripts/serve_policy.py --env LIBERO --port 3333  > "${LOG_FILE}" 2>&1 &



# CUDA_VISIBLE_DEVICES=1 nohup python openpi/scripts/serve_policy.py --env ${OPENPI_TYPE} --port 3333 > "${LOG_FILE}" 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup python openpi/scripts/serve_policy.py --env ${OPENPI_TYPE} --port 4444 > "${LOG_FILE}" 2>&1 &