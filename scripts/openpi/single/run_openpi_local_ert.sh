#!/usr/bin/env bash

# Simple launcher for API-based instruction generator (DashScope / Qwen-VL API).
# Edit the variables below to change port, mode and hyper-parameters.

set -e

# ===== User-configurable parameters =====
POLICY_HOST="127.0.0.1"
POLICY_PORT="4444"

GENERATOR_MODE="local"  # must be "api" for this script

NUM_TRIALS_PER_ANNOTATION=5
NUM_INSTRUCTIONS=5
SELECT_TOPK=3
N_ITER_ATTACK=2

LOCAL_MODEL_PATH="Qwen/Qwen3-VL-4B-Instruct"  # unused in api mode, but kept for symmetry

API_MODEL_NAME="qwen2.5-vl-72b-instruct"

# 要测试的多个 task suite
TASK_SUITES=(
  "libero_spatial"
  "libero_object"
  "libero_goal"
  "libero_10"
)

# ===== Script logic (normally you don't need to change below) =====
ROOT_DIR="/root/autodl-tmp/code/attackVLA/pi0_reward_server"
cd "${ROOT_DIR}"

# Record script start time
SCRIPT_START_TIME=$(date +%s)
echo "========== Script started at $(date '+%Y-%m-%d %H:%M:%S') =========="

for TASK_SUITE_NAME in "${TASK_SUITES[@]}"; do
  echo "========== Running ERT for task suite: ${TASK_SUITE_NAME} =========="
  TASK_START_TIME=$(date +%s)
  
  python3 embodyRedTeaming_baseline/main_openpi.py \
    --policy_host "${POLICY_HOST}" \
    --policy_port "${POLICY_PORT}" \
    --generator_mode "${GENERATOR_MODE}" \
    --num_trials_per_annotation "${NUM_TRIALS_PER_ANNOTATION}" \
    --num_instructions "${NUM_INSTRUCTIONS}" \
    --select_topk "${SELECT_TOPK}" \
    --n_iter_attack "${N_ITER_ATTACK}" \
    --local_model_path "${LOCAL_MODEL_PATH}" \
    --api_model_name "${API_MODEL_NAME}" \
    --task_suite_name "${TASK_SUITE_NAME}"
  
  TASK_END_TIME=$(date +%s)
  TASK_DURATION=$((TASK_END_TIME - TASK_START_TIME))
  TASK_HOURS=$((TASK_DURATION / 3600))
  TASK_MINUTES=$(((TASK_DURATION % 3600) / 60))
  TASK_SECONDS=$((TASK_DURATION % 60))
  echo "========== Task suite ${TASK_SUITE_NAME} completed in ${TASK_HOURS}h ${TASK_MINUTES}m ${TASK_SECONDS}s =========="
done

# Calculate and display total script execution time
SCRIPT_END_TIME=$(date +%s)
SCRIPT_DURATION=$((SCRIPT_END_TIME - SCRIPT_START_TIME))
SCRIPT_HOURS=$((SCRIPT_DURATION / 3600))
SCRIPT_MINUTES=$(((SCRIPT_DURATION % 3600) / 60))
SCRIPT_SECONDS=$((SCRIPT_DURATION % 60))
echo "========== Script ended at $(date '+%Y-%m-%d %H:%M:%S') =========="
echo "========== Total execution time: ${SCRIPT_HOURS}h ${SCRIPT_MINUTES}m ${SCRIPT_SECONDS}s =========="

