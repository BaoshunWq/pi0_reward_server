#!/usr/bin/env bash

set -e

# Policy server endpoint
POLICY_HOST="0.0.0.0"
POLICY_PORT="23451"

# Instruction generation
GENERATOR_MODE="api"            # local or api
LOCAL_MODEL_PATH="verl_trained_ckpts/rover_12_03_qwen3_vl_4b"
API_MODEL_NAME="qwen2.5-vl-72b-instruct"

# Attack parameters
NUM_TRIALS_PER_ANNOTATION=1
NUM_INSTRUCTIONS=5
SELECT_TOPK=1
N_ITER_ATTACK=1
FAILURE_THRESHOLD=0.5

# Task suites to evaluate
TASK_SUITES=(
  "libero_spatial"
)

ROOT_DIR="/data1/baoshuntong/code/attackVLA/pi0_reward_server"
cd "${ROOT_DIR}"

echo "========== OpenVLA-OFT ERT started at $(date '+%Y-%m-%d %H:%M:%S') =========="

for TASK_SUITE_NAME in "${TASK_SUITES[@]}"; do
  echo "========== Running OpenVLA-OFT ERT for task suite: ${TASK_SUITE_NAME} =========="

  python3 embodyRedTeaming_baseline/run_openvla_oft_ert.py \
    --policy_host "${POLICY_HOST}" \
    --policy_port "${POLICY_PORT}" \
    --generator_mode "${GENERATOR_MODE}" \
    --num_trials_per_annotation "${NUM_TRIALS_PER_ANNOTATION}" \
    --num_instructions "${NUM_INSTRUCTIONS}" \
    --select_topk "${SELECT_TOPK}" \
    --n_iter_attack "${N_ITER_ATTACK}" \
    --failure_threshold "${FAILURE_THRESHOLD}" \
    --local_model_path "${LOCAL_MODEL_PATH}" \
    --api_model_name "${API_MODEL_NAME}" \
    --task_suite_name "${TASK_SUITE_NAME}"

  echo "========== Completed suite: ${TASK_SUITE_NAME} =========="
done

echo "========== OpenVLA-OFT ERT finished at $(date '+%Y-%m-%d %H:%M:%S') =========="

