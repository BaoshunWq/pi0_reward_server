#!/usr/bin/env bash
set -euo pipefail

# Minimal wrapper for unified_eval.py
# Maps requested flags to actual CLI names in Args
# --backend -> --BACKEND
# --verl-model -> --verl_model_path
# --task-suite -> --task_suite_name
# --trials -> --num_trials_per_task

export CUDA_VISIBLE_DEVICES=2

PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../../openpi/examples/libero && pwd)"

"$PYTHON_BIN" "$SCRIPT_DIR/unified_eval.py" \
  --BACKEND "verl_qwen" \
  --verl_model_path "/root/autodl-tmp/code/attackVLA/rover_verl/checkpoints/custom_rover_qwen2_5_vl_lora_20251113_121506/global_step_100" \
  --task_suite_name "libero_spatial" \
  --num_trials_per_task 1 \
  --mode "vlm" \
  --wandb_project "unified_eval_test" \
  --wandb_entity "tongbs-sysu" \
  --host "0.0.0.0" \
  --port 12345 \
  --resize_size 224 \
  --num_instructions 1 \
  --select_topk 5 \
  --failure_threshold 5 \
  --task_to_huglinks_json_path "libero-init-frames/json_data_for_rl/vlm_initial_state_links_new.json" \
  --output_path "data/unified_eval_results.json" \
  --whole_acc_log_path "data/whole_acc_log.json" \



