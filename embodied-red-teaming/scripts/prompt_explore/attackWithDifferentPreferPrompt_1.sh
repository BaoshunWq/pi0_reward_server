#!/usr/bin/env bash
set -euo pipefail

# Configs (override via env if needed)
MODEL="${MODEL:-qwen2.5-vl-72b-instruct}"
TASK_SUITE="${TASK_SUITE:-libero_spatial}"
JSON_PATH="${JSON_PATH:-libero-init-frames/json_data_for_rl/vlm_initial_state_links.json}"
TRIALS="${TRIALS:-3}"
NUM_INSTR="${NUM_INSTR:-6}"
TOPK="${TOPK:-3}"
CUDA_DEV="${CUDA_VISIBLE_DEVICES:-4,5}"

# 12 categories in PROMPT_CATEGORIES
CATEGORIES=(
  REL_SYN
  VERB_STYLE
  ORDER
  FORMAT
#   FUNC_WORDS
#   CHAR_NOISE
#   PRONOUN
#   UNDER_OVER
#   FRAME
#   NEGATION
#   CONFLICT
#   DISTRACTOR
)

mkdir -p logs

for KEY in "${CATEGORIES[@]}"; do
  echo ">>> Running prefer_prompt_key=${KEY}"
  CUDA_VISIBLE_DEVICES="${CUDA_DEV}" python a_evalUnderAttack/main.py \
    --pretrained_checkpoint "${MODEL}" \
    --task_suite_name "${TASK_SUITE}" \
    --task_to_huglinks_json_path "${JSON_PATH}" \
    --num_trials_per_task "${TRIALS}" \
    --num_instructions "${NUM_INSTR}" \
    --select_topk "${TOPK}" \
    --prefer_prompt_key "${KEY}" \
    | tee "logs/${TASK_SUITE}_${MODEL}_${KEY}.log"
done
