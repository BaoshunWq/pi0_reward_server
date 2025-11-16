#!/usr/bin/env bash

set -euo pipefail

# Defaults
BACKEND="${BACKEND:-verl_qwen}"             # verl_qwen | qwenvl | qwen_llm
VERL_MODEL_PATH="${VERL_MODEL_PATH:-/root/autodl-tmp/trained_models/Qwen2.5-1.5B-Instruct_full_train_step200}"
QWEN_MODE="${QWEN_MODE:-api}"               # local | api
QWEN_VL_ID="${QWEN_VL_ID:-Qwen/Qwen2-VL-2B-Instruct}"
QWEN_LLM_ID="${QWEN_LLM_ID:-Qwen/Qwen2.5-7B-Instruct}"
TASK_SUITE="${TASK_SUITE:-libero_spatial}"  # libero_spatial | libero_object | libero_goal | libero_10 | libero_90
TRIALS="${TRIALS:-10}"
SAVE_VIDEOS="${SAVE_VIDEOS:-false}"
PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"
RESIZE="${RESIZE:-224}"
REPLAN="${REPLAN:-5}"
SEMANTIC_TYPE="${SEMANTIC_TYPE:-clip}"      # clip | deberta
NUM_INSTR="${NUM_INSTR:-10}"
TOPK="${TOPK:-5}"
FAIL_THRESHOLD="${FAIL_THRESHOLD:-5}"
TASK_LINKS_JSON="${TASK_LINKS_JSON:-libero-init-frames/json_data_for_rl/vlm_initial_state_links_new.json}"

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Options (env var overrides also supported):
  --backend {verl_qwen|qwenvl|qwen_llm}     Backend type (default: $BACKEND)
  --verl-model PATH                         VERL-trained Qwen model path (default: $VERL_MODEL_PATH)
  --qwen-mode {local|api}                   Qwen-VL mode (default: $QWEN_MODE)
  --qwen-vl-id HF_OR_PATH                   Qwen-VL model id/path (default: $QWEN_VL_ID)
  --qwen-llm-id HF_OR_PATH                  Qwen LLM id/path (default: $QWEN_LLM_ID)
  --task-suite NAME                         Libero suite (default: $TASK_SUITE)
  --trials N                                Num trials per task (default: $TRIALS)
  --save-videos {true|false}                Save rollout videos (default: $SAVE_VIDEOS)
  --host HOST                               Policy server host (default: $HOST)
  --port PORT                               Policy server port (default: $PORT)
  --resize N                                Observation resize (default: $RESIZE)
  --replan N                                Replan steps (default: $REPLAN)
  --semantic {clip|deberta}                 Semantic embedder (default: $SEMANTIC_TYPE)
  --num-instr N                             Num generated instructions (default: $NUM_INSTR)
  --topk N                                  Select top-k instructions (default: $TOPK)
  --fail-threshold N                        Failure threshold per annotation (default: $FAIL_THRESHOLD)
  --task-links-json PATH                    Task->links json (default: $TASK_LINKS_JSON)
  -h, --help                                Show this help

Examples:
  # Use VERL-trained Qwen (LLM) for rewriting
  $(basename "$0") --backend verl_qwen --verl-model /path/to/verl_model

  # Use Qwen-VL as the VLM backend
  $(basename "$0") --backend qwenvl --qwen-mode local --qwen-vl-id Qwen/Qwen2.5-VL-7B-Instruct

  # Use plain Qwen LLM as backend
  $(basename "$0") --backend qwen_llm --qwen-llm-id Qwen/Qwen2.5-7B-Instruct
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --backend) BACKEND="$2"; shift 2;;
    --verl-model) VERL_MODEL_PATH="$2"; shift 2;;
    --qwen-mode) QWEN_MODE="$2"; shift 2;;
    --qwen-vl-id) QWEN_VL_ID="$2"; shift 2;;
    --qwen-llm-id) QWEN_LLM_ID="$2"; shift 2;;
    --task-suite) TASK_SUITE="$2"; shift 2;;
    --trials) TRIALS="$2"; shift 2;;
    --save-videos) SAVE_VIDEOS="$2"; shift 2;;
    --host) HOST="$2"; shift 2;;
    --port) PORT="$2"; shift 2;;
    --resize) RESIZE="$2"; shift 2;;
    --replan) REPLAN="$2"; shift 2;;
    --semantic) SEMANTIC_TYPE="$2"; shift 2;;
    --num-instr) NUM_INSTR="$2"; shift 2;;
    --topk) TOPK="$2"; shift 2;;
    --fail-threshold) FAIL_THRESHOLD="$2"; shift 2;;
    --task-links-json) TASK_LINKS_JSON="$2"; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown option: $1"; usage; exit 1;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"

set -x
"$PYTHON" "$SCRIPT_DIR/vlmRewrite_main.py" \
  --BACKEND "$BACKEND" \
  --verl_model_path "$VERL_MODEL_PATH" \
  --qwen_mode "$QWEN_MODE" \
  --qwen_model_id "$QWEN_VL_ID" \
  --qwen_llm_model_id "$QWEN_LLM_ID" \
  --task_suite_name "$TASK_SUITE" \
  --num_trials_per_task "$TRIALS" \
  --save_videos "$SAVE_VIDEOS" \
  --host "$HOST" \
  --port "$PORT" \
  --resize_size "$RESIZE" \
  --replan_steps "$REPLAN" \
  --semantic_type "$SEMANTIC_TYPE" \
  --num_instructions "$NUM_INSTR" \
  --select_topk "$TOPK" \
  --failure_threshold "$FAIL_THRESHOLD" \
  --task_to_huglinks_json_path "$TASK_LINKS_JSON"
set +x


