#!/usr/bin/env bash
set -euo pipefail

SUITE="libero_spatial"
OUTDIR="./output/${SUITE}_$(date +%F_%H-%M-%S)"
LINKS="/home/baoshuntong/code/saftyEmbodyAI/embodied-red-teaming/libero-init-frames/json_data_for_rl/vlm_initial_state_links_new.json"

# 10 tasks example, 3 GPUs: 4,5,6
python -u a_evalUnderAttack_noTraining/infer_openvla_parell.py --gpu 4 --task-suite-name "$SUITE" --task-range "0-3" \
  --output-dir "$OUTDIR" --task-to-huglinks-json-path "$LINKS" 

python -u a_evalUnderAttack_noTraining/infer_openvla_parell.py --gpu 5 --task-suite-name "$SUITE" --task-range "4-6" \
  --output-dir "$OUTDIR" --task-to-huglinks-json-path "$LINKS" 

python -u a_evalUnderAttack_noTraining/infer_openvla_parell.py --gpu 6 --task-suite-name "$SUITE" --task-range "7-9" \
  --output-dir "$OUTDIR" --task-to-huglinks-json-path "$LINKS" 

wait
echo "All shards finished."
