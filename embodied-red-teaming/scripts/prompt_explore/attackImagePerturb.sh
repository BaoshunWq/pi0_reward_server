#!/usr/bin/env bash
set -euo pipefail

# 可修改：用于并行运行（false = 顺序执行；true = 每个命令放后台运行）
RUN_IN_BG=false

# 要遍历的扰动方法列表
methods=(
  "gaussian_noise"
  "shot_noise"
  "impulse_noise"
  "speckle_noise"
  "gaussian_blur"
  "defocus_blur"
  "glass_blur"
  "zoom_blur"
  "motion_blur"
  "fog"
  "frost"
  "snow"
  "spatter"
  "contrast"
  "brightness"
  "saturate"
  "pixelate"
  "elastic_transform"
  "jpeg_compression"
)

# 其它固定参数（如需改动可在这里调整）
CUDA_DEVICES="0,1"
PRETRAINED="qwen2.5-vl-72b-instruct"
TASK_SUITE="libero_spatial"
LINKS_JSON="libero-init-frames/json_data_for_rl/vlm_initial_state_links.json"
NUM_TRIALS=3
NUM_INSTRUCTIONS=5
PREFER_PROMPT_KEY="DISTRACTOR"
IS_IMG_PERTURB="True"
IS_REDT_ATTACK="False"

LOG_DIR="./logs"
mkdir -p "${LOG_DIR}"

for m in "${methods[@]}"; do
  echo "==== Running perturbation: ${m} ===="

  # 构造命令（单行，便于日志和复制）
  cmd="CUDA_VISIBLE_DEVICES=${CUDA_DEVICES} python a_evalUnderAttack/main.py \
    --pretrained_checkpoint ${PRETRAINED} \
    --task_suite_name ${TASK_SUITE} \
    --task_to_huglinks_json_path ${LINKS_JSON} \
    --num_trials_per_task ${NUM_TRIALS} \
    --num_instructions ${NUM_INSTRUCTIONS} \
    --prefer_prompt_key ${PREFER_PROMPT_KEY} \
    --is_img_perturb ${IS_IMG_PERTURB} \
    --is_red_teaming_attack ${IS_REDT_ATTACK} \
    --perturb_image_method ${m}"

  # 日志文件名（替换斜杠以防万一）
  safe_name="${m//\//_}"
  logfile="${LOG_DIR}/${safe_name}.log"

  if [ "${RUN_IN_BG}" = true ]; then
    # 后台执行（输出重定向），注意可能需要手动管理 GPU/显存冲突
    echo "Running in background; logs -> ${logfile}"
    eval "${cmd} > \"${logfile}\" 2>&1 &"
  else
    # 顺序执行，实时输出并同时写入日志（tee）
    echo "Command: ${cmd}"
    echo "Logging to ${logfile}"
    eval "${cmd}" 2>&1 | tee "${logfile}"
    echo "==== Finished ${m} ===="
  fi

  # 可选短暂停顿，防止某些资源争用问题（视情况打开）
  # sleep 5
done

echo "All done. Logs are in ${LOG_DIR}"

  
# CUDA_VISIBLE_DEVICES=0,1 python a_evalUnderAttack/main.py \
#     --pretrained_checkpoint qwen2.5-vl-72b-instruct \
#     --task_suite_name libero_spatial \
#     --task_to_huglinks_json_path libero-init-frames/json_data_for_rl/vlm_initial_state_links.json \
#     --num_trials_per_task 3 \
#     --num_instructions 5 \
#     --prefer_prompt_key DISTRACTOR \
#     --is_img_perturb True \
#     --is_red_teaming_attack False \
#     --perturb_image_method gaussian_noise \