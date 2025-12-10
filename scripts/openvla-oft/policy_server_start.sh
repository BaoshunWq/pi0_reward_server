#!/usr/bin/env bash
# 单独启动 Policy 服务器脚本
# 确保与并行启动脚本使用相同的环境设置

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="${SCRIPT_DIR}/../.."
cd "${PROJ_ROOT}"

# 确保 import 路径（与并行启动脚本保持一致）
export PYTHONPATH="${PROJ_ROOT}:${PROJ_ROOT}/openvla-oft:${PYTHONPATH:-}"

# 使用与并行启动相同的 Python 环境
export POLICY_PY="${POLICY_PY:-/root/autodl-tmp/conda/envs/openvla-oft/bin/python}"

CUDA_VISIBLE_DEVICES=1 "${POLICY_PY}" openvla-oft/scripts/serve_policy.py \
  # --pretrained-checkpoint moojink/openvla-7b-oft-finetuned-libero-spatial \
  # --policy-server-port 23451 \
  # --task_suite_name libero_spatial \
  # --num_open_loop_steps 8