#!/usr/bin/env bash
# 多服务器模式 - 每个GPU独立运行policy和reward服务器
# 这是解决EGL上下文冲突的最佳方案

GPUS="0,1" \
POLICY_PY="/root/autodl-tmp/conda/envs/openpi/bin/python" \
REWARD_PY="/root/autodl-tmp/conda/envs/openpi-libero/bin/python" \
BASE_POLICY_PORT=8000 \
BASE_REWARD_PORT=6001 \
LB_PORT=6000 \
OPENPI_ENV=PI0_LIBERO \
bash scripts/parallel/launch_multi_servers.sh

# 多服务器架构说明:
# GPU 0: Policy(8000) + Reward(6001)
# GPU 1: Policy(8001) + Reward(6002)
# 负载均衡器: 6000 → 分发到6001,6002
#
# 优点:
# - 每个GPU完全独立，无EGL冲突
# - 真正的并行处理
# - 可扩展到更多GPU
#
# 使用方法:
# 1. 启动服务器: bash scripts/unify_multi_servers.sh
# 2. 启动负载均衡器: NUM_SERVERS=2 bash scripts/parallel/launch_load_balancer.sh
# 3. 发送请求到: http://localhost:6000/score
