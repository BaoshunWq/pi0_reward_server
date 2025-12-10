#!/usr/bin/env bash
# Start a single OpenVLA-OFT reward server with Waitress.
# All paths are relative to the project root: /root/autodl-tmp/code/attackVLA/pi0_reward_server

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="${SCRIPT_DIR}"
cd "${PROJ_ROOT}"

export PYTHONPATH="${PROJ_ROOT}:${PYTHONPATH:-}"

# Ports can be overridden by env:
#   PORT=6100           reward server port
#   POLICY_PORT=23451   downstream policy websocket port
: "${PORT:=6100}"
: "${POLICY_PORT:=23451}"

echo "Starting OpenVLA-OFT reward server on port ${PORT}, policy port ${POLICY_PORT}"

waitress-serve \
  --host=0.0.0.0 \
  --port="${PORT}" \
  --threads=4 \
  --channel-timeout=600 \
  --call \
  openvla_reward_server.app_openvla:create_app


