# CUDA_VISIBLE_DEVICES=0 python openpi/examples/libero/vlmRewrite_main.py \
#   --args.task-suite-name libero_spatial \
#   --args.num-trials-per-task 10
#!/usr/bin/env bash
cd openpi
source examples/libero/.venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJ_ROOT="$SCRIPT_DIR/.."
LOGDIR="$PROJ_ROOT/logs"
mkdir -p "$LOGDIR"

export CUDA_VISIBLE_DEVICES=1

nohup bash -c "
cd \"$PROJ_ROOT\"
mkdir -p \"$LOGDIR\"
for suite in libero_spatial libero_object libero_goal libero_10; do
  echo \"=== Running \$suite ===\"
  python openpi/examples/libero/vlmRewrite_main.py \
    --args.task-suite-name \"\$suite\" \
    --args.num-trials-per-task 10 \
    > \"$LOGDIR/\${suite}_\$(date +%Y%m%d_%H%M%S).log\" 2>&1
done
" > "$LOGDIR/loop.out" 2>&1 &
echo "loop pid=$!"




