

VERL_MODEL_PATH=${1:-"/root/autodl-tmp/trained_models/Qwen2.5-1.5B-Instruct_full_train_step200"}
OUTPUT_DIR=${2:-"output/test_vlmrewrite"}
# 构建命令
python openpi/examples/libero/vlmRewrite_main.py \
    --save-videos False \
    --verl-model-path $VERL_MODEL_PATH \
    --output-path $OUTPUT_DIR/results/vlmrewrite_results.json \
    --whole-acc-log-path $OUTPUT_DIR/results/whole_acc_log.json"


