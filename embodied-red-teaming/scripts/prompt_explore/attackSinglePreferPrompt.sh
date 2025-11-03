
  
CUDA_VISIBLE_DEVICES=0,1 python a_evalUnderAttack/main.py \
    --pretrained_checkpoint qwen2.5-vl-72b-instruct \
    --task_suite_name libero_spatial \
    --task_to_huglinks_json_path libero-init-frames/json_data_for_rl/vlm_initial_state_links.json \
    --num_trials_per_task 3 \
    --num_instructions 5 \
    --prefer_prompt_key DISTRACTOR \