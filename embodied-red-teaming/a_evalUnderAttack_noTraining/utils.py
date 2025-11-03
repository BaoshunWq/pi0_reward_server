# import sys
# sys.path.insert(0, "/home/baoshuntong/code/saftyEmbodyAI/embodied-red-teaming/openvla")
from dataclasses import dataclass
import numpy as np
import tqdm
from libero.libero import benchmark
import os
from generate_intru_smolvlm import build_red_team_generator, CLIPEmbeddingModel
from pathlib import Path
from typing import Optional, Union, Dict, List, Tuple
import torch
import gc
from transformers import pipeline

from openvla.experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    quat2axisangle,
    save_rollout_video,
)
from openvla.experiments.robot.robot_utils import (
    DATE_TIME,
    get_action,
    get_image_resize_size,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)
from openvla.experiments.robot.openvla_utils import get_processor
from openvla.experiments.robot.robot_utils import get_model


def eval_libero(cfg,model,task_id,task_description,is_img_perturb=False,img_perturb_func=None) -> None:

    print(f" \n----------VLA model infer in libero environment---------- ==============with annotation : {task_description} =================\n")
    
    assert cfg.pretrained_checkpoint is not None, "cfg.pretrained_checkpoint must not be None!"
    if "image_aug" in cfg.pretrained_checkpoint:
        assert cfg.center_crop, "Expecting `center_crop==True` because model was trained with image augmentations!"
    assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"

    # Set random seed
    set_seed_everywhere(cfg.seed)

    # [OpenVLA] Set action un-normalization key
    cfg.unnorm_key = cfg.task_suite_name

    processor = None
    if cfg.model_family == "openvla":
        processor = get_processor(cfg)
    



    # [OpenVLA] Check that the model contains the action un-normalization key
    if cfg.model_family == "openvla":
        if cfg.unnorm_key not in model.norm_stats and f"{cfg.unnorm_key}_no_noops" in model.norm_stats:
            cfg.unnorm_key = f"{cfg.unnorm_key}_no_noops"
        assert cfg.unnorm_key in model.norm_stats, f"Action un-norm key {cfg.unnorm_key} not found in VLA `norm_stats`!"


    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    task = task_suite.get_task(task_id)
    initial_states = task_suite.get_task_init_states(task_id)
    env, _ = get_libero_env(task, cfg.model_family, resolution=256)

    # Initialize LIBERO task suite

    # print(f"Task suite: {cfg.task_suite_name}")
    # log_file.write(f"Task suite: {cfg.task_suite_name}\n")

    # Get expected image dimensions
    resize_size = get_image_resize_size(cfg)

    # Start evaluation
    total_episodes, total_successes = 0, 0

    # Start episodes
    task_episodes, task_successes = 0, 0
    for episode_idx in range(cfg.num_trials_per_task):
        # print(f"\nTask: {task_description}")
        # log_file.write(f"\nTask: {task_description}\n")

        # Reset environment
        env.reset()
        obs = env.set_init_state(initial_states[episode_idx])

        # Setup
        t = 0
        replay_images = []

        if cfg.task_suite_name == "libero_spatial":
            max_steps = 220  # longest training demo has 193 steps
        elif cfg.task_suite_name == "libero_object":
            max_steps = 280  # longest training demo has 254 steps
        elif cfg.task_suite_name == "libero_goal":
            max_steps = 300  # longest training demo has 270 steps
        elif cfg.task_suite_name == "libero_10":
            max_steps = 520  # longest training demo has 505 steps
        elif cfg.task_suite_name == "libero_90":
            max_steps = 400  # longest training demo has 373 steps

        # print(f"Starting episode {task_episodes+1}...")
        # log_file.write(f"Starting episode {task_episodes+1}...\n")
        while t < max_steps + cfg.num_steps_wait:
            try:
                # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                # and we need to wait for them to fall
                if t < cfg.num_steps_wait:
                    obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
                    t += 1
                    continue

                # Get preprocessed image
                img = get_libero_image(obs, resize_size)

                if is_img_perturb and img_perturb_func is not None:
                    img = img_perturb_func(img,method=cfg.perturb_image_method, severity=cfg.perturb_image_severity)

                # Save preprocessed image for replay video
                if cfg.is_save_video:
                    replay_images.append(img)

                # Prepare observations dict
                # Note: OpenVLA does not take proprio state as input
                observation = {
                    "full_image": img,
                    "state": np.concatenate(
                        (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
                    ),
                }

                # Query model to get action
                action = get_action(
                    cfg,
                    model,
                    observation,
                    task_description,
                    processor=processor,
                )

                # Normalize gripper action [0,1] -> [-1,+1] because the environment expects the latter
                action = normalize_gripper_action(action, binarize=True)

                # [OpenVLA] The dataloader flips the sign of the gripper action to align with other datasets
                # (0 = close, 1 = open), so flip it back (-1 = open, +1 = close) before executing the action
                if cfg.model_family == "openvla":
                    action = invert_gripper_action(action)

                # Execute action in environment
                obs, reward, done, info = env.step(action.tolist())
                if done:
                    task_successes += 1
                    # total_successes += 1
                    break
                t += 1

            except Exception as e:
                print(f"Caught exception: {e}")
                # log_file.write(f"Caught exception: {e}\n")
                break

        task_episodes += 1
        total_episodes += 1

        # Save a replay video of the episode

        if cfg.is_save_video and replay_images:

            save_rollout_video(cfg.save_folder,replay_images, total_episodes, success=done, task_description=task_description)

        # Log current results
        print(f"Success: {done}")
        print(f"# episodes completed so far: {total_episodes}")
        # print(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")
        # log_file.write(f"Success: {done}\n")
        # log_file.write(f"# episodes completed so far: {total_episodes}\n")
        # # log_file.write(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)\n")
        # log_file.flush()
        


    # Log final results
    print(f"\nCurrent task success rate: {float(task_successes) / float(task_episodes)}")
    # print(f"Current total success rate: {float(total_successes) / float(total_episodes)}")
    # log_file.write(f"Current task success rate: {float(task_successes) / float(task_episodes)}\n")
    # # log_file.write(f"Current total success rate: {float(total_successes) / float(total_episodes)}\n")
    # log_file.flush()
    # if cfg.use_wandb:
    #     wandb.log(
    #         {
    #             f"success_rate/{task_description}": float(task_successes) / float(task_episodes),
    #             f"num_episodes/{task_description}": task_episodes,
    #         }
    #     )

    # Save local log file
    # log_file.close()


    env.close()
    del env
    torch.cuda.synchronize()
    gc.collect(); torch.cuda.empty_cache()

    print(f" \n----------libero environment closed---------- \n")

    return (task_episodes, task_successes)


def load_relate_model(cfg):

    device_model = "cuda"  # 绑定到“可见列表索引0”，也就是模型卡

    if cfg.semantic_type == "clip":

        semantic_model = CLIPEmbeddingModel(device=device_model)
    else:
        semantic_model = pipeline("text-classification", model="MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli")   # deberta-v3-large-mnli    roberta-large-mnli
        # stop = set(stopwords.words("english")) | set(string.punctuation)

    # 2) 选择生成后端：'smolvlm' 或 'qwenvl'

    print(f"[ERT] Using backend: {cfg.BACKEND}")

    if cfg.BACKEND == "smolvlm":
        red_team = build_red_team_generator(
            backend="smolvlm",
            embedding_model=semantic_model,
            model_path="HuggingFaceTB/SmolVLM-Instruct",  # "HuggingFaceTB/SmolVLM-500M-Instruct"。HuggingFaceTB/SmolVLM-Instruct
            custom_lora_path=cfg.custom_lora_path,  # 如需可加
            device=device_model,
        )
    else:
        # Qwen-VL：本地或 API
        # - 本地：model="Qwen/Qwen2.5-VL-7B-Instruct", mode="local"
        # - API ：model="qwen2.5-vl-72b-instruct",  mode="api"（需 export DASHSCOPE_API_KEY=...）
        red_team = build_red_team_generator(
            backend="qwenvl",
            embedding_model=semantic_model,
            mode=cfg.qwen_mode,
            model=cfg.qwen_model_id,
            device=device_model,
        )
    
    openvla_model = get_model(cfg).eval().to(device_model)
    
    return red_team,openvla_model



def plot_similarity_distribution(scores: List[float], save_path: Union[str, Path], bins: int = 20, title: Optional[str] = None) -> Path:
    """Compute summary stats and plot a histogram of similarity scores.


    Args:
    scores: 相似度分数列表（0~1）
    save_path: PNG 输出路径
    bins: 直方图分箱
    title: 图标题（可选）


    Returns:
    Path to the saved PNG.
    """
    import matplotlib.pyplot as plt


    scores = np.asarray(scores, dtype=float)
    mean_val = float(np.mean(scores)) if scores.size > 0 else float("nan")
    var_val = float(np.var(scores)) if scores.size > 0 else float("nan")


    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)


    plt.figure(figsize=(6, 4))
    plt.hist(scores, bins=bins)
    plt.xlabel("Similarity score")
    plt.ylabel("Count")
    t = title or "Instruction Similarity Distribution"
    plt.title(f"{t}\nMean={mean_val:.4f}, Var={var_val:.4f}")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


    print(f"[Similarity] Mean={mean_val:.4f}, Var={var_val:.4f}, saved plot -> {save_path}")
    return save_path


def parse_task_and_links(cfg,task_to_links):

    task_suite_name = cfg.task_suite_name
    task_suite = task_to_links[task_suite_name]
    task_language_list = task_suite.keys()

    return task_language_list, task_suite