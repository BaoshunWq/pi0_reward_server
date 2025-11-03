#!/usr/bin/env python3
# main.py
"""
ERT main loop:
- 加载任务->image links
- 调用 generate_instructions.EmbodiedRedTeamModel 生成 candidate instructions
- 在 OpenVLA + LIBERO 上评估每条 instruction 的 success_rate
- 选失败示例反馈给生成器，迭代若干轮
- 输出最终结果到 YAML/JSON
"""
import torch.multiprocessing as mp
mp.set_start_method("spawn", force=True)
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, "/home/baoshuntong/code/saftyEmbodyAI/embodied-red-teaming/LIBERO")
os.environ["CUDA_VISIBLE_DEVICES"] = '7'
# os.environ.setdefault("CUDA_VISIBLE_DEVICES", ["0","1"])   # 绑定单卡（按需改）
os.environ.setdefault("MUJOCO_GL", "egl")            # 指定用 EGL（headless）
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("EGL_DEVICE_ID", "1")          # 绑定到同一张卡


BACKEND = "qwenvl"

# import warnings
# warnings.filterwarnings("ignore", category=RuntimeWarning)
import os
import yaml
import json
import random
from tqdm import tqdm
from dataclasses import dataclass
import draccus
from typing import Optional, Union
from pathlib import Path
import re
from datetime import datetime
import torch
import gc

# 获取当前时间
now = datetime.now()
NOW_TIME_STR = now.strftime("%Y-%m-%d_%H-%M-%S")

from LIBERO.libero.libero import benchmark
from utils import eval_libero
# from generate_instructions import EmbodiedRedTeamModelWithQwen2, CLIPEmbeddingModel
from generate_intru_smolvlm import build_red_team_generator, CLIPEmbeddingModel
from openvla.experiments.robot.openvla_utils import get_processor
from openvla.experiments.robot.robot_utils import get_model
from openvla.experiments.robot.libero.libero_utils import get_libero_env
# from cache_py.image_perturb import perturb_image

@dataclass
class GenerateConfig:
    # fmt: off

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "openvla"                    # Model family
    pretrained_checkpoint: Union[str, Path] = "openvla/openvla-7b-finetuned-libero-spatial"     # Pretrained checkpoint path
    load_in_8bit: bool = False                       # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                       # (For OpenVLA only) Load with 4-bit quantization

    center_crop: bool = True                         # Center crop? (if trained w/ random crop image aug)

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = "libero_spatial"          # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    num_steps_wait: int = 10                         # Number of steps to wait for objects to stabilize in sim
    
    num_trials_per_task: int = 3                    # Number of rollouts per task

    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note: Optional[str] = None                # Extra note to add in run ID for logging
    local_log_dir: str = "./experiments/logs"        # Local directory for eval logs

    use_wandb: bool = False                          # Whether to also log results in Weights & Biases
    wandb_project: str = "redTeamingOpenvla"        # Name of W&B project to log to (use default!)
    wandb_entity: str = "tongbs-sysu"          # Name of entity to log under

    seed: int = 7                                    # Random Seed (for reproducibility)

    # fmt: on
    # user-customized config

    is_save_video: bool = False                       # Whether to save rollout video

    save_folder: str = "./output_videos/"          # Folder to save rollout videos

    task_to_huglinks_json_path: str = "/home/baoshuntong/code/saftyEmbodyAI/embodied-red-teaming/libero-init-frames/json_data_for_rl/vlm_initial_state_links_new.json"  # JSON file mapping task names to image URLs

    examples_path: str = ""                          # YAML file with existing task->examples (optional)

    redTeaming_vlm_model = "HuggingFaceTB/SmolVLM-500M-Instruct"               # VLM model for instruction generation (e.g. gpt-4, gpt-3.5-turbo)
    # . #HuggingFaceTB/SmolVLM-Instruct


    failure_threshold: float = 0.5               # Success rate <= threshold considered a failure

    output_path: str = f"./output/{NOW_TIME_STR}/redteaming_results.json"  # Path to save the attack results

    num_instructions : int = 5                     # Number of instructions to request per generation

    select_topk: int = 3                             # Select top-k instructions from the generated candidates

    prefer_prompt_key : str = "PRONOUN"          # Prefer prompt key for instruction generation, options: FRAME, DESCRIBE, GENERAL


    whole_acc_log_path : str = "/home/baoshuntong/code/saftyEmbodyAI/embodied-red-teaming/logs/whole_acc.log"


    is_red_teaming_attack : bool = True

    custom_lora_path: Optional[str] = "/home/baoshuntong/code/saftyEmbodyAI/embodied-red-teaming/outputs/2025-10-22_11-43-33/verl_kto_vlm/lora_kto"        # Path to custom LoRA adapter for instruction generation model


def parse_task_and_links(cfg,task_to_links):

    task_suite_name = cfg.task_suite_name
    task_suite = task_to_links[task_suite_name]
    task_language_list = task_suite.keys()

    return task_language_list, task_suite


# -------------- 主循环 --------------
@draccus.wrap()
def run_ert_loop(cfg: GenerateConfig):

    if cfg.task_suite_name == "libero_spatial":
        cfg.pretrained_checkpoint = "openvla/openvla-7b-finetuned-libero-spatial"
    elif cfg.task_suite_name == "libero_object":
        cfg.pretrained_checkpoint = "openvla/openvla-7b-finetuned-libero-object"
    elif cfg.task_suite_name == "libero_goal":
        cfg.pretrained_checkpoint = "openvla/openvla-7b-finetuned-libero-goal"
    elif cfg.task_suite_name == "libero_10":
        cfg.pretrained_checkpoint = "openvla/openvla-7b-finetuned-libero-10"
    else:
        raise Exception


    # load task images
    with open(cfg.task_to_huglinks_json_path, "r") as f:
        task_to_links = json.load(f)  # {task: [image_url, ...]}
    
            # [OpenVLA] Get Hugging Face processor
    processor = None
    if cfg.model_family == "openvla":
        processor = get_processor(cfg)
    
    openvla_model = get_model(cfg).eval().to("cuda")
    

    clip_embedder = CLIPEmbeddingModel(device="cuda")

    # 2) 选择生成后端：'smolvlm' 或 'qwenvl'

    print(f"[ERT] Using backend: {BACKEND}")

    if BACKEND == "smolvlm":
        red_team = build_red_team_generator(
            backend="smolvlm",
            embedding_model=clip_embedder,
            model_path="HuggingFaceTB/SmolVLM-Instruct",
            # custom_lora_path="...",  # 如需可加
            device="cuda",
        )
    else:
        # Qwen-VL：本地或 API
        # - 本地：model="Qwen/Qwen2.5-VL-7B-Instruct", mode="local"
        # - API ：model="qwen2.5-vl-72b-instruct",  mode="api"（需 export DASHSCOPE_API_KEY=...）
        red_team = build_red_team_generator(
            backend="qwenvl",
            embedding_model=clip_embedder,
            mode=os.getenv("QWEN_MODE", "local"),
            model=os.getenv("QWEN_MODEL", "Qwen/Qwen2-VL-2B-Instruct"),
            device="cuda",
        )

    task_language_list, task_links_suite = parse_task_and_links(cfg,task_to_links)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    # num_tasks_in_suite = task_suite.n_tasks
    print(f"Task suite: {cfg.task_suite_name}")

    suite_task_num = 0
    suite_task_success = 0

    results = []  # task -> list of dicts per round / per candidate

    # cfg.save_folder = f"{cfg.task_suite_name}/{cfg.perturb_image_method}/{cfg.prefer_prompt_key}"

    for task_id, task_language in enumerate(task_language_list):

        instruction = task_language.replace("_", " ")

        print(f"\ntask {task_id+1} :  {instruction}")

        task = task_suite.get_task(task_id)
        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)
        # Initialize LIBERO environment and task description
        env, _ = get_libero_env(task, cfg.model_family, resolution=256)


        current_task_links = task_links_suite[task_language]

        image_url = random.choice(current_task_links)

        # generate instruction group (EmbodiedRedTeamModel handles rejection sampling)
        if cfg.is_red_teaming_attack:

            annotations, = red_team(instruction, image_url=image_url, prefer_prompt_key=cfg.prefer_prompt_key,
                                num_instructions=cfg.num_instructions,select_topk=cfg.select_topk)
        else:
            annotations = [instruction]
        
        done_anotations = []
        fail_anotations = []

        current_task_result = {}


        for i,annotation in enumerate(tqdm(annotations,desc="Annotations", total=len(annotations),
            dynamic_ncols=True, disable=False, file=sys.stdout)):


            annotation = re.sub(r'^\s*\d+\.\s*', '', annotation).strip()

            print(f"Red team generated instruction: {annotation} \n")

            img_perturb_func = None


            # if cfg.is_img_perturb:
            #     img_perturb_func = perturb_image
            #     print(" eval under image perturb")

            task_episodes, task_successes = eval_libero(cfg, openvla_model,processor,env,initial_states,annotation,)

            suite_task_success += task_successes
            suite_task_num += task_episodes

            success_rate = task_successes / task_episodes

            if success_rate <= cfg.failure_threshold:
                fail_anotations.append(annotation)
            else:
                done_anotations.append(annotation)
            
        print(f" \n=============== total success rate : {suite_task_success / suite_task_num}")


        env.close()
        del env
        torch.cuda.synchronize()
        gc.collect(); torch.cuda.empty_cache()
        

        current_task_result['image_url'] = image_url
        current_task_result['done_anotations'] = done_anotations
        current_task_result['fail_anotations'] = fail_anotations
        current_task_result['task'] = instruction

        results.append(current_task_result)
        # save intermediate results to disk after each task
        os.makedirs(os.path.dirname(cfg.output_path), exist_ok=True)
        with open(cfg.output_path, "w") as fout:
            json.dump(results, fout,indent=4)
        
        
    
    all_success_rate = suite_task_success / suite_task_num
    
    print(f"Overall success rate on {suite_task_num} episodes: {all_success_rate:.3f}")

    curr_prefer_prompt_key_result = {}

    curr_prefer_prompt_key_result[cfg.task_suite_name] = {}

    curr_prefer_prompt_key_result[cfg.task_suite_name][cfg.prefer_prompt_key] = all_success_rate

    os.makedirs(os.path.dirname(cfg.whole_acc_log_path), exist_ok=True)
    with open(cfg.whole_acc_log_path, "a") as fout:
        json.dump(curr_prefer_prompt_key_result, fout,indent=4)

    print(f"Done. Results saved to {cfg.output_path}")




if __name__ == "__main__":

    # import torch
    # torch.cuda.set_device(2)
    run_ert_loop()
