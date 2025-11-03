import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, "/home/baoshuntong/code/saftyEmbodyAI/embodied-red-teaming/LIBERO")

# # ==== GPU 绑核设置（务必在导入 torch/mujoco 等大库之前）====
# MODEL_GPU_ID = 6   # 模型跑的物理GPU编号（nvidia-smi里看到的）
# EGL_GPU_ID   = 7   # 渲染用的物理GPU编号（nvidia-smi里看到的）

# # 让当前进程“只看见”两张目标卡，并且固定顺序：先模型卡，再渲染卡
# # 这样 MUJOCO_EGL_DEVICE_ID=1 就总能指向第二张（也就是渲染卡）
# os.environ["CUDA_DEVICE_ORDER"]   = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = f"{MODEL_GPU_ID},{EGL_GPU_ID}"

# # 告诉 MuJoCo 用 EGL 后端，并指定它在“可见设备列表”里的索引
# # 由于我们把可见顺序固定成 [模型卡, 渲染卡]，所以渲染卡永远是索引 1
# os.environ["MUJOCO_GL"] = "egl"
# os.environ["PYOPENGL_PLATFORM"] = "egl"
# os.environ["MUJOCO_EGL_DEVICE_ID"] = "7"  # 这里是“本进程可见GPU列表”的索引，不是物理号

# # 某些环境下（例如 Mesa EGL）需要这个变量；NVIDIA 驱动下留着也无害，作为兜底
# os.environ.setdefault("EGL_VISIBLE_DEVICES", "7")

# # 避免 PyTorch 过度碎片化导致奇怪崩溃（可选）
# os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


import numpy as np
# import warnings
# warnings.filterwarnings("ignore", category=RuntimeWarning)

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


# 获取当前时间
now = datetime.now()
NOW_TIME_STR = now.strftime("%Y-%m-%d_%H-%M-%S")


from utils import eval_libero,load_relate_model,plot_similarity_distribution,parse_task_and_links
# from generate_instructions import EmbodiedRedTeamModelWithQwen2, CLIPEmbeddingModel


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
    
    num_trials_per_task: int = 1                    # Number of rollouts per task

    output_root_dir: str = f"./output/{NOW_TIME_STR}/{task_suite_name}"  # Root directory to save results

    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note: Optional[str] = None                # Extra note to add in run ID for logging
    local_log_dir: str = f"{output_root_dir}/logs"        # Local directory for eval logs

    use_wandb: bool = False                          # Whether to also log results in Weights & Biases
    wandb_project: str = "redTeamingOpenvla"        # Name of W&B project to log to (use default!)
    wandb_entity: str = "tongbs-sysu"          # Name of entity to log under

    seed: int = 7                                    # Random Seed (for reproducibility)

    # fmt: on
    # user-customized config

    is_save_video: bool = False                       # Whether to save rollout video

    save_folder: str = f"{output_root_dir}/videos"          # Folder to save rollout videos

    task_to_huglinks_json_path: str = "/home/baoshuntong/code/saftyEmbodyAI/embodied-red-teaming/libero-init-frames/json_data_for_rl/vlm_initial_state_links_new.json"  # JSON file mapping task names to image URLs

    examples_path: str = ""                          # YAML file with existing task->examples (optional)

    redTeaming_vlm_model = "HuggingFaceTB/SmolVLM-500M-Instruct"               # VLM model for instruction generation (e.g. gpt-4, gpt-3.5-turbo)
    # . #HuggingFaceTB/SmolVLM-Instruct


    failure_threshold: float = 0.5               # Success rate <= threshold considered a failure

    output_path: str = f"{output_root_dir}/redteaming_results.json"  # Path to save the attack results

    num_instructions : int = 5                     # Number of instructions to request per generation

    select_topk: int = 3                             # Select top-k instructions from the generated candidates

    prefer_prompt_key : str = ""          # Prefer prompt key for instruction generation, options: FRAME, DESCRIBE, GENERAL


    whole_acc_log_path : str = f"{output_root_dir}/logs/whole_acc.log"


    is_red_teaming_attack : bool = True

    # custom_lora_path: Optional[str] = "outputs/2025-10-21_15-03-36/verl_dpo_vlm/lora_dpo"        # Path to custom LoRA adapter for instruction generation model
    custom_lora_path: Optional[str] = ""        # Path to custom LoRA adapter for instruction generation model

    BACKEND = "qwenvl"  # 'smolvlm' 或 'qwenvl' 

    simiarity_image_path : str = f"{output_root_dir}/similarity_distribution.png"

    qwen_mode: str = "api"   # local or api

    qwen_model_id: str = "Qwen/Qwen2-VL-2B-Instruct"  # Qwen-VL-Chat-7B or Qwen-VL-Chat-14B

    semantic_type: str = "clip"   # clip or entail




# -------------- 主循环 --------------

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

    red_team,openvla_model = load_relate_model(cfg)


    task_language_list, task_links_suite = parse_task_and_links(cfg,task_to_links)

    # benchmark_dict = benchmark.get_benchmark_dict()
    # num_tasks_in_suite = task_suite.n_tasks
    print(f"Task suite: {cfg.task_suite_name}")

    suite_task_num = 0
    suite_task_success = 0

    annotations_smi_all = []

    results = []  # task -> list of dicts per round / per candidate

    # cfg.save_folder = f"{cfg.task_suite_name}/{cfg.perturb_image_method}/{cfg.prefer_prompt_key}"

    for task_id, task_language in enumerate(task_language_list):

        instruction = task_language.replace("_", " ")

        print(f"\ntask {task_id+1} :  {instruction}")

        current_task_links = task_links_suite[task_language]

        image_url = random.choice(current_task_links)

        # generate instruction group (EmbodiedRedTeamModel handles rejection sampling)
        if cfg.is_red_teaming_attack:

            annotations,annotations_smi = red_team(instruction,semantic_type= cfg.semantic_type, image_url=image_url,num_instructions=cfg.num_instructions,select_topk=cfg.select_topk)
        else:
            annotations = [instruction]
        
        annotations_smi_all.extend(annotations_smi)
        annotations_smi = [float(x) for x in annotations_smi]  # 去掉 np.float32
        
        done_anotations = []
        fail_anotations = []
        done_annotations_smi = []
        fail_annotations_smi = []

        current_task_result = {}


        for i,annotation in enumerate(tqdm(annotations,desc="Annotations", total=len(annotations),
            dynamic_ncols=True, disable=False, file=sys.stdout)):


            annotation = re.sub(r'^\s*\d+\.\s*', '', annotation).strip()

            print(f"\nRed team generated instruction: {annotation}.           It's similarity score is {annotations_smi[i]:.2f}")

            img_perturb_func = None


            # if cfg.is_img_perturb:
            #     img_perturb_func = perturb_image
            #     print(" eval under image perturb")

            task_episodes, task_successes = eval_libero(cfg,openvla_model,task_id,instruction,)

            suite_task_success += task_successes
            suite_task_num += task_episodes

            success_rate = task_successes / task_episodes

            if success_rate <= cfg.failure_threshold:
                fail_anotations.append(annotation)
                fail_annotations_smi.append(annotations_smi[i])
            else:
                done_anotations.append(annotation)
                done_annotations_smi.append(annotations_smi[i])
        print(f" \n=============== total success rate : {suite_task_success / suite_task_num}")

        # finally:
            # if env is not None:
            #     try: env.close()
            #     except Exception as e: print(f"[WARN] env.close() failed: {e}")
            # del env
            # torch.cuda.synchronize()
            # gc.collect(); torch.cuda.empty_cache()
        
        

        

        current_task_result['image_url'] = image_url
        current_task_result['done_anotations'] = done_anotations
        current_task_result['fail_anotations'] = fail_anotations
        current_task_result['donw_annotations_smi'] = done_annotations_smi
        current_task_result['fail_annotations_smi'] = fail_annotations_smi
        current_task_result['task'] = instruction

        results.append(current_task_result)
        # save intermediate results to disk after each task
        os.makedirs(os.path.dirname(cfg.output_path), exist_ok=True)
        with open(cfg.output_path, "w") as fout:
            json.dump(results, fout,indent=4)
        
        
    
    all_success_rate = suite_task_success / suite_task_num
    
    print(f"Overall success rate on {suite_task_num} episodes: {all_success_rate:.3f}")


    mean_val = np.mean(annotations_smi_all)
    var_val = np.var(annotations_smi_all)

    print(f" Overall task similarity  Mean: {mean_val:.4f}, Variance: {var_val:.4f}")

    curr_prefer_prompt_key_result = {}

    curr_prefer_prompt_key_result[cfg.task_suite_name] = {}

    curr_prefer_prompt_key_result[cfg.task_suite_name][cfg.prefer_prompt_key] = all_success_rate

    os.makedirs(os.path.dirname(cfg.whole_acc_log_path), exist_ok=True)
    with open(cfg.whole_acc_log_path, "a") as fout:
        json.dump(curr_prefer_prompt_key_result, fout,indent=4)

    print(f"Done. Results saved to {cfg.output_path}")

    plot_similarity_distribution(annotations_smi_all,cfg.simiarity_image_path,)





    # import torch
    # torch.cuda.set_device(2)
if __name__ == "__main__":

    cfg = GenerateConfig()

    run_ert_loop(cfg=cfg)



