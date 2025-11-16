"""
统一评估脚本 - 支持VLM和LLM两种模式
根据mode参数选择使用VLM rewrite还是直接使用训练的LLM
"""
import os
# 设置tokenizers并行处理环境变量，避免警告
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
import collections
import dataclasses
import logging
import math
import pathlib
import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm
import tyro
import json
import random
import gc
from utils import load_relate_model, parse_task_and_links
import wandb
from typing import Optional

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256
from datetime import datetime

now = datetime.now()
formatted_time = now.strftime('%Y-%m-%d_%H:%M:%S')


@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Mode selection
    #################################################################################################################
    mode: str = "vlm"  # "llm" for trained LLM, "vlm" for VLM rewrite
    
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "0.0.0.0"
    port: int = 3333
    resize_size: int = 224
    replan_steps: int = 5

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = "libero_spatial"
    num_steps_wait: int = 50
    num_trials_per_task: int = 2

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = f"output/{formatted_time}libero/videos"
    save_videos: bool = False  # Use --save-videos to enable
    
    seed: int = 7
    custom_lora_path: str = ""
    qwen_mode: str = "api"
    qwen_model_id: str = "Qwen/Qwen2-VL-2B-Instruct"
    semantic_type: str = "clip"
    num_instructions: int = 1
    select_topk: int = 5
    task_to_huglinks_json_path: str = "libero-init-frames/json_data_for_rl/vlm_initial_state_links_new.json"
    BACKEND: str = "verl_qwen"
    output_path: str = f"data/{formatted_time}libero/results/unified_eval_results.json"
    whole_acc_log_path: str = f"data/{formatted_time}libero/results/whole_acc_log.json"
    failure_threshold: int = 5
    verl_model_path: str = "/root/autodl-tmp/code/attackVLA/rover_verl/checkpoints/custom_rover_qwen2_5_vl_lora_20251113_121506/global_step_100"
    
    # WandB config
    use_wandb: bool = False  # Use --no-use-wandb to disable
    wandb_project: str = "unified_libero_evaluation"
    wandb_entity: str = "tongbs-sysu"


def eval_libero(args: Args) -> None:
    """统一评估入口函数"""
    # 验证mode参数
    if args.mode not in ["llm", "vlm"]:
        raise ValueError(f"Invalid mode: {args.mode}. Must be 'llm' or 'vlm'")
    
    logging.info(f"Running in {args.mode.upper()} mode")
    
    # Set random seed
    np.random.seed(args.seed)
    
    # Initialize wandb (如果启用)
    if args.use_wandb:
        wandb_config = {
            "mode": args.mode,
            "task_suite_name": args.task_suite_name,
            "num_trials_per_task": args.num_trials_per_task,
            "seed": args.seed,
            "backend": args.BACKEND,
            "semantic_type": args.semantic_type,
            "num_instructions": args.num_instructions,
            "select_topk": args.select_topk,
            "failure_threshold": args.failure_threshold,
            "resize_size": args.resize_size,
            "replan_steps": args.replan_steps,
            "save_videos": args.save_videos,
        }
        
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity if args.wandb_entity else None,
            config=wandb_config,
            name=f"{args.mode}_{args.task_suite_name}_{formatted_time}",
        )
        logging.info("WandB initialized")
    else:
        logging.info("WandB disabled")

    # 加载模型
    red_team = load_relate_model(args)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name} ({num_tasks_in_suite} tasks)")

    # 只在需要保存视频时创建目录
    if args.save_videos:
        pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)
        logging.info(f"Video saving enabled: {args.video_out_path}")
    else:
        logging.info("Video saving disabled to save memory")

    # 设置max_steps
    max_steps_dict = {
        "libero_spatial": 220,
        "libero_object": 280,
        "libero_goal": 300,
        "libero_10": 520,
        "libero_90": 400,
    }
    max_steps = max_steps_dict.get(args.task_suite_name)
    if max_steps is None:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    # 初始化WebSocket客户端
    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    # VLM模式需要加载task links
    task_links_suite = None
    if args.mode == "vlm":
        with open(args.task_to_huglinks_json_path, "r") as f:
            task_to_links = json.load(f)
        task_language_list, task_links_suite = parse_task_and_links(args.task_suite_name, task_to_links)
        logging.info("VLM mode: task links loaded")

    annotations_smi_all = []
    results = []
    total_episodes, total_successes = 0, 0
    
    # 使用try-finally确保资源清理
    try:
        for task_id in tqdm.tqdm(range(num_tasks_in_suite), desc="Tasks"):
            task = task_suite.get_task(task_id)
            initial_states = task_suite.get_task_init_states(task_id)
            
            env = None
            try:
                env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

                print(f"[Task {task_id}]: source task_description: {task_description}")
                
                # 根据模式选择image_url
                if args.mode == "vlm":
                    task_links_key = task_description.replace(" ", "_")
                    current_task_links = task_links_suite[task_links_key]
                    image_url = random.choice(current_task_links)
                else:  # llm模式
                    image_url = ""
                
                # 生成指令
                annotations, annotations_smi = red_team(task_description,semantic_type=args.semantic_type,image_url=image_url,
                    num_instructions=args.num_instructions,select_topk=args.select_topk)
                
                annotations_smi_all.extend(annotations_smi)
                annotations_smi = [float(x) for x in annotations_smi]
                
                done_annotations = []
                fail_annotations = []
                done_annotations_smi = []
                fail_annotations_smi = []
                
                current_task_result = {
                    "task_id": task_id,
                    "task_description": task_description,
                    "mode": args.mode,
                    "image_url": image_url,
                }
                
                # 对每个生成的指令进行评估
                for i, (annotation, annotation_sim) in enumerate(zip(annotations, annotations_smi)):
                    
                    print(f"[Task {task_id}] Evaluating: {annotation} (similarity: {annotation_sim:.4f})")
                    
                    task_episodes, task_successes = 0, 0
                    instruction_step = task_id * len(annotations) + i
                    
                    for episode_idx in tqdm.tqdm(range(args.num_trials_per_task), 
                                                  desc=f"Episodes for instruction {i+1}", 
                                                  leave=False):
                        env.reset()
                        action_plan = collections.deque()
                        obs = env.set_init_state(initial_states[episode_idx % len(initial_states)])
                        
                        t = 0
                        replay_images = []
                        done = False
                        
                        try:
                            while t < max_steps + args.num_steps_wait:
                                # 等待物体稳定
                                if t < args.num_steps_wait:
                                    obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                                    t += 1
                                    continue
                                
                                # 获取图像
                                img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                                wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                                img = image_tools.convert_to_uint8(
                                    image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
                                )
                                wrist_img = image_tools.convert_to_uint8(
                                    image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                                )
                                
                                if args.save_videos:
                                    replay_images.append(img)
                                
                                # 获取动作
                                if not action_plan:
                                    element = {
                                        "observation/image": img,
                                        "observation/wrist_image": wrist_img,
                                        "observation/state": np.concatenate((
                                            obs["robot0_eef_pos"],
                                            _quat2axisangle(obs["robot0_eef_quat"]),
                                            obs["robot0_gripper_qpos"],
                                        )),
                                        "prompt": str(annotation),
                                    }
                                    
                                    action_chunk = client.infer(element)["actions"]
                                    assert len(action_chunk) >= args.replan_steps
                                    action_plan.extend(action_chunk[:args.replan_steps])
                                
                                action = action_plan.popleft()
                                obs, reward, done, info = env.step(action.tolist())
                                
                                if done:
                                    task_successes += 1
                                    total_successes += 1
                                    break
                                t += 1
                        
                        except Exception as e:
                            logging.error(f"Episode {episode_idx} failed: {e}")
                            done = False
                        
                        finally:
                            task_episodes += 1
                            total_episodes += 1
                            
                            # 保存视频(如果启用)
                            if args.save_videos and replay_images:
                                try:
                                    suffix = "success" if done else "failure"
                                    safe_annotation = annotation.replace(" ", "_")[:60]
                                    video_name = f"{args.mode}_task{task_id}_inst{i}_ep{episode_idx}_{suffix}_{safe_annotation}.mp4"
                                    imageio.mimwrite(
                                        pathlib.Path(args.video_out_path) / video_name,
                                        [np.asarray(x) for x in replay_images],
                                        fps=10,
                                    )
                                except Exception as e:
                                    logging.error(f"Failed to save video: {e}")
                            
                            # 立即释放内存
                            if replay_images:
                                del replay_images
                            
                            print(f"Episode {task_episodes}: {'success' if done else 'failure'}")
                            logging.debug(f"Episode {task_episodes}: {'success' if done else 'failure'}")
                    
                    # 计算当前指令的成功率
                    instruction_success_rate = float(task_successes) / float(task_episodes) if task_episodes > 0 else 0.0
                    
                    # 记录到wandb
                    if args.use_wandb:
                        wandb.log({
                            "instruction_step": instruction_step,
                            "task_id": task_id,
                            "instruction_index": i,
                            "task_name": task_description,
                            "instruction": annotation,
                            "similarity": annotation_sim,
                            "instruction_success_rate": instruction_success_rate,
                            "instruction_successes": task_successes,
                            "instruction_episodes": task_episodes,
                            "cumulative_success_rate": float(total_successes) / float(total_episodes) if total_episodes > 0 else 0.0,
                            "total_successes": total_successes,
                            "total_episodes": total_episodes,
                        })
                    
                    # 分类成功/失败指令
                    if task_successes <= args.failure_threshold:
                        fail_annotations.append(annotation)
                        fail_annotations_smi.append(annotation_sim)
                    else:
                        done_annotations.append(annotation)
                        done_annotations_smi.append(annotation_sim)
                    
                    logging.info(f"Instruction {i+1}/{len(annotations)}: {instruction_success_rate:.2%} "
                               f"({task_successes}/{task_episodes})")
                
                # 保存当前任务结果
                current_task_result.update({
                    "done_annotations": done_annotations,
                    "fail_annotations": fail_annotations,
                    "done_annotations_smi": done_annotations_smi,
                    "fail_annotations_smi": fail_annotations_smi,
                })
                results.append(current_task_result)
                
                # 保存中间结果
                os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
                with open(args.output_path, "w") as fout:
                    json.dump(results, fout, indent=4)
                
                task_success_rate = float(task_successes) / float(task_episodes) if task_episodes > 0 else 0.0
                logging.info(f"Task {task_id} completed: {task_success_rate:.2%}")
            
            finally:
                # 清理环境
                if env is not None:
                    try:
                        env.close()
                        logging.debug(f"Environment for task {task_id} closed")
                    except Exception as e:
                        logging.error(f"Failed to close environment: {e}")
                    finally:
                        del env
                
                gc.collect()
        
        # 计算最终统计
        if annotations_smi_all:
            mean_val = np.mean(annotations_smi_all)
            var_val = np.var(annotations_smi_all)
            logging.info(f"Overall similarity - Mean: {mean_val:.4f}, Variance: {var_val:.4f}")
        else:
            mean_val = 0.0
            var_val = 0.0
        
        all_success_rate = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0.0
        
        # 最终结果
        final_result = {
            args.task_suite_name: all_success_rate,
            "mode": args.mode,
            "total_successes": total_successes,
            "total_episodes": total_episodes,
            "mean_similarity": mean_val,
            "variance_similarity": var_val,
        }
        
        # 记录到wandb
        if args.use_wandb:
            wandb.log({
                "final/overall_success_rate": all_success_rate,
                "final/total_successes": total_successes,
                "final/total_episodes": total_episodes,
                "final/mean_similarity": mean_val,
                "final/variance_similarity": var_val,
            })
            
            # 创建汇总表格
            summary_table = wandb.Table(columns=["Metric", "Value"])
            summary_table.add_data("Mode", args.mode.upper())
            summary_table.add_data("Overall Success Rate", f"{all_success_rate:.4f}")
            summary_table.add_data("Total Successes", total_successes)
            summary_table.add_data("Total Episodes", total_episodes)
            summary_table.add_data("Mean Similarity", f"{mean_val:.4f}")
            summary_table.add_data("Variance Similarity", f"{var_val:.4f}")
            wandb.log({"final/summary_table": summary_table})
            
            wandb.finish()
        
        # 保存最终日志
        os.makedirs(os.path.dirname(args.whole_acc_log_path), exist_ok=True)
        with open(args.whole_acc_log_path, "a") as fout:
            json.dump(final_result, fout, indent=4)
        
        logging.info(f"=== Evaluation Complete ===")
        logging.info(f"Mode: {args.mode.upper()}")
        logging.info(f"Overall Success Rate: {all_success_rate:.2%}")
        logging.info(f"Total Episodes: {total_episodes}")
        logging.info(f"Results saved to: {args.output_path}")
    
    finally:
        # 最终清理
        try:
            if hasattr(client, '_ws') and client._ws is not None:
                client._ws.close()
                logging.debug("WebSocket client closed")
        except Exception as e:
            logging.error(f"Failed to close WebSocket client: {e}")
        
        # 清理对象
        if 'client' in locals():
            del client
        if 'task_suite' in locals():
            del task_suite
        if 'red_team' in locals():
            del red_team
        
        gc.collect()
        logging.info("All resources cleaned up")


def _get_libero_env(task, resolution, seed):
    """初始化LIBERO环境"""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task_description


def _quat2axisangle(quat):
    """四元数转轴角"""
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    args = tyro.cli(Args)
    eval_libero(args)

