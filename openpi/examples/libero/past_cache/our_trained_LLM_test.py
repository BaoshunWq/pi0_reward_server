import collections
import dataclasses
import logging
import math
import pathlib
import os
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
import gc  # 添加垃圾回收
from utils import load_relate_model,parse_task_and_links
import wandb

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data
from datetime import datetime

# 获取当前日期和时间
now = datetime.now()

# 格式化为 "年-月-日 时:分:秒"
formatted_time = now.strftime('%Y-%m-%d_%H:%M:%S')





@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = (
        "libero_spatial"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    )
    num_steps_wait: int = 50  # Number of steps to wait for objects to stabilize i n sim
    num_trials_per_task: int = 10  # Number of rollouts per task

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = f"output/{formatted_time}libero/videos"  # Path to save videos
    save_videos: bool = False  # Whether to save video replays (disable to save memory)
    
    seed: int = 7  # Random Seed (for reproducibility)
    custom_lora_path: str = ""  # smolvlm custom lora path
    qwen_mode: str = "api"  # qwenvl mode: "local" or "api"
    qwen_model_id : str = "Qwen/Qwen2-VL-2B-Instruct"
    semantic_type: str = "clip"  # "clip" or "deberta"
    num_instructions: int = 10  # number of generated instructions
    select_topk: int = 5  # select top k instructions
    # task_to_huglinks_json_path: str = "/home/baoshuntong/code/saftyEmbodyAI/redTeam_pi0/openpi/libero-init-frames/json_data_for_rl/vlm_initial_state_links_new.json"  # path to task to huglinks json
    BACKEND: str = "verl_qwen"  # "smolvlm" or "qwenvl","verl_qwen"
    output_path: str = f"data/{formatted_time}libero/results/libero_vlmrewrite_results.json"  # path to save results
    whole_acc_log_path: str = f"data/{formatted_time}libero/results/libero_whole_acc_log.json"  # path to save whole accuracy log
    failure_threshold: int = 5
    verl_model_path: str = "/root/autodl-tmp/trained_models/rover-coundown/countdown-rover/actor/global_step_200"  # path to verl qwen model
    wandb_project: str = "ourTrainedLLM_libero_evaluation_openpi"  # wandb project name
    wandb_entity: str = "tongbs-sysu"  # wandb entity (username or team name), leave empty for default


def eval_libero(args: Args) -> None:
    # Set random seed
    np.random.seed(args.seed)
    
    # Initialize wandb
    wandb_config = {
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
    }
    
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity if args.wandb_entity else None,
        config=wandb_config,
        name=f"{args.task_suite_name}_{formatted_time}",
    )

    red_team = load_relate_model(args)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}")

    # 只在需要保存视频时创建目录
    if args.save_videos:
        pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)
        logging.info(f"Video saving enabled. Videos will be saved to: {args.video_out_path}")
    else:
        logging.info("Video saving disabled to save memory")

    if args.task_suite_name == "libero_spatial":
        max_steps = 220  # longest training demo has 193 steps
    elif args.task_suite_name == "libero_object":
        max_steps = 280  # longest training demo has 254 steps
    elif args.task_suite_name == "libero_goal":
        max_steps = 300  # longest training demo has 270 steps
    elif args.task_suite_name == "libero_10":
        max_steps = 520  # longest training demo has 505 steps
    elif args.task_suite_name == "libero_90":
        max_steps = 400  # longest training demo has 373 steps
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    # 初始化WebSocket客户端
    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    annotations_smi_all = []

    results = []  # task -> list of dicts per round / per candidate
    # Start evaluation
    total_episodes, total_successes = 0, 0
    
    # 使用try-finally确保资源清理
    try:
        for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
            # Get task
            task = task_suite.get_task(task_id)

            # Get default LIBERO initial states
            initial_states = task_suite.get_task_init_states(task_id)

            # Initialize LIBERO environment and task description
            env = None  # 初始化为None便于清理
            
            try:
                env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)
                image_url = ""

                annotations,annotations_smi = red_team(task_description,semantic_type= args.semantic_type, image_url=image_url,
                                                       num_instructions=args.num_instructions,select_topk=args.select_topk)

                annotations_smi_all.extend(annotations_smi)
                annotations_smi = [float(x) for x in annotations_smi]  # 去掉 np.float32

                done_anotations = []
                fail_anotations = []
                done_annotations_smi = []
                fail_annotations_smi = []

                current_task_result = {}

                for i,(annotation,annotation_sim) in enumerate(zip(annotations,annotations_smi)):

                    print(f"evaluation task description:{annotation}, It's similarity :{annotation_sim}")
                    # Start episodes
                    task_episodes, task_successes = 0, 0
                    instruction_step = task_id * len(annotations) + i  # 全局步骤计数器

                    for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
                        logging.info(f"\nTask: {task_description}")
                        logging.info(f"evaluation task description:{annotation}, It's similarity :{annotation_sim}")

                        # Reset environment
                        env.reset()
                        action_plan = collections.deque()

                        # Set initial states
                        obs = env.set_init_state(initial_states[episode_idx])

                        # Setup
                        t = 0
                        replay_images = []
                        done = False  # 初始化done变量

                        logging.info(f"Starting episode {task_episodes+1}...")
                        
                        try:
                            while t < max_steps + args.num_steps_wait:
                                # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                                # and we need to wait for them to fall
                                if t < args.num_steps_wait:
                                    obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                                    t += 1
                                    continue

                                # Get preprocessed image
                                # IMPORTANT: rotate 180 degrees to match train preprocessing
                                img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                                wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                                img = image_tools.convert_to_uint8(
                                    image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
                                )
                                wrist_img = image_tools.convert_to_uint8(
                                    image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                                )

                                # Save preprocessed image for replay video (only if enabled)
                                if args.save_videos:
                                    replay_images.append(img)

                                if not action_plan:
                                    # Finished executing previous action chunk -- compute new chunk
                                    # Prepare observations dict
                                    element = {
                                        "observation/image": img,
                                        "observation/wrist_image": wrist_img,
                                        "observation/state": np.concatenate(
                                            (
                                                obs["robot0_eef_pos"],
                                                _quat2axisangle(obs["robot0_eef_quat"]),
                                                obs["robot0_gripper_qpos"],
                                            )
                                        ),
                                        "prompt": str(annotation),
                                    }

                                    # Query model to get action
                                    action_chunk = client.infer(element)["actions"]
                                    assert (
                                        len(action_chunk) >= args.replan_steps
                                    ), f"We want to replan every {args.replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
                                    action_plan.extend(action_chunk[: args.replan_steps])

                                action = action_plan.popleft()

                                # Execute action in environment
                                obs, reward, done, info = env.step(action.tolist())
                                if done:
                                    task_successes += 1
                                    total_successes += 1
                                    break
                                t += 1
                        
                        except Exception as e:
                            logging.error(f"Caught exception in episode {episode_idx}: {e}")
                            done = False
                        
                        finally:
                            # CRITICAL: 立即处理episode结束逻辑
                            task_episodes += 1
                            total_episodes += 1

                            # Log current results
                            logging.info(f"Success: {done}")
                            logging.info(f"# episodes completed so far: {total_episodes}")
                            logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")
                            
                            # CRITICAL: 显式删除replay_images释放内存
                            if replay_images:
                                del replay_images
                    
                    # Save a replay video after all episodes (only if enabled and last episode has images)
                    # 注意：这里需要在循环外单独保存最后一个episode的视频
                    # 计算当前指令的成功率
                    instruction_success_rate = float(task_successes) / float(task_episodes) if task_episodes > 0 else 0.0
                    
                    # 记录每个指令的结果到wandb
                    wandb.log({
                        "instruction_step": instruction_step,
                        "task_id": task_id,
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
                    
                    if task_successes <= args.failure_threshold:
                        fail_anotations.append(annotation)
                        fail_annotations_smi.append(annotations_smi[i])
                    else:
                        done_anotations.append(annotation)
                        done_annotations_smi.append(annotations_smi[i])
                    
                    current_task_result['image_url'] = image_url
                    current_task_result['done_anotations'] = done_anotations
                    current_task_result['fail_anotations'] = fail_anotations
                    current_task_result['donw_annotations_smi'] = done_annotations_smi
                    current_task_result['fail_annotations_smi'] = fail_annotations_smi
                    current_task_result['task'] = task_description

                    results.append(current_task_result)
                    
                    # save intermediate results to disk after each task
                    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
                    with open(args.output_path, "w") as fout:
                        json.dump(results, fout,indent=4)
                
                # Log final results
                logging.info(f"Current task success rate: {float(task_successes) / float(task_episodes) if task_episodes > 0 else 0.0}")
                logging.info(f"Current total success rate: {float(total_successes) / float(total_episodes) if total_episodes > 0 else 0.0}")
            
            finally:
                # CRITICAL: 清理环境资源
                if env is not None:
                    try:
                        env.close()
                        logging.info(f"Environment for task {task_id} closed successfully")
                    except Exception as e:
                        logging.error(f"Failed to close environment for task {task_id}: {e}")
                    finally:
                        del env
                
                # 强制垃圾回收
                gc.collect()

        # 计算最终统计
        if annotations_smi_all:
            mean_val = np.mean(annotations_smi_all)
            var_val = np.var(annotations_smi_all)
            print(f" Overall task similarity  Mean: {mean_val:.4f}, Variance: {var_val:.4f}")
        else:
            mean_val = 0.0
            var_val = 0.0

        all_result = {}
        all_success_rate = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0.0
        all_result[args.task_suite_name] = all_success_rate

        # 记录最终汇总信息到wandb
        wandb.log({
            "final/overall_success_rate": all_success_rate,
            "final/total_successes": total_successes,
            "final/total_episodes": total_episodes,
            "final/mean_similarity": mean_val,
            "final/variance_similarity": var_val,
        })
        
        # 创建汇总表格
        summary_table = wandb.Table(columns=["Metric", "Value"])
        summary_table.add_data("Overall Success Rate", f"{all_success_rate:.4f}")
        summary_table.add_data("Total Successes", total_successes)
        summary_table.add_data("Total Episodes", total_episodes)
        summary_table.add_data("Mean Similarity", f"{mean_val:.4f}")
        summary_table.add_data("Variance Similarity", f"{var_val:.4f}")
        wandb.log({"final/summary_table": summary_table})

        os.makedirs(os.path.dirname(args.whole_acc_log_path), exist_ok=True)
        with open(args.whole_acc_log_path, "a") as fout:
            json.dump(all_result, fout,indent=4)

        print(f"Done. Results saved to {args.output_path}")

        logging.info(f"Total success rate: {all_success_rate:.4f}")
        logging.info(f"Total episodes: {total_episodes}")
        
        # 完成wandb运行
        wandb.finish()
    
    finally:
        # CRITICAL: 最终清理
        # 关闭WebSocket客户端
        try:
            if hasattr(client, '_ws') and client._ws is not None:
                client._ws.close()
                logging.info("WebSocket client closed successfully")
        except Exception as e:
            logging.error(f"Failed to close WebSocket client: {e}")
        
        # 清理其他对象
        if 'client' in locals():
            del client
        if 'task_suite' in locals():
            del task_suite
        if 'red_team' in locals():
            del red_team
        
        # 最终垃圾回收
        gc.collect()
        logging.info("All resources cleaned up")


def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_libero)
    # args = Args()
    # eval_libero(args)
