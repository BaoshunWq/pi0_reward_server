"""
支持并行的评估模块 - 使用客户端连接池
"""
import collections
import dataclasses
import gc
import logging
import math
import pathlib

import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
import tqdm
import tyro

from client_pool import get_global_pool

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data


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
    task_suite_name: str = "libero_spatial"  # libero_spatial, libero_object, libero_goal, libero_10, libero_90
    task_id: int = 0                          # 要评测的任务 ID（基于 task suite 的索引）
    num_steps_wait: int = 10                  # 开场等待步数，等物体稳定
    num_trials_per_task: int = 10             # 同一个任务重复多少次 rollout
    instruction: str = "pick up the black bowl between the plate and the ramekin and place it on the plate"                     # 外部指令（为空则使用任务自带 language）

    #################################################################################################################
    # Utils
    import time
    current_time = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
    #################################################################################################################
    video_out_path: str = f"data/libero/{current_time}/videos"  # Path to save videos
    save_videos: bool = False                     # Whether to save video replays (disable to save memory)
    seed: int = 7                                # Random Seed (for reproducibility)


def eval_one_task(args: Args) -> float:
    """
    使用客户端池评估单个任务
    从池中获取可用的客户端，而不是创建新的客户端
    """
    # Set random seed
    np.random.seed(args.seed)

    # 从全局池获取客户端
    pool = get_global_pool()
    if pool is None:
        raise RuntimeError("Client pool not initialized. Call initialize_global_pool() first.")
    
    client_info = pool.acquire(timeout=300.0)  # 最多等待5分钟
    if client_info is None:
        raise RuntimeError("Failed to acquire client from pool")
    
    client = client_info['client']
    logging.info(f"Using client {client_info['idx']} ({client_info['host']}:{client_info['port']})")

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    if args.task_suite_name not in benchmark_dict:
        pool.release(client_info)
        raise ValueError(
            f"Unknown task suite: {args.task_suite_name}. "
            f"Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90"
        )
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name} (n_tasks={num_tasks_in_suite})")

    # Validate task_id
    if not (0 <= args.task_id < num_tasks_in_suite):
        pool.release(client_info)
        raise ValueError(f"task_id={args.task_id} out of range [0, {num_tasks_in_suite-1}]")

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    # Set suite-specific max steps
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
        pool.release(client_info)
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    env = None  # Initialize to None for proper cleanup in finally block
    
    try:
        # Pick the single task
        task = task_suite.get_task(args.task_id)
        default_task_desc = task.language
        task_description = args.instruction.strip() if args.instruction.strip() else default_task_desc

        # Default initial states for this task
        initial_states = task_suite.get_task_init_states(args.task_id)
        if len(initial_states) == 0:
            raise RuntimeError("No initial states found for the selected task.")

        # Initialize environment for this task
        env, _ = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        total_episodes, total_successes = 0, 0

        logging.info(f"=== Start evaluating ONE task ===")
        logging.info(f"Task ID: {args.task_id}")
        logging.info(f"Instruction used: {task_description}")

        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
            # Choose init state (cycle if trials > available init states)
            init_state = initial_states[episode_idx % len(initial_states)]

            # Reset environment
            env.reset()
            env.set_init_state(init_state)

            action_plan = collections.deque()
            t = 0
            replay_images = []
            done = False  # avoid UnboundLocalError if exception occurs early

            logging.info(f"Starting episode {episode_idx + 1}/{args.num_trials_per_task}...")

            try:
                while t < max_steps + args.num_steps_wait:
                    # IMPORTANT: wait for objects to settle
                    if t < args.num_steps_wait:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        if done:
                            break
                        continue

                    # Get preprocessed images (rotate 180° to match train preprocessing)
                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])

                    img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
                    )
                    wrist_img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                    )

                    # Save for replay (only if video saving is enabled)
                    if args.save_videos:
                        replay_images.append(img)

                    if not action_plan:
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
                            "prompt": str(task_description),
                        }

                        # Query model to get action chunk
                        action_chunk = client.infer(element)["actions"]
                        assert len(action_chunk) >= args.replan_steps, (
                            f"We want to replan every {args.replan_steps} steps, "
                            f"but policy only predicts {len(action_chunk)} steps."
                        )
                        action_plan.extend(action_chunk[: args.replan_steps])

                    action = action_plan.popleft()
                    obs, reward, done, info = env.step(action.tolist())

                    if done:
                        total_successes += 1
                        break

                    t += 1

            except Exception as e:
                logging.error(f"Caught exception in episode {episode_idx + 1}: {e}")

            total_episodes += 1

            # Save a replay video of the episode
            if args.save_videos and replay_images:
                suffix = "success" if done else "failure"
                safe_desc = str(task_description).replace(" ", "_")[:60]
                video_name = f"rollout_suite-{args.task_suite_name}_task-{args.task_id}_ep-{episode_idx+1}_{suffix}_{safe_desc}.mp4"
                print(f"Saving video to {args.video_out_path} / {video_name}")
                try:
                    imageio.mimwrite(
                        pathlib.Path(args.video_out_path) / video_name,
                        [np.asarray(x) for x in replay_images],
                        fps=10,
                    )
                except Exception as e:
                    logging.error(f"Failed to write video {video_name}: {e}")
            
            # CRITICAL: Explicitly delete replay_images to free memory immediately
            if replay_images:
                del replay_images
            
            # Log current results
            logging.info(f"Episode {episode_idx + 1} success: {done}")
            logging.info(f"# episodes so far: {total_episodes}")
            logging.info(f"# successes: {total_successes} ({(total_successes / total_episodes * 100):.1f}%)")

        # Final results
        success_rate = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0.0
        logging.info(f"Final success rate for task {args.task_id}: {success_rate:.3f}")
        logging.info(f"Total episodes: {total_episodes}")
        return success_rate
    
    finally:
        # CRITICAL: Clean up resources to prevent memory leaks
        if env is not None:
            try:
                env.close()
                logging.info("Environment closed successfully")
            except Exception as e:
                logging.error(f"Failed to close environment: {e}")
        
        # 将客户端归还到池中（而不是关闭）
        pool.release(client_info)
        logging.info(f"Client {client_info['idx']} released back to pool")
        
        # Explicitly delete large objects
        if 'task_suite' in locals():
            del task_suite
        
        # Force garbage collection
        gc.collect()


def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment."""
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task.language


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
    tyro.cli(eval_one_task)

