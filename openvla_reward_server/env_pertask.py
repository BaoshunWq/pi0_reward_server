import collections
import dataclasses
import gc
import logging
import math
import pathlib
import warnings
import random

import sys
sys.path.insert(0, "/root/autodl-tmp/code/attackVLA/pi0_reward_server/openpi/third_party/libero")
# 注意：EGL上下文警告是AttributeError异常，不是Warning
# 无法通过warnings模块过滤，这些警告可以安全忽略

import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm
import tyro

# Import action processing functions for OpenVLA
sys.path.insert(0, "/root/autodl-tmp/code/attackVLA/pi0_reward_server/openvla-oft/experiments/robot")
# from robot_utils import normalize_gripper_action, invert_gripper_action

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data

def invert_gripper_action(action: np.ndarray) -> np.ndarray:
    """
    Flip the sign of the gripper action (last dimension of action vector).

    This is necessary for environments where -1 = open, +1 = close, since
    the RLDS dataloader aligns gripper actions such that 0 = close, 1 = open.

    Args:
        action: Action array with gripper action in the last dimension

    Returns:
        np.ndarray: Action array with inverted gripper action
    """
    # Create a copy to avoid modifying the original
    inverted_action = action.copy()

    # Invert the gripper action
    inverted_action[..., -1] *= -1.0

    return inverted_action

def normalize_gripper_action(action: np.ndarray, binarize: bool = True) -> np.ndarray:
    """
    Normalize gripper action from [0,1] to [-1,+1] range.

    This is necessary for some environments because the dataset wrapper
    standardizes gripper actions to [0,1]. Note that unlike the other action
    dimensions, the gripper action is not normalized to [-1,+1] by default.

    Normalization formula: y = 2 * (x - orig_low) / (orig_high - orig_low) - 1

    Args:
        action: Action array with gripper action in the last dimension
        binarize: Whether to binarize gripper action to -1 or +1

    Returns:
        np.ndarray: Action array with normalized gripper action
    """
    # Create a copy to avoid modifying the original
    normalized_action = action.copy()

    # Normalize the last action dimension to [-1,+1]
    orig_low, orig_high = 0.0, 1.0
    normalized_action[..., -1] = 2 * (normalized_action[..., -1] - orig_low) / (orig_high - orig_low) - 1

    if binarize:
        # Binarize to -1 or +1
        normalized_action[..., -1] = np.sign(normalized_action[..., -1])

    return normalized_action

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
    init_state_id: int = 0                    # 初始状态 ID

    #################################################################################################################
    # Utils
    import time
    current_time = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
    #################################################################################################################
    video_out_path: str = f"data/libero/{current_time}/videos"  # Path to save videos
    save_videos: bool = False                     # Whether to save video replays (disable to save memory)
    seed: int = 7                                # Random Seed (for reproducibility)


def eval_one_task(args: Args) -> None:
    # Set random seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Set torch seed if torch is available
    try:
        import torch
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass  # torch not available, skip
    
    logging.info(f"Random seed set to: {args.seed}")

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    if args.task_suite_name not in benchmark_dict:
        raise ValueError(
            f"Unknown task suite: {args.task_suite_name}. "
            f"Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90"
        )
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name} (n_tasks={num_tasks_in_suite})")

    # Validate task_id
    if not (0 <= args.task_id < num_tasks_in_suite):
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
        # 保险起见；理论上前面已校验过
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
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

        # for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
            # Choose init state (cycle if trials > available init states)
        # episode_idx = args.init_state_id
        init_state = initial_states[args.init_state_id]

        print(f"init_state id: {args.init_state_id}")

        # Reset environment
        env.reset()
        env.set_init_state(init_state)

        action_plan = collections.deque()
        t = 0
        replay_images = []
        done = False  # avoid UnboundLocalError if exception occurs early

        # logging.info(f"Starting episode {episode_idx + 1}/{args.num_trials_per_task}...")

        try:
            # obs = env.get_observation() if hasattr(env, "get_observation") else env._get_observation()  # fallback

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
                
                # Process action before sending to environment
                # This is critical for OpenVLA: normalize gripper action and invert sign
                action = normalize_gripper_action(action, binarize=True)
                action = invert_gripper_action(action)
                
                obs, reward, done, info = env.step(action.tolist())

                if done:
                    total_successes += 1
                    break

                t += 1

        except Exception as e:
            logging.error(f"Caught exception in episode: {e}")
            import traceback
            logging.error(f"Traceback: {traceback.format_exc()}")

        total_episodes += 1

        # Save a replay video of the episode
        if args.save_videos and replay_images:
            suffix = "success" if done else "failure"
            safe_desc = str(task_description).replace(" ", "_")[:60]
            video_name = f"rollout_suite-{args.task_suite_name}_task-{args.task_id}_init-{args.init_state_id}_{suffix}_{safe_desc}.mp4"
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
        # logging.info(f"Episode {episode_idx + 1} success: {done}")
        logging.info(f"# episodes so far: {total_episodes}")
        logging.info(f"# successes: {total_successes} ({(total_successes / total_episodes * 100):.1f}%)")

        # Final results
        # Final results + **返回值**
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
        
        # Close WebSocket client connection
        try:
            if 'client' in locals() and hasattr(client, '_ws') and client._ws is not None:
                client._ws.close()
                logging.info("WebSocket client closed successfully")
        except Exception as e:
            logging.error(f"Failed to close WebSocket client: {e}")
        
        # Explicitly delete large objects
        if 'client' in locals():
            del client
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

