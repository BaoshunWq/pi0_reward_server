"""
Minimal LIBERO evaluator that always talks to a remote policy server.
Derived from `run_libero_eval.py` but all local model loading is removed.
"""

import json
import logging
import os
import sys
from collections import deque
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional
import torch
import random
import draccus
import numpy as np
import tqdm
from typing import Any, Dict, List, Optional, Tuple, Union
import tensorflow as tf
# Ensure dependent packages resolve regardless of working directory
REPO_ROOT = Path(__file__).resolve().parents[4]
OPENVLA_ROOT = REPO_ROOT / "openvla-oft"
sys.path.insert(0, str(REPO_ROOT / "openpi" / "third_party" / "libero"))
# Allow `experiments.*` imports
sys.path.insert(0, str(OPENVLA_ROOT))

from libero.libero import benchmark
from openpi_client import websocket_client_policy as _websocket_client_policy
from experiments.robot.libero.libero_utils import (  # noqa: E402
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    get_libero_wrist_image,
    quat2axisangle,
    save_rollout_video,
)
import time
MODEL_IMAGE_SIZES = {
    "openvla": 224,
    # Add other models as needed
}
DATE = time.strftime("%Y_%m_%d")
DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")
# from experiments.robot.openvla_utils import resize_image_for_policy  # noqa: E402
def resize_image_for_policy(img: np.ndarray, resize_size: Union[int, Tuple[int, int]]) -> np.ndarray:
    """
    Resize an image to match the policy's expected input size.

    Uses the same resizing scheme as in the training data pipeline for distribution matching.

    Args:
        img: Numpy array containing the image
        resize_size: Target size as int (square) or (height, width) tuple

    Returns:
        np.ndarray: The resized image
    """
    assert isinstance(resize_size, int) or isinstance(resize_size, tuple)
    if isinstance(resize_size, int):
        resize_size = (resize_size, resize_size)

    # Resize using the same pipeline as in RLDS dataset builder
    img = tf.image.encode_jpeg(img)  # Encode as JPEG
    img = tf.io.decode_image(img, expand_animations=False, dtype=tf.uint8)  # Decode back
    img = tf.image.resize(img, resize_size, method="lanczos3", antialias=True)
    img = tf.cast(tf.clip_by_value(tf.round(img), 0, 255), tf.uint8)

    return img.numpy()

def get_image_resize_size(cfg: Any) -> Union[int, tuple]:
    """
    Get image resize dimensions for a specific model.

    If returned value is an int, the resized image will be a square.
    If returned value is a tuple, the resized image will be a rectangle.

    Args:
        cfg: Configuration object with model parameters

    Returns:
        Union[int, tuple]: Image resize dimensions

    Raises:
        ValueError: If model family is not supported
    """
    if cfg.model_family not in MODEL_IMAGE_SIZES:
        raise ValueError(f"Unsupported model family: {cfg.model_family}")

    return MODEL_IMAGE_SIZES[cfg.model_family]
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

def set_seed_everywhere(seed: int) -> None:
    """
    Set random seed for all random number generators for reproducibility.

    Args:
        seed: The random seed to use
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

# from experiments.robot.robot_utils import (  # noqa: E402
#     get_image_resize_size,
#     invert_gripper_action,
#     normalize_gripper_action,
#     set_seed_everywhere,
# )

from experiments.robot.libero.constants import NUM_ACTIONS_CHUNK  # noqa: E402


class TaskSuite(str, Enum):
    LIBERO_SPATIAL = "libero_spatial"
    LIBERO_OBJECT = "libero_object"
    LIBERO_GOAL = "libero_goal"
    LIBERO_10 = "libero_10"
    LIBERO_90 = "libero_90"


TASK_MAX_STEPS = {
    TaskSuite.LIBERO_SPATIAL: 220,
    TaskSuite.LIBERO_OBJECT: 280,
    TaskSuite.LIBERO_GOAL: 300,
    TaskSuite.LIBERO_10: 520,
    TaskSuite.LIBERO_90: 400,
}


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


@dataclass
class GenerateConfig:
    #################################################################################################################
    # Policy server parameters
    #################################################################################################################
    policy_server_host: str = "localhost"  # Use "localhost" for client connections, not "0.0.0.0"
    policy_server_port: int = 23451
    wait_for_policy_server: bool = True

    #################################################################################################################
    # Policy + observation parameters
    #################################################################################################################
    model_family: str = "openvla"  # Used for resize + dummy action/gripper handling
    num_open_loop_steps: int = NUM_ACTIONS_CHUNK

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = TaskSuite.LIBERO_SPATIAL
    num_steps_wait: int = 10
    num_trials_per_task: int = 50
    initial_states_path: str = "DEFAULT"
    env_img_res: int = 256

    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note: Optional[str] = None
    local_log_dir: str = "./experiments/logs"
    use_wandb: bool = False  # Kept for interface parity; unused here
    seed: int = 7

    is_save_video: bool = False


def validate_config(cfg: GenerateConfig) -> None:
    assert cfg.task_suite_name in [suite.value for suite in TaskSuite], f"Invalid task suite: {cfg.task_suite_name}"
    if cfg.num_open_loop_steps <= 0:
        raise ValueError("num_open_loop_steps must be positive")


def setup_logging(cfg: GenerateConfig):
    run_id = f"EVAL-REMOTE-{cfg.task_suite_name}-{cfg.model_family}-{DATE_TIME}"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"

    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    logger.info("Logging to local log file: %s", local_log_filepath)

    return log_file, local_log_filepath, run_id


def log_message(message: str, log_file=None):
    logger.info(message)
    if log_file:
        log_file.write(message + "\n")
        log_file.flush()


def load_initial_states(cfg: GenerateConfig, task_suite, task_id: int, log_file=None):
    initial_states = task_suite.get_task_init_states(task_id)
    if cfg.initial_states_path != "DEFAULT":
        with open(cfg.initial_states_path, "r") as f:
            all_initial_states = json.load(f)
        log_message(f"Using initial states from {cfg.initial_states_path}", log_file)
        return initial_states, all_initial_states
    log_message("Using default initial states", log_file)
    return initial_states, None


def prepare_observation(obs, resize_size):
    img = get_libero_image(obs)
    wrist_img = get_libero_wrist_image(obs)

    observation = {
        "full_image": resize_image_for_policy(img, resize_size),
        "wrist_image": resize_image_for_policy(wrist_img, resize_size),
        "state": np.concatenate(
            (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
        ),
    }

    return observation, img


def process_action(action, model_family):
    action = normalize_gripper_action(action, binarize=True)
    if model_family == "openvla":
        action = invert_gripper_action(action)
    return action


def request_actions_from_policy(cfg: GenerateConfig, observation: dict, task_description: str, policy_client):
    payload = {
        "observation": observation,
        "task_description": task_description,
        "requested_chunk": cfg.num_open_loop_steps,
        "task_suite_name": cfg.task_suite_name,  # Send task_suite_name to policy server
    }
    response = policy_client.infer(payload)
    actions = response.get("actions")
    if actions is None:
        raise ValueError("Remote policy server response is missing `actions` field.")
    return [np.asarray(action) for action in actions]


def run_episode(
    cfg: GenerateConfig,
    env,
    task_description: str,
    policy_client,
    resize_size,
    initial_state=None,
    log_file=None,
):
    env.reset()
    if initial_state is not None:
        obs = env.set_init_state(initial_state)
    else:
        obs = env.get_observation()

    if cfg.num_open_loop_steps != NUM_ACTIONS_CHUNK:
        print(
            f"WARNING: cfg.num_open_loop_steps ({cfg.num_open_loop_steps}) does not match NUM_ACTIONS_CHUNK "
            f"({NUM_ACTIONS_CHUNK})."
        )
    action_queue = deque(maxlen=cfg.num_open_loop_steps)

    t = 0
    replay_images = []
    max_steps = TASK_MAX_STEPS[TaskSuite(cfg.task_suite_name)]
    success = False
    try:
        while t < max_steps + cfg.num_steps_wait:
            if t < cfg.num_steps_wait:
                obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
                t += 1
                continue

            observation, img = prepare_observation(obs, resize_size)
            if cfg.is_save_video:
                replay_images.append(img)

            if len(action_queue) == 0:
                actions = request_actions_from_policy(cfg, observation, task_description, policy_client)
                action_queue.extend(actions)

            action = action_queue.popleft()
            action = process_action(action, cfg.model_family)
            obs, reward, done, info = env.step(action.tolist())
            if done:
                success = True
                break
            t += 1
    except Exception as e:
        log_message(f"Episode error: {e}", log_file)

    return success, replay_images


def run_task(cfg: GenerateConfig, task_suite, task_id: int, policy_client, resize_size, total_episodes, total_successes, log_file=None):
    task = task_suite.get_task(task_id)
    initial_states, all_initial_states = load_initial_states(cfg, task_suite, task_id, log_file)
    env, task_description = get_libero_env(task, cfg.model_family, resolution=cfg.env_img_res)

    task_episodes, task_successes = 0, 0
    for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task)):
        log_message(f"\nTask: {task_description}", log_file)
        if cfg.initial_states_path == "DEFAULT":
            initial_state = initial_states[episode_idx]
        else:
            initial_states_task_key = task_description.replace(" ", "_")
            episode_key = f"demo_{episode_idx}"
            if not all_initial_states[initial_states_task_key][episode_key]["success"]:
                log_message(f"Skipping task {task_id} episode {episode_idx} due to failed expert demo!", log_file)
                continue
            initial_state = np.array(all_initial_states[initial_states_task_key][episode_key]["initial_state"])

        log_message(f"Starting episode {task_episodes + 1}...", log_file)
        success, replay_images = run_episode(
            cfg=cfg,
            env=env,
            task_description=task_description,
            policy_client=policy_client,
            resize_size=resize_size,
            initial_state=initial_state,
            log_file=log_file,
        )

        task_episodes += 1
        total_episodes += 1
        if success:
            task_successes += 1
            total_successes += 1
        if cfg.is_save_video:
            save_rollout_video(replay_images, total_episodes, success=success, task_description=task_description, log_file=log_file)

        log_message(f"Success: {success}", log_file)
        log_message(f"# episodes completed so far: {total_episodes}", log_file)
        log_message(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)", log_file)

    task_success_rate = float(task_successes) / float(task_episodes) if task_episodes > 0 else 0
    total_success_rate = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0
    log_message(f"Current task success rate: {task_success_rate}", log_file)
    log_message(f"Current total success rate: {total_success_rate}", log_file)

    return total_episodes, total_successes


def connect_policy_client(cfg: GenerateConfig):
    logger.info("Connecting to remote policy server at %s:%s", cfg.policy_server_host, cfg.policy_server_port)
    client = _websocket_client_policy.WebsocketClientPolicy(
        host=cfg.policy_server_host,
        port=cfg.policy_server_port,
    )
    metadata = client.get_server_metadata()
    logger.info("Connected to policy server. Metadata: %s", metadata)
    return client


@draccus.wrap()
def eval_libero(cfg: GenerateConfig) -> float:
    validate_config(cfg)
    set_seed_everywhere(cfg.seed)
    resize_size = get_image_resize_size(cfg)

    log_file, local_log_filepath, run_id = setup_logging(cfg)

    client = connect_policy_client(cfg)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    log_message(f"Task suite: {cfg.task_suite_name} ({num_tasks_in_suite} tasks)", log_file)

    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        total_episodes, total_successes = run_task(
            cfg=cfg,
            task_suite=task_suite,
            task_id=task_id,
            policy_client=client,
            resize_size=resize_size,
            total_episodes=total_episodes,
            total_successes=total_successes,
            log_file=log_file,
        )

    total_success_rate = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0
    log_message(f"Total success rate: {total_success_rate}", log_file)
    log_message(f"Total episodes: {total_episodes}", log_file)

    return total_success_rate


if __name__ == "__main__":
    eval_libero()

