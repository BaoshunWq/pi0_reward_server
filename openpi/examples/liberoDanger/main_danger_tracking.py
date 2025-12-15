import os
os.environ["LIBERO_CONFIG_PATH"] = "vla_simulator_env/liberoDanger/libero/configs"
import sys
sys.path.insert(0, "vla_simulator_env/liberoDanger")
import collections
import dataclasses
import json
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
import datetime
LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data
formatted_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "0.0.0.0"
    port: int = 5555  # Policy服务器端口（WebSocket），不是负载均衡器端口
    resize_size: int = 224
    replan_steps: int = 5

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = (
        "libero_spatial"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    )
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize i n sim
    num_trials_per_task: int = 1  # Number of rollouts per task

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = f"output/{formatted_time}/danger_collision_tracking/{task_suite_name}/videos"  # Path to save videos
    collision_json_path: str = f"output/{formatted_time}/collision_counts.json"  # Path to save collision counts

    seed: int = 7  # Random Seed (for reproducibility)

    is_save_video: bool = True  # 是否保存视频



    


def load_collision_json(json_path: str) -> dict:
    """加载碰撞计数JSON文件，如果不存在则返回空字典"""
    if pathlib.Path(json_path).exists():
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logging.warning(f"Failed to load collision JSON file: {e}. Starting with empty dict.")
            return {}
    return {}


def save_collision_json(json_path: str, data: dict) -> None:
    """保存碰撞计数到JSON文件"""
    # 确保目录存在
    pathlib.Path(json_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logging.info(f"Saved collision counts to {json_path}")


def update_suite_collision_count(json_path: str, suite_name: str, collision_count: int, 
                                  total_episodes: int, total_successes: int, 
                                  timestamp: str = None) -> None:
    """更新指定suite的碰撞计数到JSON文件"""
    # 加载现有数据
    data = load_collision_json(json_path)
    
    # 更新或添加suite记录
    if suite_name not in data:
        data[suite_name] = []
    
    # 添加新的运行记录
    record = {
        'collision_count': collision_count,
        'total_episodes': total_episodes,
        'total_successes': total_successes,
        'success_rate': float(total_successes) / float(total_episodes) if total_episodes > 0 else 0.0,
        'timestamp': timestamp or datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    
    data[suite_name].append(record)
    
    # 保存更新后的数据
    save_collision_json(json_path, data)
    
    logging.info(f"Updated collision count for suite '{suite_name}': {collision_count} episodes with collision "
                 f"({total_episodes} total episodes, {total_successes} successes)")


def eval_libero(args: Args) -> None:
    # Set random seed
    np.random.seed(args.seed)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}")

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

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

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    # Start evaluation
    total_episodes, total_successes = 0, 0
    task_collision_rates = []  # 存储每个任务的碰撞率
    suite_total_collision_count = 0  # 整个suite的总碰撞次数（每次进入危险区域+1）
    
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        # Start episodes
        task_episodes, task_successes = 0, 0
        task_collision_count_total = 0  # 当前任务的总碰撞次数
        task_move_step_count = 0
        
        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
            logging.info(f"\nTask: {task_description}")

            # Reset environment
            env.reset()
            action_plan = collections.deque()

            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])

            # Setup
            t = 0
            replay_images = []
            episode_has_collision = False  # 当前episode是否发生了碰撞（只记录一次）

            logging.info(f"Starting episode {task_episodes+1}...")
            while t < max_steps + args.num_steps_wait:
                try:
                    # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                    # and we need to wait for them to fall
                    if t < args.num_steps_wait:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        # 在等待阶段也检查碰撞
                        if not episode_has_collision and info.get('collision_count', 0) > 0:
                            episode_has_collision = True
                        # print(f"info: {info}")
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

                    # Save preprocessed image for replay video
                    if args.is_save_video:
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
                            "prompt": str(task_description),
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
                    
                    # 检查是否发生了碰撞（只记录一次，不统计进入次数）
                    if not episode_has_collision and info.get('collision_count', 0) > 0:
                        episode_has_collision = True
                    
                    # print(f"info: {info}")
                    if done:
                        task_successes += 1
                        total_successes += 1
                        print(f"t: {t}")
                        break
                        
                    
                    t += 1

                except Exception as e:
                    logging.error(f"Caught exception: {e}")
                    break

            task_episodes += 1
            total_episodes += 1
            # 每个episode只记录是否发生碰撞（0或1）
            episode_collision_count = 1 if episode_has_collision else 0
            task_collision_count_total += episode_collision_count  # 累加当前episode的碰撞次数（0或1）
            suite_total_collision_count += episode_collision_count  # 累加到suite总数
            task_move_step_count += t

            # Save a replay video of the episode
            if args.is_save_video:
                suffix = "success" if done else "failure"
                task_segment = task_description.replace(" ", "_")
                print(f"Saving video to {args.video_out_path}/{f'rollout_{task_segment}_{suffix}.mp4'}")
                imageio.mimwrite(
                    pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_{suffix}.mp4",
                    [np.asarray(x) for x in replay_images],
                fps=10,
            )

            # Log current results
            logging.info(f"Success: {done}")
            logging.info(f"# episodes completed so far: {total_episodes}")
            logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")
            logging.info(f"Episode has collision: {episode_has_collision} (count: {episode_collision_count})")

        # 计算当前任务的碰撞率（发生碰撞的episode比例）
        task_collision_rate = task_collision_count_total / task_episodes if task_episodes > 0 else 0.0
        task_collision_rates.append(task_collision_rate)
        
        # Log final results
        logging.info(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        logging.info(f"Current task collision rate: {task_collision_rate:.4f} ({task_collision_count_total}/{task_episodes} episodes had collision)")
        logging.info(f"Current task collision count: {task_collision_count_total} (episodes with collision)")
        logging.info(f"Current total success rate: {float(total_successes) / float(total_episodes)}")

    # 计算最终的碰撞统计
    mean_collision_rate = np.mean(task_collision_rates) if len(task_collision_rates) > 0 else 0.0
    variance_collision_rate = np.var(task_collision_rates) if len(task_collision_rates) > 0 else 0.0
    
    logging.info(f"Total success rate: {float(total_successes) / float(total_episodes)}")
    logging.info(f"Total episodes: {total_episodes}")
    logging.info(f"Suite total collision count: {suite_total_collision_count} (episodes with collision)")
    logging.info(f"Mean collision rate: {mean_collision_rate:.4f} (proportion of episodes with collision)")
    logging.info(f"Variance collision rate: {variance_collision_rate:.6f}")
    logging.info(f"Task collision rates: {[f'{rate:.4f}' for rate in task_collision_rates]}")
    
    # 将suite的碰撞次数写入JSON文件
    update_suite_collision_count(
        args.collision_json_path,
        args.task_suite_name,
        suite_total_collision_count,
        total_episodes,
        total_successes,
        formatted_time
    )


def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    print(f"task_bddl_file: {task_bddl_file}")
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

