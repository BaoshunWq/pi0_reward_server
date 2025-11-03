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
from utils import load_relate_model,parse_task_and_links

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

    seed: int = 7  # Random Seed (for reproducibility)
    custom_lora_path: str = ""  # smolvlm custom lora path
    qwen_mode: str = "api"  # qwenvl mode: "local" or "api"
    qwen_model_id : str = "Qwen/Qwen2-VL-2B-Instruct"
    semantic_type: str = "clip"  # "clip" or "deberta"
    num_instructions: int = 10  # number of generated instructions
    select_topk: int = 5  # select top k instructions
    task_to_huglinks_json_path: str = "/home/baoshuntong/code/saftyEmbodyAI/redTeam_pi0/openpi/libero-init-frames/json_data_for_rl/vlm_initial_state_links_new.json"  # path to task to huglinks json
    BACKEND: str = "qwenvl"  # "smolvlm" or "qwenvl"
    output_path: str = f"data/{formatted_time}libero/results/libero_vlmrewrite_results.json"  # path to save results
    whole_acc_log_path: str = f"data/{formatted_time}libero/results/libero_whole_acc_log.json"  # path to save whole accuracy log
    failure_threshold: int = 5


def eval_libero(args: Args) -> None:
    # Set random seed
    np.random.seed(args.seed)



    red_team = load_relate_model(args)

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

    with open(args.task_to_huglinks_json_path, "r") as f:
        task_to_links = json.load(f)  # {task: [image_url, ...]}

    task_language_list, task_links_suite = parse_task_and_links(args.task_suite_name,task_to_links)

    annotations_smi_all = []

    results = []  # task -> list of dicts per round / per candidate
    # Start evaluation
    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        task_links_key = task_description.replace(" ", "_")

        current_task_links = task_links_suite[task_links_key]

        image_url = random.choice(current_task_links)



        annotations,annotations_smi = red_team(task_description,semantic_type= args.semantic_type, image_url=image_url,
                                               num_instructions=args.num_instructions,select_topk=args.select_topk)

        # import pdb
        # pdb.set_trace()
        annotations_smi_all.extend(annotations_smi)
        annotations_smi = [float(x) for x in annotations_smi]  # 去掉 np.float32

        done_anotations = []
        fail_anotations = []
        done_annotations_smi = []
        fail_annotations_smi = []

        current_task_result = {}

        for i,(annotation,annotation_sim) in enumerate(zip(annotations,annotations_smi)):

            # task_description = annotation

            print(f"evaluation task description:{annotation}, It's similarity :{annotation_sim}")
        # Start episodes
            task_episodes, task_successes = 0, 0

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

                logging.info(f"Starting episode {task_episodes+1}...")
                while t < max_steps + args.num_steps_wait:
                    # try:
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

                    # Save preprocessed image for replay video
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
                            # "prompt": str(task_description),
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

                    # except Exception as e:
                    #     logging.error(f"Caught exception: {e}")
                    #     break

                task_episodes += 1
                total_episodes += 1

                # Save a replay video of the episode
                suffix = "success" if done else "failure"
                # task_segment = task_description.replace(" ", "_")
                task_segment = annotation.replace(" ", "_")
                imageio.mimwrite(
                    pathlib.Path(args.video_out_path) / f"rollout{suffix}_{task_segment}.mp4",
                    [np.asarray(x) for x in replay_images],
                    fps=10,
                )

                 # Log current results
                logging.info(f"Success: {done}")
                logging.info(f"# episodes completed so far: {total_episodes}")
                logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")
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
        logging.info(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        logging.info(f"Current total success rate: {float(total_successes) / float(total_episodes)}")

    mean_val = np.mean(annotations_smi_all)
    var_val = np.var(annotations_smi_all)
    print(f" Overall task similarity  Mean: {mean_val:.4f}, Variance: {var_val:.4f}")

    all_result = {}

    all_success_rate = float(total_successes) / float(total_episodes)

    all_result[args.task_suite_name] = all_success_rate

    os.makedirs(os.path.dirname(args.whole_acc_log_path), exist_ok=True)
    with open(args.whole_acc_log_path, "a") as fout:
        json.dump(all_result, fout,indent=4)

    print(f"Done. Results saved to {args.output_path}")

    logging.info(f"Total success rate: {float(total_successes) / float(total_episodes)}")
    logging.info(f"Total episodes: {total_episodes}")


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
