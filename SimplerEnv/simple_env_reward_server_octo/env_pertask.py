import dataclasses
import numpy as np
from simpler_env.utils.env.env_builder import build_maniskill2_env, get_robot_control_mode
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
from .policy_client import HttpPolicyClient


@dataclasses.dataclass
class Args:
    host: str = "0.0.0.0"
    port: int = 8001
    resize_size: int = 256
    replan_steps: int = 1
    robot: str = "widowx"
    env_name: str = "PutCarrotOnPlateInScene-v0"
    scene_name: str = "bridge_table_1_v1"
    num_steps_wait: int = 10
    num_trials_per_task: int = 1
    instruction: str = ""
    init_state_id: int = 0
    rgb_overlay_path: str | None = None
    control_freq: int = 5
    sim_freq: int = 500
    max_episode_steps: int = 80
    additional_env_build_kwargs: dict | None = None
    obs_camera_name: str | None = None
    action_scale: float = 1.0
    policy_setup: str = "google_robot"
    model_type: str = "octo-base"
    seed: int = 7


DUMMY_ACTION = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float32)


def eval_one_task(args: Args) -> float:
    if args.additional_env_build_kwargs is None:
        args.additional_env_build_kwargs = {}
    control_mode = get_robot_control_mode(args.robot, "octo")
    kwargs = dict(
        obs_mode="rgbd",
        robot=args.robot,
        sim_freq=args.sim_freq,
        control_mode=control_mode,
        control_freq=args.control_freq,
        max_episode_steps=args.max_episode_steps,
        scene_name=args.scene_name,
        camera_cfgs={"add_segmentation": True},
        rgb_overlay_path=args.rgb_overlay_path,
    )
    env = build_maniskill2_env(args.env_name, **args.additional_env_build_kwargs, **kwargs)
    client = HttpPolicyClient(args.host, args.port)
    session_id = f"octo-{args.env_name}-{args.scene_name}-{args.init_state_id}-{args.seed}"
    client.reset(session_id, args.instruction, policy_setup=args.policy_setup, action_scale=args.action_scale, model_type=args.model_type)
    total_successes = 0
    total_episodes = 0
    for _ in range(args.num_trials_per_task):
        obs, _ = env.reset()
        image = get_image_from_maniskill2_obs_dict(env, obs, camera_name=args.obs_camera_name)
        t = 0
        done = False
        while t < args.max_episode_steps and not done:
            if t < args.num_steps_wait:
                obs, reward, done, truncated, info = env.step(DUMMY_ACTION)
                if done or truncated:
                    break
                image = get_image_from_maniskill2_obs_dict(env, obs, camera_name=args.obs_camera_name)
                t += 1
                continue
            rep = client.infer(session_id, image, task_description=args.instruction)
            act = rep["action"]
            action_vec = np.concatenate([np.asarray(act["world_vector"]), np.asarray(act["rot_axangle"]), np.asarray(act["gripper"])])
            obs, reward, done, truncated, info = env.step(action_vec)
            if done:
                total_successes += 1
                break
            image = get_image_from_maniskill2_obs_dict(env, obs, camera_name=args.obs_camera_name)
            t += 1
        total_episodes += 1
    env.close()
    return float(total_successes) / float(total_episodes) if total_episodes > 0 else 0.0

