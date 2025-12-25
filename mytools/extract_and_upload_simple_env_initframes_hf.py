import os
import json
import numpy as np
from PIL import Image
from tqdm import tqdm
from huggingface_hub import HfApi, HfFolder, upload_file, create_repo
import simpler_env
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict

HF_USERNAME = "TBS2001"
HF_REPO_NAME = "simple-env-init-frames"
TASK_LIST = [
    "google_robot_pick_coke_can",
]  # 若无法自动枚举，将使用此后备列表
LOCAL_SAVE_DIR = "simple-env-init-frames"
OUTPUT_JSON = "simple-env-init-frames/json_data_for_rl/simple_env_initial_state_links.json"


def _get_all_simple_env_tasks():
    try:
        tasks = getattr(simpler_env, "ENVIRONMENTS", None)
        if isinstance(tasks, (list, tuple)) and tasks:
            return list(tasks)
    except Exception:
        pass
    return TASK_LIST


def make_env(task_name: str):
    env = simpler_env.make(task_name)
    return env


def extract_initial_frame(env, task_name: str, task_id: int, img_id: int, save_dir: str) -> str:
    obs, _ = env.reset()
    img = get_image_from_maniskill2_obs_dict(env, obs)
    img = np.array(img, dtype=np.uint8)
    os.makedirs(save_dir, exist_ok=True)
    file_name = f"{task_name}_task-{task_id}_img-{img_id}_frame0.png"
    save_path = os.path.join(save_dir, file_name)
    Image.fromarray(img).save(save_path)
    return save_path


def upload_to_huggingface(local_path: str, username: str, repo_name: str) -> str:
    api = HfApi()
    repo_id = f"{username}/{repo_name}"
    try:
        create_repo(repo_id, repo_type="dataset", private=False)
    except Exception:
        pass
    remote_path = os.path.basename(local_path)
    try:
        upload_file(
            path_or_fileobj=local_path,
            path_in_repo=remote_path,
            repo_id=repo_id,
            repo_type="dataset",
            token=HfFolder.get_token(),
        )
    except Exception:
        pass
    return f"https://huggingface.co/datasets/{repo_id}/resolve/main/{remote_path}"


def main():
    os.makedirs(LOCAL_SAVE_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    result = {"simple_env": {}}
    all_tasks = _get_all_simple_env_tasks()
    for task_id, task_name in enumerate(tqdm(all_tasks, desc="Tasks")):
        env = make_env(task_name)
        save_path = extract_initial_frame(env, task_name, task_id, 0, LOCAL_SAVE_DIR)
        env.close()
        image_url = upload_to_huggingface(save_path, HF_USERNAME, HF_REPO_NAME)
        result["simple_env"][task_name] = [image_url]
        with open(OUTPUT_JSON, "w") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
