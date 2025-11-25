import sys
import os
import json
import numpy as np
from PIL import Image
from tqdm import tqdm
from huggingface_hub import HfApi, HfFolder, upload_file, create_repo

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, "/root/autodl-tmp/code/attackVLA/pi0_reward_server/openpi/third_party/libero")

from libero.libero import benchmark  # noqa: E402
from libero.libero import get_libero_path  # noqa: E402
from libero.libero.envs import OffScreenRenderEnv  # noqa: E402

# ========== 用户需修改的部分 ==========
HF_USERNAME = "TBS2001"  # 你的 HF 用户名
HF_REPO_NAME = "libero-init-frames-multiview"  # 数据集仓库名
TASK_LIST = [
    'libero_spatial',
    'libero_object',
    'libero_goal',
    "libero_10",
    # 'libero_90',
]
LOCAL_SAVE_DIR = "./libero-init-frames_new"
OUTPUT_JSON = "libero-init-frames_new/json_data_for_rl/vlm_initial_state_links.json"
CAMERA_NAMES = ["agentview", "robot0_eye_in_hand", "frontview", "birdview"]
# =====================================


def get_libero_env(task, resolution=256):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = os.path.join(
        get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
    )
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": resolution,
        "camera_widths": resolution,
        "camera_names": CAMERA_NAMES,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(0)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def make_env(task):
    """创建一个 LIBERO 环境实例"""
    env, _ = get_libero_env(task, resolution=384)
    return env


def extract_initial_frames(env, task_name, task_id, img_id, save_dir):
    """Reset 环境并保存第 0 帧（初始观测图像）的多视角图像。"""
    obs = env.reset()

    if not isinstance(obs, dict):
        raise ValueError("obs 不是字典格式，无法解析多视角图像。")

    saved_paths = {}
    task_folder = task_name.problem_folder

    for cam in CAMERA_NAMES:
        key = f"{cam}_image"
        if key not in obs:
            raise ValueError(f"在 obs 中找不到 {key}。请确认 camera_names 配置。")

        img = np.array(obs[key], dtype=np.uint8)
        # img = np.flip(img, axis=(0, 1))  # or: img = np.rot90(img, 2)

        os.makedirs(save_dir, exist_ok=True)
        file_name = f"{task_folder}_task-{task_id}_img-{img_id}_{cam}_frame0.png"
        save_path = os.path.join(save_dir, file_name)
        Image.fromarray(img).save(save_path)
        saved_paths[cam] = save_path

    if not saved_paths:
        raise ValueError("未能保存任何视角图像。")

    return saved_paths


def upload_to_huggingface(local_path, username, repo_name):
    """上传到 HuggingFace Hub 并返回文件的原始 URL。"""
    api = HfApi()
    repo_id = f"{username}/{repo_name}"
    try:
        create_repo(repo_id, repo_type="dataset", private=False)
    except Exception:
        pass  # 已存在则忽略

    remote_path = os.path.basename(local_path)
    # upload_file(
    #     path_or_fileobj=local_path,
    #     path_in_repo=remote_path,
    #     repo_id=repo_id,
    #     repo_type="dataset",
    #     token=HfFolder.get_token(),
    # )
    return f"https://huggingface.co/datasets/{repo_id}/resolve/main/{remote_path}"


def main():
    os.makedirs(LOCAL_SAVE_DIR, exist_ok=True)
    task_to_url = {}

    benchmark_dict = benchmark.get_benchmark_dict()

    for task_name in TASK_LIST:
        per_suite_task_to_url = {}
        task_suite = benchmark_dict[task_name]()
        num_tasks_in_suite = task_suite.n_tasks

        for task_id in tqdm(range(num_tasks_in_suite)):
            task = task_suite.get_task(task_id)
            task_key = task.language.lower().replace(" ", "_")
            img_id = 0

            env = make_env(task)
            saved_paths = extract_initial_frames(env, task, task_id, img_id, LOCAL_SAVE_DIR)
            env.close()

            cam_urls = {}
            for cam, path in saved_paths.items():
                cam_urls[cam] = upload_to_huggingface(path, HF_USERNAME, HF_REPO_NAME)

            per_suite_task_to_url[task_key] = cam_urls

        task_to_url[task_name] = per_suite_task_to_url

        os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
        with open(OUTPUT_JSON, "a") as f:
            json.dump(task_to_url, f, indent=2)

    print(f"\n✅ 多视角初始帧已全部上传，JSON 保存至: {OUTPUT_JSON}")
    print(f"示例输出:\n{json.dumps(list(task_to_url.items())[:2], indent=2)}")


if __name__ == "__main__":
    main()

