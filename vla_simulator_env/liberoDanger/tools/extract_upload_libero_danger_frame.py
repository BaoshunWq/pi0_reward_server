import sys, os

os.environ["LIBERO_CONFIG_PATH"] = "vla_simulator_env/liberoDanger/libero/configs"
sys.path.insert(0, "vla_simulator_env/liberoDanger")
import json
import numpy as np
from PIL import Image
from tqdm import tqdm
from libero.libero import benchmark
from huggingface_hub import HfApi, HfFolder, upload_file, create_repo
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv



# ========== 用户需修改的部分 ==========
HF_USERNAME = "TBS2001"                # 你的 HF 用户名
HF_REPO_NAME = "libero_init_frame_with_danger_zone"          # 数据集仓库名

TASK_LIST = [
    'libero_spatial', 
    'libero_object', 
    'libero_goal', 
    'libero_10'
    # 'libero_90', 

]
LOCAL_SAVE_DIR = "libero-init-frames_with_danger_zone"
OUTPUT_JSON = "libero-init-frames_with_danger_zone/json_data_for_rl/vlm_initial_state_links.json"
# =====================================


def get_libero_env(task, resolution=384):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(0)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description

def make_env(task: str):
    """创建一个 LIBERO 环境实例"""

    # Initialize LIBERO environment and task description
    env, task_description = get_libero_env(task, resolution=384)

    return env

def extract_initial_frame(env, task_name: str, task_id,img_id, save_dir: str) -> str:

    #  img_id 用于区分同一任务下的不同初始状态图像，同一任务可能会初始多次
    """reset 环境并保存第 0 帧（初始观测图像）"""
    obs = env.reset()
    img = None
    if isinstance(obs, dict):
        k = "agentview_image" # only applicable to libero
        img = obs[k]

    elif isinstance(obs, (list, tuple)):
        img = obs[0]
    else:
        raise ValueError("未知 obs 格式，请检查 env.reset() 输出。")

    if img is None:
        raise ValueError("在 obs 中找不到 RGB 图像。")
    
    task_folder = task_name.problem_folder  # libero10
    
    # task_language = task_name.language.lower().replace(" ", "_")

    file_name = f"{task_folder}_task-{task_id}_img-{img_id}_frame0.png"

    # task_key = task_folder + "_" + task_language

    img = np.array(img, dtype=np.uint8)
#     img = np.flipud(img)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, file_name)
    Image.fromarray(img).save(save_path)

    return save_path

def upload_to_huggingface(local_path: str, username: str, repo_name: str) -> str:
    """上传到 HuggingFace Hub 并返回文件的原始 URL"""
    api = HfApi()
    repo_id = f"{username}/{repo_name}"
    # 若仓库不存在则创建
    try:
        create_repo(repo_id, repo_type="dataset", private=False)
    except Exception:
        pass  # 已存在则忽略

    remote_path = f"{os.path.basename(local_path)}"
    # upload_file(
    #     path_or_fileobj=local_path,
    #     path_in_repo=remote_path,
    #     repo_id=repo_id,
    #     repo_type="dataset",
    #     token=HfFolder.get_token(),
    # )
    # 构造可直接访问的 raw 文件链接
    return f"https://huggingface.co/datasets/{repo_id}/resolve/main/{remote_path}"

def main():
    os.makedirs(LOCAL_SAVE_DIR, exist_ok=True)
    # Ensure OUTPUT_JSON parent directory exists
    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    task_to_url = {}

    task_name_list = TASK_LIST  # 可自定义任务列表

    # task_name = "libero_10"  # Optional: 'libero_spatial', 'libero_object', 'libero_goal', 'libero_90', 'libero_10', 'libero_100'

        # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()


    for task_name in task_name_list:

        per_suite_task_to_url = {}
        task_suite = benchmark_dict[task_name]()
        num_tasks_in_suite = task_suite.n_tasks

        for task_id in tqdm(range(num_tasks_in_suite)):
            # Get task
            task = task_suite.get_task(task_id)

            task_key = task.language.lower().replace(" ", "_")

            # # Get default LIBERO initial states
            # initial_states = task_suite.get_task_init_states(task_id)

            img_id = 0  # 用于区分同一任务下的不同初始状态图像，同一任务可能会初始多次

            env = make_env(task)
            save_path = extract_initial_frame(env, task,task_id,img_id, LOCAL_SAVE_DIR)
            env.close()

            # image_url = upload_to_huggingface(save_path, HF_USERNAME, HF_REPO_NAME)
            per_suite_task_to_url[task_key] = [save_path]
        
        task_to_url[task_name] = per_suite_task_to_url

        with open(OUTPUT_JSON, "a") as f:
            json.dump(task_to_url, f, indent=2)

    print(f"\n✅ 初始帧已全部上传，JSON 保存至: {OUTPUT_JSON}")
    print(f"示例输出:\n{json.dumps(list(task_to_url.items())[:2], indent=2)}")

if __name__ == "__main__":
    main()
