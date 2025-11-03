import os
os.environ['CUDA_VISIBLE_DEVICES'] = '5'
import numpy as np
# ---- compatibility patch for numpy>=1.24 ----
if not hasattr(np, "float"):
    np.float = float
if not hasattr(np, "int"):
    np.int = int
if not hasattr(np, "bool"):
    np.bool = bool
# ---------------------------------------------

import argparse
from collections import Counter, defaultdict
import logging
import os
from pathlib import Path
import sys
import time
from openvla.experiments.robot.openvla_utils import get_processor
from openvla.experiments.robot.robot_utils import get_model
# This is for using the locally installed repo clone when using slurm
from calvin.calvin_models.calvin_agent.models.calvin_base_model import CalvinBaseModel

sys.path.insert(0, Path(__file__).absolute().parents[2].as_posix())
from dataclasses import dataclass
import draccus
from calvin.calvin_models.calvin_agent.evaluation.multistep_sequences import get_sequences
from calvin.calvin_models.calvin_agent.evaluation.utils import (
    collect_plan,
    count_success,
    create_tsne,
    get_default_model_and_env,
    get_env_state_for_initial_condition,
    get_log_dir,
    join_vis_lang,
    print_and_save,
)
from calvin.calvin_models.calvin_agent.utils.utils import get_all_checkpoints, get_checkpoints_for_epochs, get_last_checkpoint
import hydra
import numpy as np
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from termcolor import colored
from tqdm.auto import tqdm

from calvin.calvin_env.calvin_env.envs.play_table_env import get_env

from openvla.experiments.robot.robot_utils import (
    DATE_TIME,
    get_action,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)
from openvla.experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    resize_image,
    get_libero_image,
    quat2axisangle,
    save_rollout_video,
)
from typing import Optional, Union

logger = logging.getLogger(__name__)

EP_LEN = 360
NUM_SEQUENCES = 100
from datetime import datetime

# 获取当前时间
now = datetime.now()
NOW_TIME_STR = now.strftime("%Y-%m-%d_%H-%M-%S")
@dataclass
class GenerateConfig:
    # fmt: off

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "openvla"                    # Model family
    pretrained_checkpoint: Union[str, Path] = "openvla/openvla-7b"     # Pretrained checkpoint path
    load_in_8bit: bool = False                       # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                       # (For OpenVLA only) Load with 4-bit quantization

    center_crop: bool = True                         # Center crop? (if trained w/ random crop image aug)

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = "bridge_orig"          # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    num_steps_wait: int = 10                         # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 1                    # Number of rollouts per task

    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note: Optional[str] = None                # Extra note to add in run ID for logging
    local_log_dir: str = "./experiments/logs"        # Local directory for eval logs

    use_wandb: bool = False                          # Whether to also log results in Weights & Biases
    wandb_project: str = "redTeamingOpenvla"        # Name of W&B project to log to (use default!)
    wandb_entity: str = "tongbs-sysu"          # Name of entity to log under

    seed: int = 7                                    # Random Seed (for reproducibility)

    # fmt: on
    # user-customized config

    is_save_video: bool = False                       # Whether to save rollout video

    task_to_huglinks_json_path: str = "vlm_initial_state_links.json"  # JSON file mapping task names to image URLs

    examples_path: str = ""                          # YAML file with existing task->examples (optional)

    redTeaming_vlm_model = "qwen2.5-vl-72b-instruct"               # VLM model for instruction generation (e.g. gpt-4, gpt-3.5-turbo)

    num_instructions : int = 10                     # Number of instructions to request per generation

    failure_threshold: float = 0.5               # Success rate <= threshold considered a failure

    output_path: str = f"./output/{NOW_TIME_STR}/redteaming_results.json"  # Path to save the attack results

        # CALVIN evaluation-specific parameters (来自 args)
    #################################################################################################################
    dataset_path: str = "calvin/dataset/calvin_debug_dataset"   # --dataset_path
    train_folder: Optional[str] = None                           # --train_folder
    checkpoints: Optional[str] = None                            # --checkpoints
    checkpoint: Optional[str] = None                             # --checkpoint
    last_k_checkpoints: Optional[int] = None                     # --last_k_checkpoints
    custom_model: bool = True                                    # --custom_model
    debug: bool = False                                          # --debug
    eval_log_dir: Optional[str] = None                           # --eval_log_dir
    device: int = 0                                              # --device


def get_calvin_image(obs, resize_size):
    """Extracts image from observations and preprocesses it."""
    assert isinstance(resize_size, int) or isinstance(resize_size, tuple)
    if isinstance(resize_size, int):
        resize_size = (resize_size, resize_size)
    img = obs['rgb_obs']['rgb_static']
    img = img[::-1, ::-1]  # IMPORTANT: rotate 180 degrees to match train preprocessing
    img = resize_image(img, resize_size)
    return img

def get_epoch(checkpoint):
    if "=" not in checkpoint.stem:
        return "0"
    checkpoint.stem.split("=")[1]


def make_env(dataset_path):
    val_folder = Path(dataset_path) / "validation"
    env = get_env(val_folder, show_gui=False)

    # insert your own env wrapper
    # env = Wrapper(env)
    return env


class CustomModel(CalvinBaseModel):
    def __init__(self,cfg,):
        logger.warning("Please implement these methods as an interface to your custom model architecture.")
        processor = None
        if cfg.model_family == "openvla":
            self.processor = get_processor(cfg)
        self.openvla_model = get_model(cfg)
        self.cfg = cfg

    # def reset(self):
    #     """
    #     This is called
    #     """
    #     # raise NotImplementedError

    def step(self, obs, goal):
        """
        Args:
            obs: environment observations
            goal: embedded language goal
        Returns:
            action: predicted action
        """
        # obs['rgb_obs']['rgb_static'].shape.           (200, 200, 3)

        img = get_calvin_image(obs , get_image_resize_size(self.cfg))
        observation = {
                    "full_image": img,
                    # "state": np.concatenate(
                    #     (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
                    # ),
                }
        
        task_description = goal

        action = get_action(
                    self.cfg,
                    self.openvla_model,
                    observation,
                    task_description,
                    processor=self.processor,
                )
        
        action = normalize_gripper_action(action, binarize=True)

                # [OpenVLA] The dataloader flips the sign of the gripper action to align with other datasets
                # (0 = close, 1 = open), so flip it back (-1 = open, +1 = close) before executing the action
        # if self.cfg.model_family == "openvla":
        #     action = invert_gripper_action(action)

        return action

def evaluate_policy(model, env, epoch=1, eval_log_dir=None, debug=False, create_plan_tsne=False):
    """
    Run this function to evaluate a model on the CALVIN challenge.

    Args:
        model: Must implement methods of CalvinBaseModel.
        env: (Wrapped) calvin env.
        epoch:
        eval_log_dir: Path where to log evaluation results. If None, logs to /tmp/evaluation/
        debug: If True, show camera view and debug info.
        create_plan_tsne: Collect data for TSNE plots of latent plans (does not work for your custom model)

    Returns:
        Dictionary with results
    """
    conf_dir = "calvin/calvin_models/conf"
    task_cfg = OmegaConf.load("calvin/calvin_env/conf/tasks/new_playtable_tasks.yaml")
    task_oracle = hydra.utils.instantiate(task_cfg)
    val_annotations = OmegaConf.load("calvin/calvin_models/conf/annotations/new_playtable_validation.yaml")

    eval_log_dir = get_log_dir(eval_log_dir)

    eval_sequences = get_sequences(NUM_SEQUENCES,num_workers=64)

    results = []
    plans = defaultdict(list)

    if not debug:
        eval_sequences = tqdm(eval_sequences, position=0, leave=True)

    for initial_state, eval_sequence in eval_sequences:
        result = evaluate_sequence(env, model, task_oracle, initial_state, eval_sequence, val_annotations, plans, debug)
        results.append(result)
        if not debug:
            eval_sequences.set_description(
                " ".join([f"{i + 1}/5 : {v * 100:.1f}% |" for i, v in enumerate(count_success(results))]) + "|"
            )

    if create_plan_tsne:
        create_tsne(plans, eval_log_dir, epoch)
    print_and_save(results, eval_sequences, eval_log_dir, epoch)

    return results


def evaluate_sequence(env, model, task_checker, initial_state, eval_sequence, val_annotations, plans, debug):
    """
    Evaluates a sequence of language instructions.
    """
    robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
    env.reset(robot_obs=robot_obs, scene_obs=scene_obs)

    success_counter = 0
    if debug:
        time.sleep(1)
        print()
        print()
        print(f"Evaluating sequence: {' -> '.join(eval_sequence)}")
        print("Subtask: ", end="")
    for subtask in eval_sequence:
        success = rollout(env, model, task_checker, subtask, val_annotations, plans, debug)
        if success:
            success_counter += 1
        else:
            return success_counter
    return success_counter


def rollout(env, model, task_oracle, subtask, val_annotations, plans, debug):
    """
    Run the actual rollout on one subtask (which is one natural language instruction).
    """
    if debug:
        print(f"{subtask} ", end="")
        time.sleep(0.5)
    obs = env.get_obs()
    # get lang annotation for subtask
    lang_annotation = val_annotations[subtask][0]
    # model.reset()
    start_info = env.get_info()

    for step in range(EP_LEN):
        action = model.step(obs, lang_annotation)
        obs, _, _, current_info = env.step(action)
        if debug:
            img = env.render(mode="rgb_array")
            join_vis_lang(img, lang_annotation)
            # time.sleep(0.1)
        if step == 0:
            # for tsne plot, only if available
            collect_plan(model, plans, subtask)

        # check if current step solves a task
        current_task_info = task_oracle.get_task_info_for_set(start_info, current_info, {subtask})
        if len(current_task_info) > 0:
            if debug:
                print(colored("success", "green"), end=" ")
            return True
    if debug:
        print(colored("fail", "red"), end=" ")
    return False

# -------------- 主循环 --------------
@draccus.wrap()
def main(args: GenerateConfig):
    seed_everything(0, workers=True)  # type:ignore
    # parser = argparse.ArgumentParser(description="Evaluate a trained model on multistep sequences with language goals.")
    # parser.add_argument("--dataset_path", type=str, default="calvin/dataset/calvin_debug_dataset", help="Path to the dataset root directory.")

    # # arguments for loading default model
    # parser.add_argument(
    #     "--train_folder", type=str, help="If calvin_agent was used to train, specify path to the log dir."
    # )
    # parser.add_argument(
    #     "--checkpoints",
    #     type=str,
    #     default=None,
    #     help="Comma separated list of epochs for which checkpoints will be loaded",
    # )
    # parser.add_argument(
    #     "--checkpoint",
    #     type=str,
    #     default=None,
    #     help="Path of the checkpoint",
    # )
    # parser.add_argument(
    #     "--last_k_checkpoints",
    #     type=int,
    #     help="Specify the number of checkpoints you want to evaluate (starting from last). Only used for calvin_agent.",
    # )

    # # arguments for loading custom model or custom language embeddings
    # parser.add_argument(
    #     "--custom_model", action="store_true",default=True, help="Use this option to evaluate a custom model architecture."
    # )

    # parser.add_argument("--debug", action="store_true", help="Print debug info and visualize environment.")

    # parser.add_argument("--eval_log_dir", default=None, type=str, help="Where to log the evaluation results.")

    # parser.add_argument("--device", default=0, type=int, help="CUDA device")
    # args = parser.parse_args()

    # evaluate a custom model

    if args.custom_model:
        model = CustomModel(args)
        if args.model_family == "openvla":
            args.unnorm_key = args.task_suite_name
            # In some cases, the key must be manually modified (e.g. after training on a modified version of the dataset
            # with the suffix "_no_noops" in the dataset name)
            if args.unnorm_key not in model.openvla_model.norm_stats and f"{args.unnorm_key}_no_noops" in model.openvla_model.norm_stats:
                args.unnorm_key = f"{args.unnorm_key}_no_noops"
            assert args.unnorm_key in model.openvla_model.norm_stats, f"Action un-norm key {args.unnorm_key} not found in VLA `norm_stats`!"

        env = make_env(args.dataset_path)
        evaluate_policy(model, env, debug=args.debug)
    else:
        assert "train_folder" in args

        checkpoints = []
        if args.checkpoints is None and args.last_k_checkpoints is None and args.checkpoint is None:
            print("Evaluating model with last checkpoint.")
            checkpoints = [get_last_checkpoint(Path(args.train_folder))]
        elif args.checkpoints is not None:
            print(f"Evaluating model with checkpoints {args.checkpoints}.")
            checkpoints = get_checkpoints_for_epochs(Path(args.train_folder), args.checkpoints)
        elif args.checkpoints is None and args.last_k_checkpoints is not None:
            print(f"Evaluating model with last {args.last_k_checkpoints} checkpoints.")
            checkpoints = get_all_checkpoints(Path(args.train_folder))[-args.last_k_checkpoints :]
        elif args.checkpoint is not None:
            checkpoints = [Path(args.checkpoint)]

        env = None
        for checkpoint in checkpoints:
            epoch = get_epoch(checkpoint)
            model, env, _ = get_default_model_and_env(
                args.train_folder,
                args.dataset_path,
                checkpoint,
                env=env,
                device_id=args.device,
            )
            evaluate_policy(model, env, epoch, eval_log_dir=args.eval_log_dir, debug=args.debug, create_plan_tsne=True)


if __name__ == "__main__":
    main()
