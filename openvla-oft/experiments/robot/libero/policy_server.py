"""Serve OpenVLA(-OFT) policies over websocket for remote evaluation."""

from dataclasses import dataclass
import logging
from typing import Any, Dict, Optional
from enum import Enum
class TaskSuite(str, Enum):
    LIBERO_SPATIAL = "libero_spatial"
    LIBERO_OBJECT = "libero_object"
    LIBERO_GOAL = "libero_goal"
    LIBERO_10 = "libero_10"
    LIBERO_90 = "libero_90"

import draccus
import numpy as np
from openpi_client import base_policy as _base_policy

from experiments.robot.libero.constants import NUM_ACTIONS_CHUNK  # noqa: E402
@dataclass
class GenerateConfig:
    #################################################################################################################
    # Model-specific parameters (kept minimal but sufficient to run the server)
    #################################################################################################################
    model_family: str = "openvla"
    pretrained_checkpoint: str = "moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10"
    use_l1_regression: bool = True
    use_diffusion: bool = False
    num_diffusion_steps_inference: int = 50
    use_film: bool = False
    num_images_in_input: int = 2
    use_proprio: bool = True
    center_crop: bool = True
    num_open_loop_steps: int = NUM_ACTIONS_CHUNK
    lora_rank: int = 32
    unnorm_key: str = ""
    load_in_8bit: bool = False
    load_in_4bit: bool = False

    #################################################################################################################
    # Policy server parameters
    #################################################################################################################
    policy_server_host: str = "0.0.0.0"
    policy_server_port: int = 23451
    wait_for_policy_server: bool = True

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
# from experiments.robot.libero.run_libero_eval_server import (
#     # GenerateConfig,
#     # request_actions_from_policy,
#     validate_config,
# )
def validate_config(cfg: GenerateConfig) -> None:
    assert cfg.task_suite_name in [suite.value for suite in TaskSuite], f"Invalid task suite: {cfg.task_suite_name}"
    if cfg.num_open_loop_steps <= 0:
        raise ValueError("num_open_loop_steps must be positive")

def request_actions_from_policy(
    cfg: GenerateConfig,
    observation: dict,
    task_description: str,
    model,
    processor=None,
    action_head=None,
    proprio_projector=None,
    noisy_action_projector=None,
    policy_client=None,
):
    """Fetch an action chunk either from a local model or a remote policy server."""
    if policy_client is not None:
        payload = {
            "observation": observation,
            "task_description": task_description,
            "requested_chunk": cfg.num_open_loop_steps,
        }
        response = policy_client.infer(payload)
        actions = response.get("actions")
        if actions is None:
            raise ValueError("Remote policy server response is missing `actions` field.")
        return [np.asarray(action) for action in actions]

    from experiments.robot.robot_utils import get_action
    return get_action(
        cfg,
        model,
        observation,
        task_description,
        processor=processor,
        action_head=action_head,
        proprio_projector=proprio_projector,
        noisy_action_projector=noisy_action_projector,
        use_film=cfg.use_film,
    )

from experiments.robot.openvla_utils import (
    get_action_head,
    get_noisy_action_projector,
    get_processor,
    get_proprio_projector,
    resize_image_for_policy,
)
from experiments.robot.robot_utils import (
    DATE_TIME,
    get_action,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)
class PolicyExecutionMode(str, Enum):
    LOCAL = "local"
    REMOTE = "remote"
def check_unnorm_key(cfg: GenerateConfig, model) -> None:
    """Check that the model contains the action un-normalization key."""
    # Initialize unnorm_key
    unnorm_key = cfg.task_suite_name

    # In some cases, the key must be manually modified (e.g. after training on a modified version of the dataset
    # with the suffix "_no_noops" in the dataset name)
    if unnorm_key not in model.norm_stats and f"{unnorm_key}_no_noops" in model.norm_stats:
        unnorm_key = f"{unnorm_key}_no_noops"

    assert unnorm_key in model.norm_stats, f"Action un-norm key {unnorm_key} not found in VLA `norm_stats`!"

    # Set the unnorm_key in cfg
    cfg.unnorm_key = unnorm_key
def initialize_model(cfg: GenerateConfig):
    """Initialize model and associated components."""
    # Load model
    model = get_model(cfg)

    # Load proprio projector if needed
    proprio_projector = None
    if cfg.use_proprio:
        proprio_projector = get_proprio_projector(
            cfg,
            model.llm_dim,
            proprio_dim=8,  # 8-dimensional proprio for LIBERO
        )

    # Load action head if needed
    action_head = None
    if cfg.use_l1_regression or cfg.use_diffusion:
        action_head = get_action_head(cfg, model.llm_dim)

    # Load noisy action projector if using diffusion
    noisy_action_projector = None
    if cfg.use_diffusion:
        noisy_action_projector = get_noisy_action_projector(cfg, model.llm_dim)

    # Get OpenVLA processor if needed
    processor = None
    if cfg.model_family == "openvla":
        processor = get_processor(cfg)
        check_unnorm_key(cfg, model)

    return model, action_head, proprio_projector, noisy_action_projector, processor
from experiments.robot.robot_utils import (
    DATE_TIME,
    get_action,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)

from experiments.robot.libero.websocket_policy_server import WebsocketPolicyServer
from experiments.robot.robot_utils import get_image_resize_size, set_seed_everywhere

logger = logging.getLogger(__name__)


@dataclass
class PolicyServerConfig(GenerateConfig):
    """Extend evaluation config with server-specific settings."""

    metadata_note: Optional[str] = None

    def __post_init__(self):
        # Ensure we always run the local model when serving.
        self.policy_mode = PolicyExecutionMode.LOCAL


class OpenVLAPolicy(_base_policy.BasePolicy):
    """Adapter that exposes get_action() over the BasePolicy infer API."""

    def __init__(
        self,
        cfg: GenerateConfig,
        model,
        action_head=None,
        proprio_projector=None,
        noisy_action_projector=None,
        processor=None,
    ):
        self._cfg = cfg
        self._model = model
        self._action_head = action_head
        self._proprio_projector = proprio_projector
        self._noisy_action_projector = noisy_action_projector
        self._processor = processor
        self._resize_size = get_image_resize_size(cfg)

    def infer(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert OpenPI-style input format to OpenVLA format.
        
        OpenPI format (from client):
        {
            "observation/image": ...,
            "observation/wrist_image": ...,
            "observation/state": ...,
            "prompt": ...,
            "task_suite_name": ...  # Optional: for dynamic unnorm_key selection
        }
        
        OpenVLA format (expected by get_vla_action):
        {
            "full_image": ...,
            "wrist_image": ...,
            "state": ...
        }
        """
        # Check if payload is in OpenPI format (flat observation keys)
        if "observation/image" in payload or "observation/wrist_image" in payload:
            # Convert OpenPI format to OpenVLA format
            observation = {
                "full_image": payload.get("observation/image"),
                "wrist_image": payload.get("observation/wrist_image"),
                "state": payload.get("observation/state"),
            }
            task_description = payload.get("prompt", payload.get("task_description", ""))
        elif "observation" in payload:
            # Already in nested format (legacy OpenVLA format)
            observation = payload.get("observation")
            task_description = payload.get("task_description", "")
        else:
            raise ValueError(
                "Remote request must contain either OpenPI format keys "
                "('observation/image', 'observation/wrist_image', 'observation/state', 'prompt') "
                "or nested format ('observation', 'task_description')."
            )
        
        if observation is None:
            raise ValueError("Remote request missing observation data.")
        
        # Dynamically set unnorm_key based on task_suite_name from payload
        original_unnorm_key = self._cfg.unnorm_key
        task_suite_name = payload.get("task_suite_name")
        if task_suite_name:
            # Validate and set unnorm_key for this request
            unnorm_key = task_suite_name
            if unnorm_key not in self._model.norm_stats and f"{unnorm_key}_no_noops" in self._model.norm_stats:
                unnorm_key = f"{unnorm_key}_no_noops"
            if unnorm_key in self._model.norm_stats:
                self._cfg.unnorm_key = unnorm_key
            else:
                logging.warning(
                    f"Task suite {task_suite_name} not found in model norm_stats. "
                    f"Available keys: {list(self._model.norm_stats.keys())}. "
                    f"Using default unnorm_key: {original_unnorm_key}"
                )
        
        try:
            actions = request_actions_from_policy(
                self._cfg,
                observation,
                task_description,
                self._model,
                processor=self._processor,
                action_head=self._action_head,
                proprio_projector=self._proprio_projector,
                noisy_action_projector=self._noisy_action_projector,
                policy_client=None,
            )
        finally:
            # Restore original unnorm_key
            self._cfg.unnorm_key = original_unnorm_key

        return {
            "actions": [np.asarray(action, dtype=np.float32) for action in actions],
            "metadata": {"num_actions": len(actions)},
        }


def _build_metadata(cfg: PolicyServerConfig) -> Dict[str, Any]:
    metadata = {
        "model_family": cfg.model_family,
        "num_open_loop_steps": cfg.num_open_loop_steps,
        "note": cfg.metadata_note,
    }
    return {k: v for k, v in metadata.items() if v is not None}


@draccus.wrap()
def serve_policy(cfg: PolicyServerConfig) -> None:
    """CLI entrypoint to host an OpenVLA policy server."""
    validate_config(cfg)
    set_seed_everywhere(cfg.seed)

    model, action_head, proprio_projector, noisy_action_projector, processor = initialize_model(cfg)

    policy = OpenVLAPolicy(cfg, model, action_head, proprio_projector, noisy_action_projector, processor)
    metadata = _build_metadata(cfg)
    server = WebsocketPolicyServer(
        policy=policy,
        host=cfg.policy_server_host,
        port=cfg.policy_server_port,
        metadata=metadata,
    )

    logger.info(
        "Serving OpenVLA policy on %s:%s (metadata=%s)",
        cfg.policy_server_host,
        cfg.policy_server_port,
        metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    serve_policy()

