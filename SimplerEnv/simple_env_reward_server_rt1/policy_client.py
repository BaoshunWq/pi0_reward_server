import base64
import requests
import numpy as np


class HttpPolicyClient:
    def __init__(self, host: str, port: int):
        self.base = f"http://{host}:{port}"

    def reset(self, session_id: str, task_description: str, policy_setup: str = "google_robot", action_scale: float = 1.0, saved_model_path: str = "rt_1_x_tf_trained_for_002272480_step"):
        payload = {
            "session_id": session_id,
            "task_description": task_description,
            "policy_setup": policy_setup,
            "action_scale": action_scale,
            "saved_model_path": saved_model_path,
        }
        r = requests.post(f"{self.base}/reset", json=payload, timeout=300)
        r.raise_for_status()
        return r.json()

    def infer(self, session_id: str, image: np.ndarray, task_description: str | None = None):
        payload = {"session_id": session_id}
        payload["image_nd"] = {
            "__numpy__": base64.b64encode(image.tobytes()).decode("ascii"),
            "dtype": str(image.dtype),
            "shape": list(image.shape),
        }
        if task_description is not None:
            payload["task_description"] = task_description
        r = requests.post(f"{self.base}/infer", json=payload, timeout=300)
        r.raise_for_status()
        return r.json()
