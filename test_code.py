import requests

URL = "http://127.0.0.1:6100/score"

payload = {
    "responses": [
        {"outputs":[{"text":"open the middle drawer of the cabinet"}]},
    ],
    "metas": [
        {
            "original_instruction": "open the middle drawer of the cabinet",
            "suite": "libero_goal",
            "task_id": 0,
            "seed": 7
        },
    ],
    "reward_function_kwargs": {
        "alpha": 1.0,
        "beta": 0.1,
        "gamma": 0.5,
        "num_trials_per_task": 1,
        "center_crop": 1,
        "libero_cfg": {
            "model_family": "openvla",
        }
    }
}

resp = requests.post(URL, json=payload, timeout=1800)
print(resp.status_code)
print(resp.json())