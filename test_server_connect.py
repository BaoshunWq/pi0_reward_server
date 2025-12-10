import requests

URL = "http://127.0.0.1:6000/score"

# 测试所有 suite
test_suites = [
    # "libero_spatial",
    # "libero_object", 
    # "libero_goal",
    "libero_10",
    # "libero_90"
]

for suite in test_suites:
    print(f"\n{'='*60}")
    print(f"测试 Suite: {suite}")
    print(f"{'='*60}")
    
    payload = {
        "responses": [
            {"outputs":[{"text":"put both the alphabet soup and the tomato sauce in the basket"}]},
        ],
        "metas": [
            {
                "original_instruction": "put both the alphabet soup and the tomato sauce in the basket",
                "suite": suite,
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
    
    try:
        resp = requests.post(URL, json=payload, timeout=1800)
        print(f"状态码: {resp.status_code}")
        result = resp.json()
        print(f"结果: {result}")
        if isinstance(result, list) and len(result) > 0:
            print(f"成功率: {result[0]}")
    except Exception as e:
        print(f"错误: {e}")

print(f"\n{'='*60}")
print("所有 suite 测试完成")
print(f"{'='*60}")