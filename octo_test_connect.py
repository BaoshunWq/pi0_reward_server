import requests

URL = "http://127.0.0.1:6101/score"

# 测试所有 suite
test_suites = [
    "libero_spatial",
    # "libero_object", 
    # "libero_goal",
    # "libero_10",
]

for suite in test_suites:
    print(f"\n{'='*60}")
    print(f"测试 Suite: {suite}")
    print(f"{'='*60}")
    
    payload = {
            "responses": [{"outputs":[{"text":"put the carrot on the plate"}]}],
            "metas": [{"env_name":"PutCarrotOnPlateInScene-v0","robot":"widowx","init_state_id":0}],
            "reward_function_kwargs": {
                "num_trials_per_task": 1,
                "simple_env_cfg": {"port":8001,"policy_setup":"widowx_bridge","model_type":"octo-base"}
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




