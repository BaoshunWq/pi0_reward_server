#!/usr/bin/env python3
"""
示例客户端 - 演示如何调用并行版本的Pi0 Reward Server
"""
import requests
import json
import time


def test_health(base_url):
    """测试健康检查端点"""
    print("=== 测试健康检查 ===")
    response = requests.get(f"{base_url}/health")
    print(f"状态码: {response.status_code}")
    print(f"响应: {response.text}")
    print()


def test_pool_status(base_url):
    """测试连接池状态"""
    print("=== 测试连接池状态 ===")
    response = requests.get(f"{base_url}/pool_status")
    print(f"状态码: {response.status_code}")
    print(f"响应: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
    print()


def test_score_single(base_url):
    """测试单个任务评分"""
    print("=== 测试单个任务评分 ===")
    
    payload = {
        "responses": [
            "pick up the black bowl and place it on the plate"
        ],
        "metas": [
            {
                "suite": "libero_spatial",
                "task_id": 0,
                "seed": 7
            }
        ],
        "reward_function_kwargs": {
            "num_trials_per_task": 1,  # 快速测试，只运行1次
            "max_workers": 1
        }
    }
    
    print(f"请求数据: {json.dumps(payload, indent=2, ensure_ascii=False)}")
    
    start_time = time.time()
    response = requests.post(
        f"{base_url}/score",
        json=payload,
        timeout=600  # 10分钟超时
    )
    elapsed = time.time() - start_time
    
    print(f"状态码: {response.status_code}")
    print(f"耗时: {elapsed:.2f}秒")
    print(f"响应: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
    print()


def test_score_parallel(base_url):
    """测试并行多任务评分"""
    print("=== 测试并行多任务评分 ===")
    
    payload = {
        "responses": [
            "put the robot toy on the shelf",
            "pick up the red bowl and place it on the rack",
            "move the plate to the left of the bowl",
            "put the toy car next to the plate"
        ],
        "metas": [
            {"suite": "libero_spatial", "task_id": 0, "seed": 7},
            {"suite": "libero_spatial", "task_id": 1, "seed": 7},
            {"suite": "libero_spatial", "task_id": 2, "seed": 7},
            {"suite": "libero_spatial", "task_id": 3, "seed": 7}
        ],
        "reward_function_kwargs": {
            "num_trials_per_task": 2,  # 每个任务2次trial
            "max_workers": 2  # 使用2个GPU并行
        }
    }
    
    print(f"请求数据: {len(payload['responses'])}个任务")
    print(f"并行度: {payload['reward_function_kwargs']['max_workers']}")
    
    start_time = time.time()
    response = requests.post(
        f"{base_url}/score",
        json=payload,
        timeout=3600  # 1小时超时
    )
    elapsed = time.time() - start_time
    
    print(f"状态码: {response.status_code}")
    print(f"总耗时: {elapsed:.2f}秒")
    print(f"平均每任务: {elapsed/len(payload['responses']):.2f}秒")
    
    if response.status_code == 200:
        result = response.json()
        print(f"成功率结果: {result.get('done_result', [])}")
    else:
        print(f"错误响应: {response.text}")
    print()


def test_autodl_public_access():
    """
    测试从其他服务器访问AutoDL上的服务
    需要先在AutoDL控制台配置端口映射
    """
    print("=== 测试AutoDL公网访问 ===")
    print("注意: 需要先在AutoDL控制台配置端口映射")
    print()
    
    # 替换为你的AutoDL公网地址
    autodl_url = input("请输入AutoDL公网地址 (例如 http://region-xxx.seetacloud.com:12345): ").strip()
    
    if not autodl_url:
        print("跳过AutoDL公网访问测试")
        return
    
    # 测试健康检查
    try:
        response = requests.get(f"{autodl_url}/health", timeout=10)
        print(f"✓ 连接成功!")
        print(f"状态: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"✗ 连接失败: {e}")
    print()


def main():
    # 本地测试
    LOCAL_URL = "http://localhost:6006"
    
    print("=" * 50)
    print("Pi0 Reward Server 并行版本 - 客户端测试")
    print("=" * 50)
    print()
    
    # 基础测试
    test_health(LOCAL_URL)
    test_pool_status(LOCAL_URL)
    
    # 选择测试模式
    print("请选择测试模式:")
    print("1. 快速测试 (单个任务)")
    print("2. 并行测试 (多个任务)")
    print("3. AutoDL公网访问测试")
    print("4. 全部测试")
    
    choice = input("\n请选择 [1-4]: ").strip()
    
    if choice == "1":
        test_score_single(LOCAL_URL)
    elif choice == "2":
        test_score_parallel(LOCAL_URL)
    elif choice == "3":
        test_autodl_public_access()
    elif choice == "4":
        test_score_single(LOCAL_URL)
        test_score_parallel(LOCAL_URL)
        test_autodl_public_access()
    else:
        print("无效选择")
    
    print("测试完成!")


if __name__ == "__main__":
    main()

