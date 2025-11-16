#!/usr/bin/env python
"""
测试双环境架构的完整工作流程
模拟客户端发送batch请求到reward服务器
"""
import requests
import json
import time
import argparse
from typing import List, Dict


def create_test_batch(num_samples: int = 10) -> Dict:
    """
    创建测试batch数据
    
    Args:
        num_samples: 样本数量
    
    Returns:
        完整的请求payload
    """
    responses = []
    metas = []
    
    # 一些测试指令变体
    test_instructions = [
        "put the red bowl on the left shelf",
        "place the red bowl onto the left shelf",
        "move the red bowl to the left shelf",
        "put the bowl on the shelf",
        "place the red container on the left side",
    ]
    
    for i in range(num_samples):
        # 循环使用测试指令
        instruction = test_instructions[i % len(test_instructions)]
        
        responses.append({
            "outputs": [{"text": instruction}]
        })
        
        metas.append({
            "original_instruction": "put the red bowl on the left shelf",
            "suite": "libero_spatial",
            "task_id": 0,
            "seed": i,
            "init_state_id": 0
        })
    
    payload = {
        "responses": responses,
        "metas": metas,
        "reward_function_kwargs": {
            "alpha": 1.0,
            "beta": 0.1,
            "gamma": 0.5,
            "num_trials_per_task": 1,  # 测试时使用1次trial
            "libero_cfg": {
                "model_family": "openvla",
                "task_suite_name": "libero_spatial",
                "task_id": 0,
                "host": "0.0.0.0",
                # port会由worker自动设置
            }
        }
    }
    
    return payload


def test_health_check(reward_url: str, policy_base_port: int, num_gpus: int):
    """
    测试所有服务器的健康状态
    
    Args:
        reward_url: Reward服务器URL
        policy_base_port: Policy服务器基础端口
        num_gpus: GPU数量
    """
    print("=" * 60)
    print("健康检查")
    print("=" * 60)
    
    # 测试reward服务器
    print(f"\n[1/2] 检查Reward服务器: {reward_url}")
    try:
        resp = requests.get(f"{reward_url.rsplit('/', 1)[0]}/health", timeout=5)
        if resp.status_code == 200:
            print(f"  ✅ Reward服务器正常")
        else:
            print(f"  ❌ Reward服务器异常: {resp.status_code}")
            return False
    except Exception as e:
        print(f"  ❌ 无法连接Reward服务器: {e}")
        return False
    
    # 测试policy服务器
    print(f"\n[2/2] 检查Policy服务器 (端口 {policy_base_port} - {policy_base_port + num_gpus - 1})")
    all_ok = True
    for i in range(num_gpus):
        port = policy_base_port + i
        try:
            # 注意：policy服务器可能没有/health端点，尝试连接即可
            import socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2)
            result = sock.connect_ex(('localhost', port))
            sock.close()
            
            if result == 0:
                print(f"  ✅ GPU {i} Policy服务器 (port {port}) 正常")
            else:
                print(f"  ❌ GPU {i} Policy服务器 (port {port}) 无法连接")
                all_ok = False
        except Exception as e:
            print(f"  ❌ GPU {i} Policy服务器检查失败: {e}")
            all_ok = False
    
    print()
    return all_ok


def test_batch_request(reward_url: str, num_samples: int = 10, timeout: int = 300):
    """
    测试batch请求
    
    Args:
        reward_url: Reward服务器URL
        num_samples: 样本数量
        timeout: 超时时间（秒）
    """
    print("=" * 60)
    print(f"测试Batch请求 ({num_samples} 个样本)")
    print("=" * 60)
    
    # 创建测试数据
    print(f"\n[1/3] 创建测试数据...")
    payload = create_test_batch(num_samples)
    print(f"  ✅ 创建了 {len(payload['responses'])} 个样本")
    
    # 发送请求
    print(f"\n[2/3] 发送请求到 {reward_url}")
    print(f"  超时设置: {timeout}秒")
    
    start_time = time.time()
    
    try:
        resp = requests.post(
            reward_url,
            json=payload,
            timeout=timeout,
            headers={"Accept": "application/json"}
        )
        
        elapsed = time.time() - start_time
        
        print(f"  ✅ 请求完成，耗时: {elapsed:.2f}秒")
        print(f"  平均每样本: {elapsed/num_samples:.2f}秒")
        
        # 检查响应
        resp.raise_for_status()
        
    except requests.exceptions.Timeout:
        print(f"  ❌ 请求超时 (>{timeout}秒)")
        return False
    
    except requests.exceptions.RequestException as e:
        print(f"  ❌ 请求失败: {e}")
        if hasattr(e, 'response') and e.response is not None:
            try:
                error_data = e.response.json()
                print(f"  错误详情: {json.dumps(error_data, indent=2)}")
            except:
                print(f"  响应内容: {e.response.text[:500]}")
        return False
    
    # 解析结果
    print(f"\n[3/3] 解析结果...")
    try:
        data = resp.json()
        done_result = data.get("done_result", [])


        print(f"response data: {data}")
        
        if not isinstance(done_result, list):
            print(f"  ❌ done_result不是列表: {type(done_result)}")
            return False
        
        if len(done_result) != num_samples:
            print(f"  ⚠️  结果数量不匹配: 期望{num_samples}, 实际{len(done_result)}")
        
        print(f"  ✅ 收到 {len(done_result)} 个结果")
        
        # 显示结果统计
        if done_result:
            avg_success = sum(done_result) / len(done_result)
            max_success = max(done_result)
            min_success = min(done_result)
            
            print(f"\n  结果统计:")
            print(f"    平均成功率: {avg_success:.2%}")
            print(f"    最高成功率: {max_success:.2%}")
            print(f"    最低成功率: {min_success:.2%}")
            
            # 显示前5个结果
            print(f"\n  前5个样本结果:")
            for i, result in enumerate(done_result[:5]):
                instruction = payload['responses'][i]['outputs'][0]['text']
                print(f"    [{i}] {instruction[:40]:40s} → 成功率: {result:.2%}")
            
            if len(done_result) > 5:
                print(f"    ... 还有 {len(done_result) - 5} 个结果")
        
        return True
        
    except Exception as e:
        print(f"  ❌ 解析结果失败: {e}")
        print(f"  响应内容: {resp.text[:500]}")
        return False


def main():
    parser = argparse.ArgumentParser(description="测试双环境架构")
    parser.add_argument(
        "--reward_url",
        type=str,
        default="http://localhost:45679/score",
        help="Reward服务器URL"
    )
    parser.add_argument(
        "--policy_base_port",
        type=int,
        default=45678,
        help="Policy服务器基础端口"
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=2,
        help="GPU数量"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10,
        help="测试样本数量"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="请求超时时间（秒）"
    )
    parser.add_argument(
        "--skip_health",
        action="store_true",
        help="跳过健康检查"
    )
    
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("双环境架构测试工具")
    print("=" * 60)
    print(f"Reward URL: {args.reward_url}")
    print(f"Policy端口: {args.policy_base_port} - {args.policy_base_port + args.num_gpus - 1}")
    print(f"GPU数量: {args.num_gpus}")
    print(f"测试样本: {args.num_samples}")
    print("=" * 60)
    
    # 健康检查
    if not args.skip_health:
        if not test_health_check(args.reward_url, args.policy_base_port, args.num_gpus):
            print("\n❌ 健康检查失败，请确保所有服务器已启动")
            print("\n提示: 启动服务器命令:")
            print("  bash scripts/unify_parallel.sh")
            return 1
    else:
        print("\n⚠️  跳过健康检查")
    
    # 测试batch请求
    print()
    success = test_batch_request(args.reward_url, args.num_samples, args.timeout)
    
    # 总结
    print("\n" + "=" * 60)
    if success:
        print("✅ 测试通过！系统工作正常")
        print("=" * 60)
        return 0
    else:
        print("❌ 测试失败！请检查日志")
        print("=" * 60)
        print("\n调试建议:")
        print("  1. 查看policy服务器日志: tail -f server_policy_gpu*.log")
        print("  2. 查看reward服务器输出")
        print("  3. 检查GPU内存: nvidia-smi")
        print("  4. 减少样本数量重试: --num_samples 2")
        return 1


if __name__ == "__main__":
    exit(main())
