#!/usr/bin/env python3
"""
测试内存泄漏修复的脚本
运行此脚本来验证内存使用是否正常
"""
import psutil
import requests
import time
import json

def get_memory_usage():
    """获取当前进程内存使用（MB）"""
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if 'app_pi0_libero' in ' '.join(proc.info['cmdline'] or []):
                mem_mb = proc.memory_info().rss / (1024 * 1024)
                return proc.info['pid'], mem_mb
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return None, None

def test_reward_server(num_requests=5):
    """测试 reward server 的内存使用"""
    server_url = "http://0.0.0.0:34567"
    
    # 检查服务器是否运行
    try:
        response = requests.get(f"{server_url}/health", timeout=5)
        if response.status_code != 200:
            print("❌ Reward server is not running!")
            print("Please start it with: python pi0_reward_server/app_pi0_libero.py")
            return
    except requests.exceptions.RequestException as e:
        print(f"❌ Cannot connect to reward server: {e}")
        print("Please start it with: python pi0_reward_server/app_pi0_libero.py")
        return
    
    print(f"✅ Reward server is running at {server_url}")
    print("\n" + "="*60)
    print("Testing memory usage...")
    print("="*60 + "\n")
    
    # 测试请求
    test_payload = {
        "responses": ["pick up the red bowl and place it on the shelf"],
        "metas": [{
            "suite": "libero_spatial",
            "task_id": 0,
            "seed": 0,
        }],
        "reward_function_kwargs": {
            "num_trials_per_task": 1,  # 只测试1次，加快速度
            "libero_cfg": {
                "save_videos": False,  # 禁用视频保存
            }
        }
    }
    
    memory_readings = []
    
    for i in range(num_requests):
        pid, mem_before = get_memory_usage()
        
        if pid is None:
            print(f"⚠️  Cannot find reward server process")
            break
        
        print(f"Request {i+1}/{num_requests}:")
        print(f"  PID: {pid}")
        print(f"  Memory before: {mem_before:.2f} MB")
        
        try:
            start_time = time.time()
            response = requests.post(
                f"{server_url}/score",
                json=test_payload,
                timeout=300
            )
            elapsed_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                print(f"  ✅ Request successful ({elapsed_time:.1f}s)")
                print(f"  Result: {result.get('done_result', [])}")
            else:
                print(f"  ❌ Request failed: {response.status_code}")
                print(f"  Error: {response.text[:200]}")
        except Exception as e:
            print(f"  ❌ Request error: {e}")
        
        # 等待垃圾回收
        time.sleep(2)
        
        _, mem_after = get_memory_usage()
        if mem_after:
            print(f"  Memory after: {mem_after:.2f} MB")
            print(f"  Δ Memory: {mem_after - mem_before:+.2f} MB")
            memory_readings.append((mem_before, mem_after))
        
        print()
    
    # 分析结果
    if len(memory_readings) >= 2:
        print("="*60)
        print("Memory Analysis:")
        print("="*60)
        
        initial_mem = memory_readings[0][0]
        final_mem = memory_readings[-1][1]
        total_increase = final_mem - initial_mem
        avg_increase = total_increase / len(memory_readings)
        
        print(f"Initial memory: {initial_mem:.2f} MB")
        print(f"Final memory: {final_mem:.2f} MB")
        print(f"Total increase: {total_increase:+.2f} MB")
        print(f"Average per request: {avg_increase:+.2f} MB")
        
        # 判断是否有内存泄漏
        if avg_increase > 100:
            print("\n❌ SEVERE MEMORY LEAK DETECTED!")
            print("   Memory increase > 100 MB per request")
        elif avg_increase > 50:
            print("\n⚠️  POSSIBLE MEMORY LEAK")
            print("   Memory increase > 50 MB per request")
        elif avg_increase > 10:
            print("\n⚠️  MINOR MEMORY INCREASE")
            print("   Some memory not being freed (might be normal for first few requests)")
        else:
            print("\n✅ MEMORY USAGE LOOKS GOOD!")
            print("   No significant memory leak detected")
        
        print("\n" + "="*60)
        print("Recommendations:")
        print("="*60)
        if avg_increase > 50:
            print("1. Check if env.close() is being called")
            print("2. Check if WebSocket connections are closed")
            print("3. Verify gc.collect() is running")
            print("4. Consider restarting the server")
        else:
            print("✅ Memory management appears to be working correctly")
            print("   You can now safely use the server for training")

if __name__ == "__main__":
    import sys
    
    num_requests = 5
    if len(sys.argv) > 1:
        try:
            num_requests = int(sys.argv[1])
        except ValueError:
            print(f"Invalid number of requests: {sys.argv[1]}")
            sys.exit(1)
    
    print("="*60)
    print("Pi0 Reward Server Memory Test")
    print("="*60)
    print(f"This will send {num_requests} test requests to the reward server")
    print("and monitor memory usage to detect leaks.\n")
    
    test_reward_server(num_requests)

