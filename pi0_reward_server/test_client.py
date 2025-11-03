#!/usr/bin/env python3
"""
测试客户端 - 用于从其他服务器（如 autodl）测试连接
"""

import requests
import sys
import json

def test_connection(server_ip, port=34567):
    """测试服务器连接"""
    base_url = f"http://{server_ip}:{port}"
    
    print(f"🔍 Testing connection to {base_url}")
    print("=" * 60)
    
    # 1. 健康检查
    print("\n1️⃣  Health Check:")
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            print(f"   ✅ Server is healthy: {response.text}")
        else:
            print(f"   ❌ Unexpected response: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"   ❌ Connection failed: {e}")
        print("\n💡 Troubleshooting tips:")
        print("   - Check if the server is running")
        print("   - Check firewall rules: sudo ufw allow 34567/tcp")
        print("   - Check cloud security group settings")
        print("   - Verify the IP address is correct")
        return False
    
    # 2. 测试评分接口
    print("\n2️⃣  Score API Test:")
    test_data = {
        "responses": [
            {"action": "move_forward", "value": 0.5}
        ],
        "metas": [
            {
                "original_instruction": "test task",
                "suite": "libero_object",
                "task_id": 0,
                "seed": 0,
                "init_state_id": 0
            }
        ],
        "reward_function_kwargs": {
            "alpha": 1.0,
            "beta": 0.1,
            "gamma": 0.5,
            "num_trials_per_task": 1
        }
    }
    
    try:
        response = requests.post(
            f"{base_url}/score",
            json=test_data,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"   ✅ Score API working!")
            print(f"   Response: {json.dumps(result, indent=2)}")
        else:
            print(f"   ⚠️  Response code: {response.status_code}")
            print(f"   Response: {response.text}")
            
    except requests.exceptions.RequestException as e:
        print(f"   ❌ Request failed: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("✅ All tests passed! Server is accessible.")
    return True


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_client.py <server_ip> [port]")
        print("\nExample:")
        print("  python test_client.py 192.168.1.100")
        print("  python test_client.py 192.168.1.100 34567")
        sys.exit(1)
    
    server_ip = sys.argv[1]
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 34567
    
    success = test_connection(server_ip, port)
    sys.exit(0 if success else 1)



