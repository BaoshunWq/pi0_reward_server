# 🚀 快速开始指南

## 让你的服务器支持跨服务器访问（如 autodl）

### 第一步：安装依赖

```bash
cd /home/baoshuntong/code/saftyEmbodyAI/redTeam_pi0/pi0_reward_server
pip install flask-cors
```

或者安装所有依赖：

```bash
pip install -r requirements.txt
```

### 第二步：配置防火墙

运行自动配置脚本：

```bash
./setup_firewall.sh
```

或者手动配置：

```bash
# Ubuntu/Debian
sudo ufw allow 34567/tcp
sudo ufw status

# CentOS/RHEL
sudo firewall-cmd --permanent --add-port=34567/tcp
sudo firewall-cmd --reload
```

### 第三步：获取服务器 IP 地址

```bash
# 查看本机 IP
hostname -I

# 或者
ip addr show
```

记下你的 IP 地址，例如：`192.168.1.100` 或公网 IP。

### 第四步：启动服务器

```bash
python app_pi0_libero.py
```

你应该看到类似的输出：

```
🚀 Server starting on http://0.0.0.0:34567
📡 Accessible from external servers
 * Serving Flask app 'app_pi0_libero'
 * Running on http://0.0.0.0:34567
```

### 第五步：测试本地连接

在同一台服务器上打开新终端：

```bash
curl http://localhost:34567/health
```

应该返回：`ok`

### 第六步：从 autodl 或其他服务器测试

将 `test_client.py` 复制到 autodl 服务器，或者直接在 autodl 上运行：

```bash
# 方法 1: 使用测试脚本
python test_client.py YOUR_SERVER_IP

# 方法 2: 使用 curl
curl http://YOUR_SERVER_IP:34567/health

# 方法 3: 使用 Python
python3 << EOF
import requests
response = requests.get("http://YOUR_SERVER_IP:34567/health")
print(f"Status: {response.status_code}, Response: {response.text}")
EOF
```

### 第七步：在 autodl 中调用 API

在你的 autodl 代码中：

```python
import requests

SERVER_URL = "http://YOUR_SERVER_IP:34567"

# 发送评分请求
data = {
    "responses": [
        {"action": "your_action_data"}
    ],
    "metas": [
        {
            "original_instruction": "put the red bowl on the left shelf",
            "suite": "libero_object",
            "task_id": 3,
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

response = requests.post(
    f"{SERVER_URL}/score",
    json=data,
    headers={"Content-Type": "application/json"},
    timeout=60  # 根据你的任务调整超时时间
)

result = response.json()
print(f"Score result: {result}")
```

## ☁️ 云服务器额外步骤

如果你的服务器是云服务器（阿里云、腾讯云、AWS 等），需要在云控制台配置安全组：

### 阿里云 ECS

1. 登录 [阿里云控制台](https://ecs.console.aliyun.com/)
2. 找到你的 ECS 实例
3. 点击「安全组」→「配置规则」→「添加安全组规则」
4. 配置：
   - 规则方向：入方向
   - 协议类型：自定义 TCP
   - 端口范围：34567/34567
   - 授权对象：0.0.0.0/0
   - 描述：Pi0 Reward Server

### 腾讯云 CVM

1. 登录 [腾讯云控制台](https://console.cloud.tencent.com/cvm)
2. 找到你的 CVM 实例
3. 点击「安全组」→「修改规则」→「入站规则」→「添加规则」
4. 配置：
   - 类型：自定义
   - 协议：TCP
   - 端口：34567
   - 源：0.0.0.0/0
   - 策略：允许

### AWS EC2

1. 登录 [AWS 控制台](https://console.aws.amazon.com/ec2/)
2. 找到你的 EC2 实例
3. 点击「Security Groups」→「Edit inbound rules」→「Add rule」
4. 配置：
   - Type：Custom TCP
   - Port range：34567
   - Source：0.0.0.0/0
   - Description：Pi0 Reward Server

## ❓ 故障排查

### 问题 1: 连接被拒绝 (Connection refused)

```bash
# 检查服务是否运行
ps aux | grep app_pi0_libero

# 检查端口是否被监听
sudo netstat -tulpn | grep 34567
# 或
sudo ss -tulpn | grep 34567
```

### 问题 2: 无法从外部访问

1. **检查防火墙**：
```bash
# UFW
sudo ufw status

# Firewalld
sudo firewall-cmd --list-ports
```

2. **检查云安全组**：登录云控制台确认规则已添加

3. **检查 IP 地址**：确保使用正确的 IP（公网 IP 或内网 IP）

### 问题 3: 超时 (Timeout)

可能原因：
- 服务器在 NAT 后面，需要配置端口转发
- 两台服务器网络不通
- 云服务器安全组未配置

测试网络连通性：

```bash
# 从 autodl 服务器测试
ping YOUR_SERVER_IP
telnet YOUR_SERVER_IP 34567
```

## 📊 监控和日志

### 查看实时日志

服务器会输出请求日志到控制台。如果需要保存日志：

```bash
python app_pi0_libero.py 2>&1 | tee server.log
```

### 后台运行服务器

```bash
# 使用 nohup
nohup python app_pi0_libero.py > server.log 2>&1 &

# 查看日志
tail -f server.log

# 停止服务器
pkill -f app_pi0_libero
```

## 🔒 安全提示

⚠️ 当前配置允许所有 IP 访问（`0.0.0.0/0`）。如果你知道 autodl 的具体 IP，建议在防火墙和安全组中只允许该 IP 访问，提高安全性。

## 📚 更多信息

查看详细部署文档：[README_DEPLOYMENT.md](./README_DEPLOYMENT.md)

