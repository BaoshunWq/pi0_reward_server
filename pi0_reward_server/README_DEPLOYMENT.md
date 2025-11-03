# Pi0 Reward Server 部署指南

## 1. 安装依赖

确保安装了 flask-cors：

```bash
pip install flask-cors
```

## 2. 防火墙配置

### Ubuntu/Debian 系统 (使用 ufw)

```bash
# 开放 34567 端口
sudo ufw allow 34567/tcp

# 查看防火墙状态
sudo ufw status
```

### CentOS/RHEL 系统 (使用 firewalld)

```bash
# 开放 34567 端口
sudo firewall-cmd --permanent --add-port=34567/tcp
sudo firewall-cmd --reload

# 查看已开放端口
sudo firewall-cmd --list-ports
```

### 直接使用 iptables

```bash
# 开放 34567 端口
sudo iptables -A INPUT -p tcp --dport 34567 -j ACCEPT

# 保存规则
sudo iptables-save > /etc/iptables/rules.v4
```

## 3. 启动服务器

```bash
cd /home/baoshuntong/code/saftyEmbodyAI/redTeam_pi0/pi0_reward_server
python app_pi0_libero.py
```

## 4. 从其他服务器访问

### 获取服务器 IP

```bash
# 查看内网 IP
hostname -I

# 查看公网 IP（如果有）
curl ifconfig.me
```

### 从 autodl 或其他服务器访问

使用以下 Python 代码测试连接：

```python
import requests

# 替换为你的服务器 IP 地址
SERVER_IP = "your.server.ip.address"
SERVER_PORT = 34567

# 健康检查
response = requests.get(f"http://{SERVER_IP}:{SERVER_PORT}/health")
print(f"Health check: {response.text}")

# 发送评分请求
data = {
    "responses": [{"action": "test"}],
    "metas": [{
        "original_instruction": "test task",
        "suite": "libero_object",
        "task_id": 0,
        "seed": 0,
        "init_state_id": 0
    }],
    "reward_function_kwargs": {
        "alpha": 1.0,
        "beta": 0.1,
        "gamma": 0.5
    }
}

response = requests.post(
    f"http://{SERVER_IP}:{SERVER_PORT}/score",
    json=data,
    headers={"Content-Type": "application/json"}
)

print(f"Score response: {response.json()}")
```

## 5. 云服务器额外配置

如果你使用的是阿里云、腾讯云、AWS 等云服务器，还需要在控制台配置安全组：

### 阿里云 ECS
1. 登录阿里云控制台
2. 进入 ECS 实例详情
3. 点击「安全组」→「配置规则」
4. 添加入方向规则：端口 34567，授权对象 0.0.0.0/0

### 腾讯云 CVM
1. 登录腾讯云控制台
2. 进入 CVM 实例详情
3. 点击「安全组」→「修改规则」
4. 添加入站规则：TCP:34567，来源 0.0.0.0/0

### AWS EC2
1. 登录 AWS 控制台
2. 进入 EC2 实例详情
3. 点击「Security Groups」
4. 添加 Inbound Rule：Custom TCP, Port 34567, Source 0.0.0.0/0

## 6. 生产环境建议

### 使用 Gunicorn 运行（推荐）

```bash
# 安装 gunicorn
pip install gunicorn

# 运行服务器（4个工作进程）
gunicorn -w 4 -b 0.0.0.0:34567 app_pi0_libero:create_app()
```

### 使用 systemd 设置自动启动

创建服务文件 `/etc/systemd/system/pi0-reward.service`：

```ini
[Unit]
Description=Pi0 Reward Server
After=network.target

[Service]
Type=simple
User=baoshuntong
WorkingDirectory=/home/baoshuntong/code/saftyEmbodyAI/redTeam_pi0/pi0_reward_server
ExecStart=/usr/bin/python3 app_pi0_libero.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

启动服务：

```bash
sudo systemctl daemon-reload
sudo systemctl enable pi0-reward
sudo systemctl start pi0-reward
sudo systemctl status pi0-reward
```

## 7. 安全建议

### 添加 API Token 认证

如果需要更高的安全性，可以在请求中添加 token 验证：

```python
# 在 app_pi0_libero.py 中添加
API_TOKEN = "your-secret-token-here"

@app.before_request
def check_token():
    if request.endpoint != 'health':
        token = request.headers.get('Authorization')
        if token != f"Bearer {API_TOKEN}":
            return jsonify({"error": "Unauthorized"}), 401
```

### 限制访问 IP

如果只想允许特定 IP 访问，修改 CORS 配置：

```python
CORS(app, resources={
    r"/*": {
        "origins": ["http://specific-ip:port"],  # 指定允许的 IP
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})
```

## 8. 故障排查

### 检查服务器是否监听端口

```bash
# Linux
sudo netstat -tulpn | grep 34567
# 或
sudo ss -tulpn | grep 34567
```

### 测试本地连接

```bash
curl http://localhost:34567/health
```

### 测试远程连接

```bash
# 从另一台服务器测试
curl http://your-server-ip:34567/health
```

### 查看日志

```bash
# 如果使用 systemd
sudo journalctl -u pi0-reward -f

# 如果直接运行
# 查看控制台输出
```

## 9. 常见问题

**Q: 连接被拒绝 (Connection refused)**
- 检查服务器是否正在运行
- 检查防火墙规则
- 检查云服务器安全组配置

**Q: 超时 (Connection timeout)**
- 检查网络连接
- 检查服务器公网 IP 是否正确
- 检查是否在 NAT 后面（需要端口转发）

**Q: CORS 错误**
- 确保已安装 flask-cors
- 检查 CORS 配置是否正确

**Q: 502 Bad Gateway**
- 检查应用是否正常运行
- 查看应用日志排查错误

