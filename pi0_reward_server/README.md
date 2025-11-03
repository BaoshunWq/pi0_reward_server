# Pi0 Reward Server

一个基于 Flask 的奖励评分服务器，支持远程访问和 LIBERO 任务评估。

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置防火墙（允许远程访问）

```bash
# 运行自动配置脚本
./setup_firewall.sh

# 或手动配置（Ubuntu/Debian）
sudo ufw allow 34567/tcp
```

### 3. 启动服务器

```bash
python app_pi0_libero.py
```

服务器将在 `http://0.0.0.0:34567` 启动，可以从其他服务器访问。

### 4. 测试连接

**本地测试：**
```bash
curl http://localhost:34567/health
```

**远程测试（从 autodl 或其他服务器）：**
```bash
# 查看本机 IP
hostname -I

# 在远程服务器上测试
python test_client.py <你的服务器IP>
```

## API 接口

### GET /health
健康检查接口

**响应：**
```
ok
```

### POST /score
评分接口

**请求示例：**
```json
{
  "responses": [
    {"action": "move_forward", "value": 0.5}
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
```

**响应示例：**
```json
{
  "done_result": [0.85]
}
```

## 远程访问配置

详细的部署和远程访问配置指南，请参考 [README_DEPLOYMENT.md](README_DEPLOYMENT.md)。

包括：
- 防火墙配置
- 云服务器安全组设置
- 生产环境部署建议
- 故障排查

## 文件说明

- `app_pi0_libero.py` - Flask 服务器主文件
- `reward_core.py` - 奖励计算核心逻辑
- `test_client.py` - 远程连接测试客户端
- `setup_firewall.sh` - 防火墙自动配置脚本
- `README_DEPLOYMENT.md` - 详细部署指南

## 常见问题

**Q: 其他服务器无法访问？**

1. 确保防火墙已开放 34567 端口
2. 如果是云服务器，检查安全组配置
3. 使用 `test_client.py` 脚本测试连接

**Q: 如何查看服务器 IP？**

```bash
# 内网 IP
hostname -I

# 公网 IP
curl ifconfig.me
```

## License

MIT

