# 使用 Conda 为 LIBERO 示例创建独立环境（逐步指令）

本文档等价替代以下 uv 流程：
- `uv venv --python 3.8 examples/libero/.venv`
- `source examples/libero/.venv/bin/activate`
- `uv pip sync examples/libero/requirements.txt third_party/libero/requirements.txt --extra-index-url https://download.pytorch.org/whl/cu113 --index-strategy=unsafe-best-match`
- `uv pip install -e packages/openpi-client`
- `uv pip install -e third_party/libero`

改为使用 Conda + pip，一步步执行如下命令（GPU，CUDA 11.3 轮子来源）：

---

## 逐步安装命令

- 进入 openpi 目录
```bash
cd /root/autodl-tmp/code/attackVLA/pi0_reward_server/openpi
```

- 创建并激活 Conda 环境（Python 3.8）
```bash
conda create -y -n openpi-libero python=3.9
conda activate openpi-libero
```

- 安装基础工具
```bash
conda install -y -c conda-forge git pip
```

- 按两个 requirements 文件一次性安装（使用 PyTorch CUDA 11.3 轮子源）
```bash
pip install --extra-index-url https://download.pytorch.org/whl/cu124 \
  -r examples/libero/requirements.txt \
  -r third_party/libero/requirements.txt
```

- 可编辑安装本地 openpi-client
```bash
pip install -e packages/openpi-client
```
pip install flask
- 可编辑安装第三方 libero（子模块）
```bash
pip install -e third_party/libero
```

pip install nltk
- 运行示例前（推荐）加入 PYTHONPATH，便于直接 import 自由度更高
```bash
export PYTHONPATH="$PYTHONPATH:$PWD/third_party/libero"
```

---

## 运行示例（可选）
- 运行模拟：
```bash
python examples/libero/main.py
```
- 若遇到 EGL 问题，尝试：
```bash
MUJOCO_GL=glx python examples/libero/main.py
```

---

## 说明
- 上述 `--extra-index-url https://download.pytorch.org/whl/cu113` 使 pip 能从官方 CUDA 11.3 轮子源获取 PyTorch 及其相关依赖。
- `uv pip sync` 的 `--index-strategy=unsafe-best-match` 为 uv 特性，pip 无等价参数；若出现版本冲突，请根据提示调整版本或先安装特定的 `torch` 版本后再安装其余依赖。
- 若希望 CPU-only，可将 `--extra-index-url` 改为 `https://download.pytorch.org/whl/cpu`，或直接移除该参数使用 PyPI 默认源（注意 requirements 中如固定了 GPU 依赖可能导致冲突）。
