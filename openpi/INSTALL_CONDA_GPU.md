# 使用 Conda 按 GPU 安装 openpi（逐步指令）

本文档提供等价替代以下 uv 安装流程的逐步命令：
- `GIT_LFS_SKIP_SMUDGE=1 uv sync`
- `GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .`

并保持关键行为：
- Python 3.11
- GPU 版本 `torch==2.7.1`（CUDA 12.1，对应 cu121 索引）
- `jax[cuda12]==0.5.3`
- 覆盖依赖：`ml-dtypes==0.4.1`、`tensorstore==0.1.74`
- 本地 `packages/openpi-client` 可编辑安装
- 固定提交的 `lerobot`
- 安装 git 依赖时保留 `GIT_LFS_SKIP_SMUDGE=1`

---

## 逐步安装命令

- 进入 openpi 目录
```bash
cd /root/autodl-tmp/code/attackVLA/pi0_reward_server/openpi
```

- 创建并激活 Conda 环境（Python 3.11）
```bash
conda create -y -n openpi python=3.11
conda activate openpi
```

- 安装基础工具
```bash
conda install -y -c conda-forge git pip
```

- 安装 PyTorch（GPU，CUDA 12.1）
```bash
pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.7.1
```

- 安装 JAX（GPU，CUDA 12）
```bash
pip install -U "jax[cuda12]==0.5.3" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

- 安装 uv 的覆盖依赖等价项
```bash
pip install "ml-dtypes==0.4.1" "tensorstore==0.1.74"
```

- 可编辑安装本地 openpi-client（保留 LFS 跳过）
```bash
GIT_LFS_SKIP_SMUDGE=1 pip install -e packages/openpi-client
```

- 安装指定提交的 LeRobot（保留 LFS 跳过）
```bash
GIT_LFS_SKIP_SMUDGE=1 pip install "lerobot @ git+https://github.com/huggingface/lerobot@0cf864870cf29f4738d3ade893e6fd13fbd7cdb5"
```

- 可编辑安装 openpi 本体
```bash
pip install -e .
```
pip install pytest
---

## 可选：安装 RLDS 相关依赖
若需要 `dependency-groups.rlds` 功能，可执行：
```bash
pip install "tensorflow-cpu==2.15.0" "tensorflow-datasets==4.9.9" "dm-tree>=0.1.8" "dlimp @ git+https://github.com/kvablack/dlimp@ad72ce3a9b414db2185bc0b38461d4101a65477a"
```

## 切换到 CUDA 12.4 的 PyTorch（可选）
若你的系统更适配 CUDA 12.4，将上面“安装 PyTorch”那一步替换为：
```bash
pip install --index-url https://download.pytorch.org/whl/cu124 torch==2.7.1
```

---

## 备注
- 若 JAX GPU 轮子安装失败，多半是驱动/运行时版本不匹配；可尝试 CPU 先行验证或调整 CUDA 版本。
- 如遇到 LFS 相关问题，确保 `git-lfs` 已安装并在 PATH 中；此处通过 `GIT_LFS_SKIP_SMUDGE=1` 避免 clone 时自动拉取大文件。
