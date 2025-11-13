#!/usr/bin/env bash
set -euo pipefail

# Conda-based installer for openpi
# This script mirrors the uv-based setup:
#   GIT_LFS_SKIP_SMUDGE=1 uv sync
#   GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
#
# It creates a conda environment, installs GPU-capable PyTorch and JAX,
# pins important overrides, installs workspace and git-sourced deps,
# then installs openpi in editable mode.
#
# Usage (defaults should work on most CUDA12 systems):
#   bash scripts/install_conda.sh
#
# Optional environment variables:
#   ENV_NAME   : conda environment name (default: openpi)
#   PYTHON_VER : Python version (default: 3.11)
#   GPU        : 1 to install GPU wheels (default), 0 for CPU-only
#   TORCH_CUDA : Torch CUDA tag, e.g. cu121, cu124, cpu (default: cu121)
#
# Examples:
#   ENV_NAME=openpi PYTHON_VER=3.11 bash scripts/install_conda.sh
#   GPU=0 TORCH_CUDA=cpu bash scripts/install_conda.sh   # CPU-only install
#   TORCH_CUDA=cu124 bash scripts/install_conda.sh       # CUDA 12.4 wheels

ENV_NAME=${ENV_NAME:-openpi}
PYTHON_VER=${PYTHON_VER:-3.11}
GPU=${GPU:-1}
TORCH_CUDA=${TORCH_CUDA:-cu121}

# Resolve project root to the openpi directory (script may be invoked anywhere)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$PROJECT_ROOT"

# Ensure conda is available and activatable
if ! command -v conda >/dev/null 2>&1; then
  echo "[ERROR] conda not found in PATH. Please install Miniconda/Anaconda first." >&2
  exit 1
fi
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

# Create and activate environment
if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "[INFO] Creating conda env: $ENV_NAME (python=$PYTHON_VER)"
  conda create -y -n "$ENV_NAME" python="$PYTHON_VER"
else
  echo "[INFO] Reusing existing conda env: $ENV_NAME"
fi
conda activate "$ENV_NAME"

# Base tools
conda install -y -c conda-forge git pip

# Install PyTorch first with the desired CUDA build
if [[ "$GPU" == "1" ]]; then
  echo "[INFO] Installing PyTorch GPU build (torch==2.7.1, $TORCH_CUDA)"
  # Uses PyTorch official wheel index per CUDA version
  pip install --index-url "https://download.pytorch.org/whl/${TORCH_CUDA}" "torch==2.7.1"
else
  echo "[INFO] Installing PyTorch CPU build (torch==2.7.1)"
  pip install --index-url https://download.pytorch.org/whl/cpu "torch==2.7.1"
fi

# Install JAX
# Prefer official JAX CUDA wheels index for GPU. Fall back to PyPI if needed.
if [[ "$GPU" == "1" ]]; then
  echo "[INFO] Installing JAX GPU build (jax[cuda12]==0.5.3)"
  pip install -U "jax[cuda12]==0.5.3" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html || \
  pip install "jax[cuda12]==0.5.3"
else
  echo "[INFO] Installing JAX CPU build (jax==0.5.3)"
  pip install "jax==0.5.3"
fi

# Apply uv's override-dependencies equivalents explicitly
pip install "ml-dtypes==0.4.1" "tensorstore==0.1.74"

# Respect LFS skip during git-based installs (LeRobot)
export GIT_LFS_SKIP_SMUDGE=1

# Install workspace package first so openpi can resolve local dep
pip install -e packages/openpi-client

# Install git-sourced dependency pinned by uv sources
pip install "lerobot @ git+https://github.com/huggingface/lerobot@0cf864870cf29f4738d3ade893e6fd13fbd7cdb5"

# Finally, install openpi itself in editable mode
pip install -e .

cat <<EOF
[OK] Installation complete.

To use the environment:
  conda activate $ENV_NAME

Notes:
- If you encounter JAX GPU wheel issues, ensure your host drivers match the chosen CUDA runtime.
- You may tweak TORCH_CUDA (cu121/cu124/cpu) to match your system.
- For optional RLDS extras, install later if needed:
    pip install "tensorflow-cpu==2.15.0" "tensorflow-datasets==4.9.9" "dm-tree>=0.1.8" "dlimp @ git+https://github.com/kvablack/dlimp@ad72ce3a9b414db2185bc0b38461d4101a65477a"
EOF
