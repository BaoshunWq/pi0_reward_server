#!/usr/bin/env bash
# 快速测试 Qwen2-VL-2B 模型配置（不运行完整评估）

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

echo "=========================================="
echo "Qwen2-VL-2B 快速配置测试"
echo "=========================================="

# 1. 检查 Python 环境
echo ""
echo "[1/5] 检查 Python 环境..."
python --version || { echo "错误: Python 未安装"; exit 1; }

# 2. 检查必要的包
echo ""
echo "[2/5] 检查依赖包..."
MISSING_PACKAGES=()

python -c "import transformers" 2>/dev/null || MISSING_PACKAGES+=("transformers")
python -c "import torch" 2>/dev/null || MISSING_PACKAGES+=("torch")
python -c "import qwen_vl_utils" 2>/dev/null || MISSING_PACKAGES+=("qwen-vl-utils")
python -c "import tyro" 2>/dev/null || MISSING_PACKAGES+=("tyro")
python -c "import libero" 2>/dev/null || MISSING_PACKAGES+=("libero")

if [ ${#MISSING_PACKAGES[@]} -gt 0 ]; then
    echo "警告: 缺少以下包: ${MISSING_PACKAGES[*]}"
    echo "请运行: pip install ${MISSING_PACKAGES[*]}"
else
    echo "✓ 所有依赖包已安装"
fi

# 3. 检查服务器状态
echo ""
echo "[3/5] 检查服务器状态..."
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-6000}"

if curl -s "http://${HOST}:${PORT}/health" > /dev/null 2>&1; then
    echo "✓ 服务器运行正常 (${HOST}:${PORT})"
else
    echo "✗ 服务器未响应 (${HOST}:${PORT})"
    echo "  请先启动服务器: bash scripts/start_all_parallel.sh"
fi

# 4. 检查模型文件
echo ""
echo "[4/5] 检查模型配置..."
QWEN_MODEL_ID="${QWEN_MODEL_ID:-Qwen/Qwen2-VL-2B-Instruct}"
QWEN_MODE="${QWEN_MODE:-local}"

echo "  模式: ${QWEN_MODE}"
echo "  模型: ${QWEN_MODEL_ID}"

if [ "${QWEN_MODE}" = "local" ]; then
    echo "  提示: 首次运行会自动下载模型（约 4-5GB）"
    echo "  下载位置: ~/.cache/huggingface/hub/"
elif [ "${QWEN_MODE}" = "api" ]; then
    if [ -z "${DASHSCOPE_API_KEY:-}" ]; then
        echo "  警告: DASHSCOPE_API_KEY 未设置"
        echo "  请设置: export DASHSCOPE_API_KEY=your-api-key"
    else
        echo "  ✓ DASHSCOPE_API_KEY 已设置"
    fi
fi

# 5. 检查必要文件
echo ""
echo "[5/5] 检查必要文件..."
FILES_TO_CHECK=(
    "unified_eval.py"
    "utils.py"
    "generate_intruction.py"
    "test_qwen2vl_2b.sh"
    "config_qwen2vl_2b.env"
)

ALL_FILES_EXIST=true
for file in "${FILES_TO_CHECK[@]}"; do
    if [ -f "$file" ]; then
        echo "  ✓ $file"
    else
        echo "  ✗ $file (缺失)"
        ALL_FILES_EXIST=false
    fi
done

# 汇总
echo ""
echo "=========================================="
echo "配置检查完成"
echo "=========================================="

if [ ${#MISSING_PACKAGES[@]} -eq 0 ] && [ "$ALL_FILES_EXIST" = true ]; then
    echo "✓ 所有检查通过"
    echo ""
    echo "下一步:"
    echo "  1. 确保服务器运行: bash scripts/start_all_parallel.sh"
    echo "  2. 运行测试: bash test_qwen2vl_2b.sh"
    echo "  3. 或使用配置文件: source config_qwen2vl_2b.env && bash test_qwen2vl_2b.sh"
else
    echo "✗ 部分检查失败，请先解决上述问题"
    exit 1
fi

echo ""
echo "=========================================="
