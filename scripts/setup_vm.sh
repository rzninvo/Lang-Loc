#!/bin/bash
# Setup script for Lang-Loc on a fresh VM with CUDA 12.4
# Usage: bash scripts/setup_vm.sh
set -e

REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="langloc"

echo "=== Lang-Loc VM Setup ==="
echo "Repo: $REPO_DIR"

# 0. Check CUDA
echo -e "\n=== Checking CUDA ==="
nvcc --version 2>/dev/null || echo "WARNING: nvcc not found — make sure CUDA toolkit is in PATH"
nvidia-smi | head -4 || { echo "ERROR: nvidia-smi failed — check GPU drivers"; exit 1; }

# 1. Check / install conda
echo -e "\n=== Checking conda ==="
if command -v conda &>/dev/null; then
    echo "conda found: $(conda --version)"
else
    echo "conda not found — installing Miniconda..."
    curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p "$HOME/miniconda3"
    rm /tmp/miniconda.sh
    eval "$("$HOME/miniconda3/bin/conda" shell.bash hook)"
    conda init bash
    echo "Miniconda installed. You may need to restart your shell, then re-run this script."
    exit 0
fi

# 2. Create conda environment
echo -e "\n=== Creating conda environment: $ENV_NAME ==="
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "Environment '$ENV_NAME' already exists — activating it"
else
    conda create -n "$ENV_NAME" python=3.10 -y
fi
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"
pip install --upgrade pip setuptools wheel

# 3. Install PyTorch with CUDA 12.6 (supports Blackwell / sm_120 GPUs like 5090)
echo -e "\n=== Installing PyTorch (CUDA 12.6 — Blackwell compatible) ==="
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# 4. Verify CUDA is visible
echo -e "\n=== Verifying PyTorch CUDA ==="
python3 -c "
import torch
print(f'PyTorch {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'Device: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"

# 5. Install VLM baseline dependencies
echo -e "\n=== Installing VLM baseline deps ==="
pip install transformers accelerate qwen-vl-utils
pip install numpy Pillow open3d tqdm scipy

# 6. Install the langloc package in editable mode
echo -e "\n=== Installing langloc package ==="
cd "$REPO_DIR"
pip install -e . --no-deps  # core deps already installed above

# 7. Create eval output directory
mkdir -p "$REPO_DIR/eval"

echo ""
echo "========================================="
echo "  Setup complete!"
echo "========================================="
echo ""
echo "Activate the environment:"
echo "  conda activate $ENV_NAME"
echo ""
echo "Run the VLM baseline (8B model):"
echo "  cd $REPO_DIR"
echo "  bash scripts/localization/baseline_eval_qwen.sh --model_id Qwen/Qwen3-VL-8B-Instruct"
echo ""
echo "For ScanNet:"
echo "  DATASET=scannet SCENE_ROOT=./data/scans bash scripts/localization/baseline_eval_qwen.sh --model_id Qwen/Qwen3-VL-8B-Instruct"
echo ""
echo "Install Claude CLI (optional):"
echo "  npm install -g @anthropic-ai/claude-code"
