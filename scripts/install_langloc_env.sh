#!/bin/bash
# One-shot installer for the langloc env.
#
# Picks up after setup_vm.sh (which created the env but failed at pip due to
# non-interactive conda activate). Sources conda profile.d directly and uses
# `python -m pip` everywhere so we always hit the env's pip instead of the
# system /usr/bin/pip that sits ahead of the conda env bin in PATH on this
# machine.
#
# Run from repo root.
set -euo pipefail

REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="langloc"
cd "$REPO_DIR"

echo "[INFO] Repo: $REPO_DIR"
echo "[INFO] Env:  $ENV_NAME"

CONDA_SH="$HOME/miniconda3/etc/profile.d/conda.sh"
[ -f "$CONDA_SH" ] || { echo "[ERROR] conda profile not found: $CONDA_SH"; exit 1; }
# shellcheck disable=SC1090
source "$CONDA_SH"

if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "[INFO] Creating env $ENV_NAME (Python 3.10)"
    conda create -n "$ENV_NAME" python=3.10 -y
else
    echo "[INFO] Env $ENV_NAME already exists"
fi

conda activate "$ENV_NAME"

# Hard-bind PY to the env's python so we never accidentally hit system pip
PY="$CONDA_PREFIX/bin/python"
[ -x "$PY" ] || { echo "[ERROR] env python missing: $PY"; exit 1; }

echo "[INFO] Active env: $CONDA_DEFAULT_ENV"
echo "[INFO] env python:   $PY ($($PY --version 2>&1))"
echo "[INFO] env pip:      $($PY -m pip --version)"

# 1. Upgrade core packaging tools (env-local pip)
$PY -m pip install --upgrade pip setuptools wheel

# 2. Install PyTorch 2.9.0 with CUDA 12.8 wheels — required for RTX 5090
# (Blackwell, sm_120). torch 2.6 cu126 wheels were compiled for sm_50…sm_90
# only and emit "no kernel image is available for execution on the device"
# on every CUDA op. 2.9.0 is the highest version PyG ships matching wheels
# for (`pt29cu128`, see step 3a).
echo "[INFO] Installing PyTorch 2.9.0+cu128 (sm_120 / Blackwell)"
$PY -m pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 \
    --index-url https://download.pytorch.org/whl/cu128

# Verify CUDA dispatch actually works on this GPU before proceeding.
$PY - <<'PY_EOF'
import torch
print(f"PyTorch {torch.__version__}, arch list: {torch.cuda.get_arch_list()}")
assert torch.cuda.is_available(), "CUDA not available"
props = torch.cuda.get_device_properties(0)
print(f"GPU: {props.name}, sm_{props.major}{props.minor}")
x = torch.randn(8, device="cuda")
(x * 2).sum().item()
torch.cuda.synchronize()
print("tiny CUDA op: OK")
PY_EOF

$PY - <<'PY_EOF'
import torch
print(f"PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}")
assert torch.cuda.is_available(), "CUDA not available"
print(f"Device: {torch.cuda.get_device_name(0)}")
print(f"Compiled CUDA: {torch.version.cuda}")
PY_EOF

# 3a. Install torch_scatter from PyG's prebuilt wheel index (the source sdist
# imports torch in setup.py and breaks under pip's default build isolation).
# PyG ships pt29cu128 wheels for cp310-linux_x86_64; that's our lane.
echo "[INFO] Installing torch_scatter from PyG wheel index (torch-2.9.0+cu128)"
$PY -m pip install torch_scatter \
    -f https://data.pyg.org/whl/torch-2.9.0+cu128.html

# 3b. Install repo requirements. requirements.txt does not pin torch, so the
# already-installed cu126 wheels are kept; torch_scatter is now satisfied.
echo "[INFO] Installing requirements.txt"
$PY -m pip install -r requirements.txt

# 4. Install langloc package in editable mode (no-deps so it doesn't drag in
# transitive constraints that conflict with what we just installed).
echo "[INFO] Installing langloc (editable, --no-deps)"
$PY -m pip install -e . --no-deps

# 4b. Download the spaCy model used for Word2Vec label embeddings in
# scene_graph_builder.add_embeddings_to_scene_graph(). Without this the
# keyframe-selection pipeline crashes after DPP with `OSError: [E050] Can't
# find model 'en_core_web_lg'`. ~400 MB download.
echo "[INFO] Installing spaCy en_core_web_lg"
$PY -m spacy download en_core_web_lg

# 5. Build PyTorch3D from source. Slow (15–30 min on a 24-core box).
PT3D_DIR="${PT3D_DIR:-$HOME/src/pytorch3d}"
if [ ! -d "$PT3D_DIR" ]; then
    echo "[INFO] Cloning pytorch3d to $PT3D_DIR"
    mkdir -p "$(dirname "$PT3D_DIR")"
    git clone https://github.com/facebookresearch/pytorch3d.git "$PT3D_DIR"
fi

echo "[INFO] Building pytorch3d"
pushd "$PT3D_DIR" > /dev/null
git fetch --all --tags >/dev/null 2>&1 || true
if git rev-parse v0.7.8 >/dev/null 2>&1; then
    git checkout v0.7.8
    echo "[INFO] Using pytorch3d v0.7.8"
else
    echo "[WARN] v0.7.8 tag not found, building from current branch"
fi
$PY -m pip install ninja fvcore iopath
FORCE_CUDA=1 $PY -m pip install -e . --no-build-isolation
popd > /dev/null

# 6. Final smoke import
echo "[INFO] Verifying full import stack"
$PY - <<'PY_EOF'
import importlib, sys
fail = []
for mod in [
    'torch', 'torchvision', 'pytorch3d', 'pyiqa', 'open3d', 'openai',
    'omegaconf', 'hydra', 'clip', 'transformers', 'torch_geometric',
    'torch_scatter', 'cv2', 'plyfile', 'fcl', 'spacy',
    'langloc.utils.config_loader',
]:
    try:
        m = importlib.import_module(mod)
        v = getattr(m, '__version__', '?')
        print(f"  OK   {mod:35s} {v}")
    except Exception as e:
        print(f"  FAIL {mod:35s} {type(e).__name__}: {str(e)[:80]}")
        fail.append(mod)
if fail:
    print(f"\nFAILED: {fail}")
    sys.exit(1)
print("\nALL IMPORTS OK")
PY_EOF

echo ""
echo "==================================================="
echo "  Install complete. Env: $ENV_NAME"
echo "==================================================="
