#!/bin/bash
# Upgrade langloc env to torch 2.9.0 + cu128 wheels (sm_120 / Blackwell-compatible).
#
# Why: torch 2.6 cu126 wheels were compiled for sm_50…sm_90 only. RTX 5090 is
# sm_120 (Blackwell) → "no kernel image is available for execution on the
# device" on every CUDA op. PyG wheels exist for pt29cu128 (cp310-linux) but
# not yet for pt210cu128, so 2.9.0 is the highest compatible torch.
set -euo pipefail

REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="langloc"
cd "$REPO_DIR"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
PY="$CONDA_PREFIX/bin/python"

echo "[INFO] Active env: $CONDA_DEFAULT_ENV"
echo "[INFO] env python: $PY ($($PY --version 2>&1))"

# 1. Uninstall torch + extensions tied to old ABI
echo "[INFO] Uninstalling old torch / torch_scatter / pytorch3d"
$PY -m pip uninstall -y torch torchvision torchaudio torch_scatter pytorch3d || true

# 2. Install torch 2.9.0 + cu128 stack
echo "[INFO] Installing torch 2.9.0+cu128"
$PY -m pip install \
    torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 \
    --index-url https://download.pytorch.org/whl/cu128

# 3. Sanity: arch list now includes sm_120 + tiny op works on the 5090
$PY - <<'PY_EOF'
import torch
print(f"torch {torch.__version__}")
print(f"arch list: {torch.cuda.get_arch_list()}")
assert torch.cuda.is_available(), "CUDA not available"
props = torch.cuda.get_device_properties(0)
print(f"GPU: {props.name}, sm_{props.major}{props.minor}")
x = torch.randn(8, device="cuda")
y = (x * 2).sum().item()
torch.cuda.synchronize()
print(f"tiny op result: {y:.4f} (CUDA dispatch OK)")
PY_EOF

# 4. Re-install torch_scatter for pt29cu128 (PyG prebuilt wheel)
echo "[INFO] Installing torch_scatter from PyG wheel index (pt29cu128)"
$PY -m pip install torch_scatter \
    -f https://data.pyg.org/whl/torch-2.9.0+cu128.html

# 5. Rebuild pytorch3d from source against the new torch ABI.
# Wipe the previous build artifacts in the editable clone so .o files
# don't poison the new ABI.
PT3D_DIR="${PT3D_DIR:-$HOME/src/pytorch3d}"
[ -d "$PT3D_DIR" ] || { echo "[ERROR] pytorch3d clone not found at $PT3D_DIR"; exit 1; }
pushd "$PT3D_DIR" > /dev/null
echo "[INFO] Cleaning previous pytorch3d build artifacts"
rm -rf build/ dist/ pytorch3d.egg-info/
find . -name '*.so' -delete
find . -name '*.o' -delete
$PY -m pip install ninja fvcore iopath
echo "[INFO] Rebuilding pytorch3d (FORCE_CUDA=1)"
FORCE_CUDA=1 $PY -m pip install -e . --no-build-isolation
popd > /dev/null

# 6. Final import + CUDA dispatch check
echo "[INFO] Final import + CUDA-dispatch verification"
$PY - <<'PY_EOF'
import importlib, sys
mods = [
    "torch", "torchvision", "pytorch3d", "pyiqa", "open3d", "openai",
    "omegaconf", "hydra", "clip", "transformers", "torch_geometric",
    "torch_scatter", "cv2", "plyfile", "fcl", "spacy",
    "langloc.utils.config_loader",
]
fail = []
for m in mods:
    try:
        mod = importlib.import_module(m)
        v = getattr(mod, "__version__", "?")
        print(f"  OK   {m:35s} {v}")
    except Exception as e:
        print(f"  FAIL {m:35s} {type(e).__name__}: {str(e)[:80]}")
        fail.append(m)
assert not fail, f"failed imports: {fail}"

# Check pytorch3d CUDA op specifically
import torch
from pytorch3d.renderer import PerspectiveCameras
from pytorch3d.structures import Meshes
verts = torch.randn(1, 4, 3, device="cuda")
faces = torch.tensor([[[0,1,2],[1,2,3]]], device="cuda")
m = Meshes(verts=verts, faces=faces)
print(f"  OK   pytorch3d Meshes on CUDA: {m.verts_packed().device}")
print("\nALL IMPORTS OK — sm_120 dispatch confirmed.")
PY_EOF

echo ""
echo "==================================================="
echo "  Upgrade complete. Env: $ENV_NAME (torch 2.9.0+cu128)"
echo "==================================================="
