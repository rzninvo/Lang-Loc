#!/bin/bash
# Validate the canonical V2 retraining: rebuild caches from a fresh
# `data/model_checkpoints/graph2graph/canonical_v2/<ckpt>` checkpoint and
# run Tables 1+2 evaluation.
#
# Cache dir is `outputs/canonical_v2_eval_caches/` (symlinks to the input
# graph files in VLSG_TEXT_v2/VLSG_Files/). Override with CACHE_DIR=...
# Override checkpoint via CHECKPOINT=... (default: last.pth).

set -euo pipefail

REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

CACHE_DIR="${CACHE_DIR:-outputs/canonical_v2_eval_caches}"
CHECKPOINT="${CHECKPOINT:-data/model_checkpoints/graph2graph/canonical_v2/last.pth}"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate langloc
PY="$CONDA_PREFIX/bin/python"

if [ ! -f "$CHECKPOINT" ]; then
    echo "[ERROR] checkpoint not found: $CHECKPOINT" >&2
    echo "        run langloc/retrieval/train.py first" >&2
    exit 1
fi
for f in 3dssg_graphs_518D.pt scanscribe_graphs_test_518D.pt scanscribe_cleaned_original_518D.pt; do
    if [ ! -e "$CACHE_DIR/$f" ]; then
        echo "[ERROR] missing input graph file: $CACHE_DIR/$f" >&2
        echo "        symlink it from VLSG_TEXT_v2/VLSG_Files/$f" >&2
        exit 1
    fi
done

echo "[VALIDATE] checkpoint=$CHECKPOINT"
echo "[VALIDATE] cache_dir=$CACHE_DIR"
echo

echo "[VALIDATE] step 1/2: rebuild caches"
"$PY" -m scripts.retrieval.precompute_eval_embeddings \
    --checkpoint "$CHECKPOINT" \
    --cache_dir  "$CACHE_DIR" \
    --device cuda

echo
echo "[VALIDATE] step 2/2: Tables 1 + 2 (seed=42, weights 0.33/0.33/0.34)"
"$PY" -m langloc.retrieval.eval --cache_dir "$CACHE_DIR" --mode both
