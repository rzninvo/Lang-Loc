#!/bin/bash
# Validate the canonical V2 retraining: rebuild caches from a fresh
# `data/model_checkpoints/graph2graph/canonical_v2/<ckpt>` checkpoint and
# run Tables 1+2 evaluation.
#
# Uses a separate cache dir (`outputs/canonical_v2_eval_caches/`) that
# symlinks the input graph files from `data/processed_data/eval_pool/` —
# this keeps the paper caches in `eval_pool/` (db_emb_cache.pt,
# query_emb_cache.pt) intact while the canonical_v2 retrain produces its
# own caches alongside.
#
# Override CACHE_DIR=... or CHECKPOINT=... if your data lives elsewhere.

set -euo pipefail

REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

CACHE_DIR="${CACHE_DIR:-outputs/canonical_v2_eval_caches}"
CHECKPOINT="${CHECKPOINT:-data/model_checkpoints/graph2graph/canonical_v2/last.pth}"
EVAL_POOL="data/processed_data/eval_pool"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate langloc
PY="$CONDA_PREFIX/bin/python"

if [ ! -f "$CHECKPOINT" ]; then
    echo "[ERROR] checkpoint not found: $CHECKPOINT" >&2
    echo "        run langloc.retrieval.train first" >&2
    exit 1
fi

# Set up cache dir with symlinks to eval pool inputs (idempotent)
mkdir -p "$CACHE_DIR"
for f in 3dssg_graphs_518D.pt scanscribe_graphs_test_518D.pt scanscribe_cleaned_original_518D.pt; do
    src="$EVAL_POOL/$f"
    dst="$CACHE_DIR/$f"
    if [ ! -e "$src" ]; then
        echo "[ERROR] missing input graph file: $src" >&2
        exit 1
    fi
    if [ ! -L "$dst" ] && [ ! -f "$dst" ]; then
        ln -s "$(realpath "$src")" "$dst"
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
