#!/bin/bash
# Reproduce paper Tables 1+2 (scene retrieval Recall@k on ScanScribe).
#
# Two modes:
#   --use_cache (default): use the pre-computed db_emb_cache.pt + query_emb_cache.pt
#                          shipped in the Colab artifact. Fast (~2 sec).
#   --rebuild_cache:       re-run the full DualSceneAlignerV2 + SimpleGraphMatcher
#                          forward pass on every DB and query graph to produce
#                          fresh caches (~3-5 min on a CUDA box).
#
# Both modes use seed=42 and `langloc/retrieval/eval.py` which mirrors
# Shirley's `eval_518_multitask_original_table1_v2.py` (Eq. 8 weights
# 0.33/0.33/0.34, 218-scene Table 1 distractor pool, 10 outer × 100 inner
# rounds).
#
# Default cache_dir points at `VLSG_TEXT_v2/VLSG_Files`. Override via
# CACHE_DIR=... if your data lives elsewhere.

set -euo pipefail

REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

CACHE_DIR="${CACHE_DIR:-VLSG_TEXT_v2/VLSG_Files}"
CHECKPOINT="${CHECKPOINT:-$CACHE_DIR/checkpoints/epoch_70_163_cliprel.pth}"
MODE="use_cache"
for arg in "$@"; do
    case "$arg" in
        --rebuild_cache) MODE="rebuild_cache" ;;
        --use_cache)     MODE="use_cache" ;;
        *) echo "Unknown arg: $arg"; exit 1 ;;
    esac
done

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate langloc
PY="$CONDA_PREFIX/bin/python"

echo "[REPRODUCE] cache_dir=$CACHE_DIR"
echo "[REPRODUCE] mode=$MODE"

if [ "$MODE" = "rebuild_cache" ]; then
    if [ ! -f "$CHECKPOINT" ]; then
        echo "[ERROR] checkpoint not found: $CHECKPOINT" >&2
        echo "        download epoch_70_163_cliprel.pth from the released artifact" >&2
        exit 1
    fi
    echo "[REPRODUCE] rebuilding caches from $CHECKPOINT"
    "$PY" -m scripts.retrieval.precompute_eval_embeddings \
        --checkpoint "$CHECKPOINT" \
        --cache_dir  "$CACHE_DIR" \
        --device cuda
fi

if [ ! -f "$CACHE_DIR/db_emb_cache.pt" ] || [ ! -f "$CACHE_DIR/query_emb_cache.pt" ]; then
    echo "[ERROR] missing caches in $CACHE_DIR; rerun with --rebuild_cache" >&2
    exit 1
fi

echo
echo "[REPRODUCE] running Tables 1 + 2 eval (seed=42, weights 0.33/0.33/0.34)"
"$PY" -m langloc.retrieval.eval --cache_dir "$CACHE_DIR" --mode both
