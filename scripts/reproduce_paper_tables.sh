#!/bin/bash
# Reproduce paper Tables 1+2+3 (scene retrieval Recall@k on ScanScribe).
#
#   Table 1: 10-candidate pool, ScanScribe-text queries (Top-1/2/3/5)
#   Table 2: full 55-test-scene pool, ScanScribe-text queries (Top-5/10/20/30)
#   Table 3: 10-candidate pool, LLM-image-derived queries (Top-1/2/3/5)
#
# Modes:
#   --use_cache (default): use the paper's pre-computed db_emb_cache.pt +
#                          query_emb_cache.pt, plus our regenerated
#                          query_emb_cache_img.pt. Fast (~3 sec).
#   --rebuild_cache:       re-run the full DualSceneAlignerV2 +
#                          SimpleGraphMatcher forward pass to produce fresh
#                          caches (~3-5 min on a CUDA box).
#
# All modes use seed=42 and `langloc/retrieval/eval.py` (mirrors Shirley's
# `eval_518_multitask_original_table1_v2.py`/`eval_518_multitask.py`):
# Eq. 8 weights 0.33/0.33/0.34, 218-scene Tables 1+3 distractor pool,
# 10 outer × 100 inner rounds.
#
# Default cache_dir is `data/processed_data/eval_pool/`. Override via
# CACHE_DIR=... if your data lives elsewhere.

set -euo pipefail

REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

CACHE_DIR="${CACHE_DIR:-data/processed_data/eval_pool}"
CHECKPOINT="${CHECKPOINT:-data/model_checkpoints/graph2graph/paper/epoch_70_163_cliprel.pth}"
IMG_QUERY_PATH="${IMG_QUERY_PATH:-data/processed_data/scanscribe/scanscribe_text_graphs_from_image_desc_node_edge_features.pt}"
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
echo "[REPRODUCE] checkpoint=$CHECKPOINT"
echo "[REPRODUCE] mode=$MODE"

if [ "$MODE" = "rebuild_cache" ]; then
    if [ ! -f "$CHECKPOINT" ]; then
        echo "[ERROR] checkpoint not found: $CHECKPOINT" >&2
        exit 1
    fi
    echo "[REPRODUCE] step 1/2: rebuilding Tables 1+2 caches"
    "$PY" -m scripts.retrieval.precompute_eval_embeddings \
        --checkpoint "$CHECKPOINT" \
        --cache_dir  "$CACHE_DIR" \
        --device cuda

    echo "[REPRODUCE] step 2/2: rebuilding Table 3 query cache (image-derived)"
    "$PY" -m scripts.retrieval.precompute_eval_embeddings \
        --checkpoint "$CHECKPOINT" \
        --cache_dir  "$CACHE_DIR" \
        --query_path "$IMG_QUERY_PATH" \
        --cache_suffix _img --skip_db \
        --device cuda
fi

if [ ! -f "$CACHE_DIR/db_emb_cache.pt" ] || [ ! -f "$CACHE_DIR/query_emb_cache.pt" ]; then
    echo "[ERROR] missing Tables 1+2 caches in $CACHE_DIR; rerun with --rebuild_cache" >&2
    exit 1
fi
if [ ! -f "$CACHE_DIR/query_emb_cache_img.pt" ]; then
    echo "[ERROR] missing Table 3 cache ($CACHE_DIR/query_emb_cache_img.pt); rerun with --rebuild_cache" >&2
    exit 1
fi

echo
echo "[REPRODUCE] running Tables 1 + 2 (seed=42, weights 0.33/0.33/0.34)"
"$PY" -m langloc.retrieval.eval --cache_dir "$CACHE_DIR" --mode both

echo
echo "[REPRODUCE] running Table 3  (seed=42, weights 0.33/0.33/0.34)"
"$PY" -m langloc.retrieval.eval --cache_dir "$CACHE_DIR" --mode table3
