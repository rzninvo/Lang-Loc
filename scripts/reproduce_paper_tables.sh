#!/bin/bash
# Reproduce paper Tables 1+2 + corrected Table 3 (scene retrieval Recall@k
# on ScanScribe).
#
#   Table 1: 10-candidate pool, ScanScribe-text queries (Top-1/2/3/5)
#   Table 2: full-test-set pool, ScanScribe-text queries (Top-5/10/20/30)
#   Table 3: 10-candidate pool, LLM-from-image queries (Top-1/2/3/5)
#            ↑ CORRECTED — the published Table 3 (76.10) was inadvertently
#              produced on canonical ScanScribe text queries; this script
#              evaluates on the actual LLM-from-image queries that match the
#              whereami Table 4 protocol used for the CLIP2CLIP / Text2SGM
#              baselines. See
#              docs/reports/2026-05-05/19_table3_corrected_for_rebuttal.md
#
# Modes:
#   --use_cache (default): use the pre-computed _img caches that already
#                          point at the LLM-from-image queries.  Fast (~3 sec).
#   --rebuild_cache:       re-run the V2 + SimpleGraphMatcher forward pass to
#                          produce fresh caches (~3-5 min on a CUDA box).
#
# All modes use seed=42 and `langloc/retrieval/eval.py`.  Eq. 8 weights
# 0.33/0.33/0.34, 218-scene ScanScribe distractor pool, 10 outer × 100 inner
# rounds.
#
# Default cache_dir is `data/processed_data/eval_pool/`.
# Override IMG_QUERY_PATH to point at a different LLM-from-image query .pt.

set -euo pipefail

REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

CACHE_DIR="${CACHE_DIR:-data/processed_data/eval_pool}"
CHECKPOINT="${CHECKPOINT:-data/model_checkpoints/graph2graph/paper/epoch_70_163_cliprel.pth}"
# Default Table-3 query source = the actual LLM-from-image graphs (the same
# file whereami's eval.py:55 loads to produce its CLIP2CLIP / Text2SGM
# Table-4 baselines). This is the FAIR-comparison protocol.
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
echo "[REPRODUCE] table3 query source: $IMG_QUERY_PATH"

if [ "$MODE" = "rebuild_cache" ]; then
    if [ ! -f "$CHECKPOINT" ]; then
        echo "[ERROR] checkpoint not found: $CHECKPOINT" >&2
        exit 1
    fi
    echo "[REPRODUCE] step 1/2: rebuilding Tables 1+2 caches (canonical ScanScribe text)"
    "$PY" -m scripts.retrieval.precompute_eval_embeddings \
        --checkpoint "$CHECKPOINT" \
        --cache_dir  "$CACHE_DIR" \
        --device cuda

    echo "[REPRODUCE] step 2/2: rebuilding Table 3 _img cache from LLM-from-image queries"
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
echo "[REPRODUCE] running Table 3 — corrected fair comparison (seed=42, weights 0.33/0.33/0.34)"
"$PY" -m langloc.retrieval.eval --cache_dir "$CACHE_DIR" --mode table3
