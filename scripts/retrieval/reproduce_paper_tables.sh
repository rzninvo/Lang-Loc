#!/bin/bash
# Reproduce paper Tables 1, 2, and 3 (scene-graph-to-scene-graph
# retrieval on ScanScribe queries against the 3DSSG database).
#
#   Tab. 1  — Top-{1,2,3,5} of 10 candidates  (ScanScribe-text queries)
#   Tab. 2  — Top-{5,10,20,30} of all test scenes (ScanScribe-text queries)
#   Tab. 3  — Top-{1,2,3,5,10} of 10 candidates (LLM-from-image queries,
#             same protocol as the whereami CLIP2CLIP / Text2SGM
#             baselines in their Table 4).
#
# Usage:
#   bash scripts/retrieval/reproduce_paper_tables.sh             # Tabs 1+2+3 (default)
#   bash scripts/retrieval/reproduce_paper_tables.sh tables12    # Tabs 1+2 only
#   bash scripts/retrieval/reproduce_paper_tables.sh table3      # Tab 3 only
#   bash scripts/retrieval/reproduce_paper_tables.sh all --skip_precompute
#
# Reference numbers (paper):
#
#   Tab. 1 (Top-k of 10, ScanScribe-text)
#     Top-1: 76.70   Top-2: 90.40   Top-3: 96.10   Top-5: 98.90
#   Tab. 2 (Top-k of all, ScanScribe-text)
#     Top-5: 83.30   Top-10: 91.60  Top-20: 97.10  Top-30: 98.80
#   Tab. 3 (Top-k of 10, LLM-from-image — corrected fair-comparison protocol)
#     Top-1 ~76% (paper-reported); reproduction lands at 59–62% — see
#     the rebuttal report for the data-mismatch discussion.
#
# First-call cost:
#   - Precompute caches: ~3 min on GPU, ~10 min CPU (one-time).
#   - Eval:              ~3 s per table once caches exist.
#
# Required inputs (download instructions in README §"Data layout"):
#   - data/model_checkpoints/graph2graph/paper/epoch_70_163_cliprel.pth
#   - data/processed_data/scanscribe/scanscribe_text_graphs_test_518D.pt
#   - data/processed_data/scanscribe/scanscribe_text_graphs_from_image_desc_node_edge_features.pt
#   - data/processed_data/scanscribe/scanscribe_cleaned_original_518D.pt
#   - data/processed_data/3dssg/3dssg_graphs_518D.pt
set -euo pipefail

case "${1:-}" in
    -h|--help)
        sed -n '2,36p' "$0" | sed -e 's/^# \?//'
        exit 0
        ;;
esac

WHICH="${1:-all}"
case "$WHICH" in
    all|tables12|table3) shift || true ;;
    --*)                 WHICH="all" ;;   # forwarded flag, not a selector
    *)
        echo "[ERROR] Unknown selector '$WHICH'. Use 'all', 'tables12', or 'table3'." >&2
        exit 1
        ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT" || exit 1

# Pick the langloc conda env's python; override with PYTHON_BIN.
PYTHON_BIN="${PYTHON_BIN:-$HOME/miniconda3/envs/langloc/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
    if command -v python >/dev/null 2>&1; then
        PYTHON_BIN="$(command -v python)"
    else
        echo "[ERROR] Cannot locate a Python interpreter. Set PYTHON_BIN." >&2
        exit 1
    fi
fi

CACHE_DIR="${CACHE_DIR:-data/processed_data/eval_pool}"
CHECKPOINT="${CHECKPOINT:-data/model_checkpoints/graph2graph/paper/epoch_70_163_cliprel.pth}"
SCANSCRIBE_DIR="${SCANSCRIBE_DIR:-data/processed_data/scanscribe}"
DEVICE="${DEVICE:-cuda}"

QUERY_TEXT="${SCANSCRIBE_DIR}/scanscribe_text_graphs_test_518D.pt"
QUERY_IMG="${SCANSCRIBE_DIR}/scanscribe_text_graphs_from_image_desc_node_edge_features.pt"

SKIP_PRECOMPUTE=0
EXTRA_ARGS=()
for arg in "$@"; do
    case "$arg" in
        --skip_precompute) SKIP_PRECOMPUTE=1 ;;
        *)                 EXTRA_ARGS+=("$arg") ;;
    esac
done

mkdir -p "$CACHE_DIR"

precompute () {
    # $1: query .pt path
    # $2: cache suffix ("" for text, "_img" for LLM-from-image)
    # $3: human label
    local QPATH="$1" SUFFIX="$2" LABEL="$3"

    if [[ ! -f "$QPATH" ]]; then
        echo "[ERROR] Query graphs missing for ${LABEL}: ${QPATH}" >&2
        echo "        See README §\"Data layout\" for the canonical download." >&2
        exit 1
    fi
    if [[ ! -f "$CHECKPOINT" ]]; then
        echo "[ERROR] Retrieval checkpoint missing: ${CHECKPOINT}" >&2
        exit 1
    fi

    local DB_CACHE="${CACHE_DIR}/db_emb_cache.pt"
    local Q_CACHE="${CACHE_DIR}/query_emb_cache${SUFFIX}.pt"

    if [[ -f "$Q_CACHE" && "$SKIP_PRECOMPUTE" == "1" ]]; then
        echo "[INFO] ${LABEL}: cache present and --skip_precompute set; skipping."
        return 0
    fi

    local SKIP_DB_FLAG=()
    if [[ -f "$DB_CACHE" ]]; then
        # Reuse the existing DB cache — it's shared by Tabs 1+2+3.
        SKIP_DB_FLAG=(--skip_db)
    fi

    echo "[INFO] ${LABEL}: precomputing embeddings (~3 min on GPU)…"
    "${PYTHON_BIN}" -m scripts.retrieval.precompute_eval_embeddings \
        --checkpoint "$CHECKPOINT" \
        --cache_dir  "$CACHE_DIR" \
        --query_path "$QPATH" \
        --cache_suffix "$SUFFIX" \
        --device "$DEVICE" \
        --seed 42 \
        "${SKIP_DB_FLAG[@]}"
}

eval_table () {
    # $1: --mode value  ($2: suffix, $3: label)
    local MODE="$1" SUFFIX="$2" LABEL="$3"
    echo
    echo "================================================================"
    echo "[INFO] ${LABEL}  (--mode ${MODE}, seed=42)"
    echo "================================================================"
    "${PYTHON_BIN}" -m langloc.retrieval.eval \
        --cache_dir "$CACHE_DIR" \
        --mode "$MODE" \
        --query_cache_suffix "$SUFFIX" \
        --seed 42 \
        "${EXTRA_ARGS[@]}"
}

if [[ "$WHICH" == "all" || "$WHICH" == "tables12" ]]; then
    precompute "$QUERY_TEXT" ""    "Tables 1+2 (ScanScribe-text queries)"
    eval_table "both"        ""    "Tables 1 + 2"
fi

if [[ "$WHICH" == "all" || "$WHICH" == "table3" ]]; then
    precompute "$QUERY_IMG"  "_img" "Table 3 (LLM-from-image queries)"
    eval_table "table3"      "_img" "Table 3"
fi

echo
echo "[INFO] Done."
