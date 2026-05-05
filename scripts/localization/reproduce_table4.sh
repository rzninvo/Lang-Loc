#!/bin/bash
# Reproduce paper Table 4 fine-localization numbers on the 100-scene subsets
# the paper evaluates (3RScan and ScanNet).
#
# Two datasets supported (Table 4 row):
#
#   3rscan  — Table 4(a). Subset list:
#             manifests/3rscan_table4_subset_100.txt (sourced from
#             whereami's `subset_100_scene_ids.txt`).
#   scannet — Table 4(b). Subset list:
#             manifests/scannet_table4_first_100.txt (the first 100
#             entries of configs/manifests/scannetv2_all.txt).
#
# Two protocols supported:
#
#   raw    — caption SceneGraph built directly from `visible_objects`
#            (structured GT shortcut; no API calls; fast smoke check).
#   parsed — caption SceneGraph built from `*_parsed.json` produced by
#            `langloc.dataset.annotation.parse_descriptions` (paper
#            protocol; one-time GPT-4o-mini precompute, ~$0.03/100
#            scenes, ~5 min wall-clock).
#
# Usage:
#   bash scripts/localization/reproduce_table4.sh raw                  # 3rscan / raw
#   bash scripts/localization/reproduce_table4.sh parsed               # 3rscan / parsed
#   bash scripts/localization/reproduce_table4.sh raw scannet          # 4(b) raw
#   bash scripts/localization/reproduce_table4.sh parsed scannet       # 4(b) parsed
#   bash scripts/localization/reproduce_table4.sh parsed scannet --skip_precompute
#
# Reference numbers (paper Table 4 "LangLoc w/o dialog"):
#
#   3RScan  — Pos mean 1.712 / med 1.551 m   Angle mean 46.07° / med 37.24°
#             Top-10 mean 1.037 / med 0.941  3D IoU mean 0.172
#   ScanNet — Pos mean 1.382 / med 1.099 m   Angle mean 42.67° / med 34.66°
#             Top-10 mean 1.254 / med 1.065  3D IoU mean 0.236
set -euo pipefail

PROTOCOL="${1:-parsed}"
shift || true
DATASET="${1:-3rscan}"
case "$DATASET" in
    3rscan|scannet) shift || true ;;
    --*)            DATASET="3rscan" ;;  # Forwarded flag, not a dataset.
    *)              shift || true ;;     # Unknown — leave as-is and forward.
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT" || exit 1

# Pick the langloc conda env's python explicitly so the spaCy model and
# project dependencies are available regardless of the user's $PATH.
# Override with PYTHON_BIN=/path/to/python if installed differently.
PYTHON_BIN="${PYTHON_BIN:-$HOME/miniconda3/envs/langloc/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
    if command -v python >/dev/null 2>&1; then
        PYTHON_BIN="$(command -v python)"
    else
        echo "[ERROR] Cannot locate a Python interpreter.  Set PYTHON_BIN." >&2
        exit 1
    fi
fi

case "$DATASET" in
    3rscan)
        SUBSET_FILE="manifests/3rscan_table4_subset_100.txt"
        DATA_ROOT="data/3RScan"
        OVERLAY="3rscan"
        ;;
    scannet)
        SUBSET_FILE="manifests/scannet_table4_first_100.txt"
        DATA_ROOT="data/scans"
        OVERLAY="scannet"
        ;;
    *)
        echo "[ERROR] Unknown dataset '$DATASET'.  Use '3rscan' or 'scannet'." >&2
        exit 1
        ;;
esac

if [[ ! -f "$SUBSET_FILE" ]]; then
    echo "[ERROR] Subset list missing: $SUBSET_FILE" >&2
    exit 1
fi

SKIP_PRECOMPUTE=0
EXTRA_ARGS=()
for arg in "$@"; do
    case "$arg" in
        --skip_precompute) SKIP_PRECOMPUTE=1 ;;
        *)                 EXTRA_ARGS+=("$arg") ;;
    esac
done

# Precompute *_parsed.json over the subset's frames if needed.
if [[ "$PROTOCOL" == "parsed" && "$SKIP_PRECOMPUTE" == "0" ]]; then
    SCENE_ARGS=$(awk 'NF' "$SUBSET_FILE" | tr '\n' ' ')
    PARSED_PRESENT=$(find "$DATA_ROOT" -path '*output/descriptions/*_parsed.json' 2>/dev/null | wc -l)
    NEEDED=$(($(wc -l < "$SUBSET_FILE") * 10))  # ~10 frames/scene
    if [[ "$PARSED_PRESENT" -lt "$NEEDED" ]]; then
        echo "[INFO] Found ${PARSED_PRESENT} *_parsed.json under ${DATA_ROOT}; running precompute (~5 min, ~\$0.03)..."
        "${PYTHON_BIN}" -m langloc.dataset.annotation.parse_descriptions \
            --data_root "$DATA_ROOT" \
            --scene_ids $SCENE_ARGS \
            --workers 8 \
            --seed 42
    else
        echo "[INFO] Found ${PARSED_PRESENT} *_parsed.json files under ${DATA_ROOT}; skipping precompute."
    fi
fi

# Build the comma-separated scene list expected by Hydra.
SCENE_LIST=$(awk 'NF{printf "%s,", $1}' "$SUBSET_FILE" | sed 's/,$//')

OUT_FILE="eval/eval_metrics_table4_${DATASET}_${PROTOCOL}.json"
mkdir -p eval

echo "[INFO] Running localization (dataset=${DATASET}, protocol=${PROTOCOL}, seed=42, 100 scenes)..."
"${PYTHON_BIN}" -m langloc.localization.cli \
    "localization=${OVERLAY}" \
    localization.seed=42 \
    "localization.caption_source=${PROTOCOL}" \
    "+localization.scene_ids=[${SCENE_LIST}]" \
    localization.show_3d=false \
    localization.show_heatmap=false \
    localization.show_arrows=false \
    "localization.save_metrics=${OUT_FILE}" \
    "${EXTRA_ARGS[@]}"

echo "[INFO] Metrics saved to ${OUT_FILE}"
