#!/bin/bash
# Reproduce paper Table 4(a) fine-localization numbers on the exact 100-scene
# subset used in the paper (3RScan).
#
# Two protocols are supported:
#
#   raw    — caption SceneGraph built directly from `visible_objects`
#            (structured GT shortcut; no API calls; fast smoke check).
#   parsed — caption SceneGraph built from `*_parsed.json` produced by
#            `langloc.dataset.annotation.parse_descriptions` (paper
#            protocol; one-time GPT-4o-mini precompute on the 1000
#            frames in the subset, ~$0.03, ~5 min wall-clock).
#
# Usage:
#   bash scripts/localization/reproduce_table4.sh raw
#   bash scripts/localization/reproduce_table4.sh parsed         # auto-runs precompute
#   bash scripts/localization/reproduce_table4.sh parsed --skip_precompute
#
# Reference numbers (paper Table 4(a) "LangLoc w/o dialog", 3RScan 100-scene):
#
#   Pos err mean       1.712 m         Pos err median     1.551 m
#   Top-10 mean        1.037 m         Top-10 median      0.941 m
#   Angle err mean     46.07°          Angle err median   37.24°
#   3D IoU mean        0.172
#
# Numbers logged to docs/experiments/<date>_table4_port_progress.md when a run
# is recorded; the paper subset list itself lives at
# manifests/3rscan_table4_subset_100.txt and is part of the repo.
set -euo pipefail

PROTOCOL="${1:-parsed}"
shift || true

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

SUBSET_FILE="manifests/3rscan_table4_subset_100.txt"
if [[ ! -f "$SUBSET_FILE" ]]; then
    echo "[ERROR] Paper-subset list missing: $SUBSET_FILE" >&2
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

# Precompute *_parsed.json on the paper subset if requested and missing.
if [[ "$PROTOCOL" == "parsed" && "$SKIP_PRECOMPUTE" == "0" ]]; then
    SCENE_ARGS=$(awk 'NF' "$SUBSET_FILE" | tr '\n' ' ')
    PARSED_PRESENT=$(find data/3RScan -path '*output/descriptions/*_parsed.json' | wc -l)
    if [[ "$PARSED_PRESENT" -lt 1000 ]]; then
        echo "[INFO] Found ${PARSED_PRESENT} *_parsed.json files (need 1000)."
        echo "[INFO] Running parse_descriptions on the paper subset (~5 min, ~\$0.03)..."
        "${PYTHON_BIN}" -m langloc.dataset.annotation.parse_descriptions \
            --data_root data/3RScan \
            --scene_ids $SCENE_ARGS \
            --workers 8 \
            --seed 42
    else
        echo "[INFO] Found ${PARSED_PRESENT} *_parsed.json files; skipping precompute."
    fi
fi

# Build the comma-separated scene list expected by Hydra.
SCENE_LIST=$(awk 'NF{printf "%s,", $1}' "$SUBSET_FILE" | sed 's/,$//')

OUT_FILE="eval/eval_metrics_table4_${PROTOCOL}.json"
mkdir -p eval

echo "[INFO] Running localization (protocol=${PROTOCOL}, seed=42, 100 scenes)..."
"${PYTHON_BIN}" -m langloc.localization.cli \
    localization=3rscan \
    localization.seed=42 \
    "localization.caption_source=${PROTOCOL}" \
    "+localization.scene_ids=[${SCENE_LIST}]" \
    localization.show_3d=false \
    localization.show_heatmap=false \
    localization.show_arrows=false \
    "localization.save_metrics=${OUT_FILE}" \
    "${EXTRA_ARGS[@]}"

echo "[INFO] Metrics saved to ${OUT_FILE}"
