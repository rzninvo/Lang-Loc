#!/bin/bash
# Reproduce paper Table 5 (= paper Supp Table 8 = Master Table 6.6)
# fine-localization numbers on the full LangLoc dataset (3RScan, 1319
# scenes that have BOTH descriptions and a 3DSSG scene graph).
#
# This is the full-dataset counterpart to scripts/localization/
# reproduce_table4.sh — the protocol differs in two ways from Table 4:
#
#   1. Scene set:        all 1319 eligible scenes (vs the 100-scene
#                        subset for Table 4).  Listed in
#                        manifests/3rscan_table5_full.txt.
#   2. Frame policy:     `all` (paper Supp Table 7 underlines `all`
#                        for the complete-dataset evaluation; Table 4
#                        used `max_visible`).  Each parsed frame in
#                        a scene contributes one independent eval and
#                        the per-scene metric is averaged over its
#                        frames.
#
# The caption protocol is fixed at `parsed` (paper protocol — re-parse
# the natural-language description via GPT-4o-mini, ground to
# `visible_objects` at γ=0.7).  `raw` is not exposed for Table 5
# because that combination is not what the paper reports.
#
# Usage:
#   bash scripts/localization/reproduce_table5.sh           # auto-runs precompute
#   bash scripts/localization/reproduce_table5.sh --skip_precompute
#
# Reference numbers (paper Table 5 / Master Table 6.6, "LangLoc w/o dialog"):
#
#   Top-10 Pos    Mean 1.153   Med 0.951
#   Pos           Mean 1.534   Med 1.308
#   Angle         Mean 46.85°  Med 39.80°
#   3D IoU        Mean 0.147
#
# Cost on first call:
#   - Precompute: 1319 scenes × ~10 frames = ~13 000 frames through
#     GPT-4o-mini.  ~$0.40, ~30 min wall-clock at `--workers 8`.
#   - Eval:       ~13k frame-evals × ~0.6 s = ~2 h wall-clock.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT" || exit 1

PYTHON_BIN="${PYTHON_BIN:-$HOME/miniconda3/envs/langloc/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
    if command -v python >/dev/null 2>&1; then
        PYTHON_BIN="$(command -v python)"
    else
        echo "[ERROR] Cannot locate a Python interpreter.  Set PYTHON_BIN." >&2
        exit 1
    fi
fi

SUBSET_FILE="manifests/3rscan_table5_full.txt"
DATA_ROOT="data/3RScan"
if [[ ! -f "$SUBSET_FILE" ]]; then
    echo "[ERROR] Full-dataset manifest missing: $SUBSET_FILE" >&2
    echo "        Rebuild it from the intersection of description-bearing" >&2
    echo "        scenes under $DATA_ROOT and the 3DSSG scene-graph DB." >&2
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

# Precompute *_parsed.json for the 1319 scenes' frames if needed.  The
# precompute script skips frames that already have a *_parsed.json
# sibling, so re-running is cheap and idempotent.
if [[ "$SKIP_PRECOMPUTE" == "0" ]]; then
    SCENE_ARGS=$(awk 'NF' "$SUBSET_FILE" | tr '\n' ' ')
    PARSED_PRESENT=$(find "$DATA_ROOT" -path '*output/descriptions/*_parsed.json' 2>/dev/null | wc -l)
    SOURCE_PRESENT=$(find "$DATA_ROOT" -path '*output/descriptions/frame-*.json' -not -name '*_parsed.json' 2>/dev/null | wc -l)
    echo "[INFO] ${PARSED_PRESENT} *_parsed.json present, ${SOURCE_PRESENT} source frame JSONs."
    if [[ "$PARSED_PRESENT" -lt "$SOURCE_PRESENT" ]]; then
        echo "[INFO] Running parse_descriptions on the full subset (~30 min, ~\$0.40)..."
        "${PYTHON_BIN}" -m langloc.dataset.annotation.parse_descriptions \
            --data_root "$DATA_ROOT" \
            --scene_ids $SCENE_ARGS \
            --workers 8 \
            --seed 42
    else
        echo "[INFO] Precompute already complete; skipping."
    fi
fi

SCENE_LIST=$(awk 'NF{printf "%s,", $1}' "$SUBSET_FILE" | sed 's/,$//')
OUT_FILE="eval/eval_metrics_table5.json"
mkdir -p eval

echo "[INFO] Running Table 5 evaluation:"
echo "       dataset      = 3rscan"
echo "       protocol     = parsed (paper §3.3 / Supp §4.3)"
echo "       frame_policy = all (Supp Table 7 underlined value)"
echo "       seed         = 42"
echo "       scenes       = $(wc -l < "$SUBSET_FILE")"

"${PYTHON_BIN}" -m langloc.localization.cli \
    localization=3rscan \
    localization.seed=42 \
    localization.caption_source=parsed \
    localization.frame_policy=all \
    "+localization.scene_ids=[${SCENE_LIST}]" \
    localization.show_3d=false \
    localization.show_heatmap=false \
    localization.show_arrows=false \
    "localization.save_metrics=${OUT_FILE}" \
    "${EXTRA_ARGS[@]}"

echo "[INFO] Metrics saved to ${OUT_FILE}"
