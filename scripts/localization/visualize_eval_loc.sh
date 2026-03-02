#!/bin/bash
# Helper script to run localization evaluation with sensible defaults.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT" || exit 1

python -m src.localization.cli \
    localization.save_metrics=./eval/eval_metrics.json \
    localization.log_file=./eval/eval_loc_summary.log \
    localization.frame_policy=max_visible \
    localization.top_k=10 \
    localization.grid_step=0.25 \
    localization.prediction_strategy=weighted
