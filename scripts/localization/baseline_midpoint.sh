#!/bin/bash
# Run midpoint baseline localization evaluator.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT" || exit 1

python -m langloc.localization.baseline_midpoint \
    localization.frame_policy=max_visible \
    baseline.random_pitch_deg=30.0 \
    baseline.save_metrics=./eval/baseline_eval_metrics_mid_point.json \
    baseline.log_file=./eval/baseline_eval_loc_summary_mid_point.log
