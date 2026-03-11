#!/bin/bash
# Run dialogue disambiguation evaluation.
#
# Requires candidate poses from localization (run run_candidates.sh first).
#
# Usage:
#   bash scripts/dialogue/run_eval.sh
#
# Override any Hydra parameter via CLI:
#   bash scripts/dialogue/run_eval.sh dialogue.max_rounds=8

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT" || exit 1

python -m langloc.dialogue.cli \
    "$@"
