#!/bin/bash
# Export localization candidate poses for downstream dialogue evaluation.
#
# Runs localization in candidates mode (no metrics computed) and writes
# ranked grid poses to a JSON file for use by the dialogue system.
#
# Usage:
#   bash scripts/localization/run_candidates.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT" || exit 1

python -m langloc.localization.cli \
    localization.mode=candidates \
    "$@"
