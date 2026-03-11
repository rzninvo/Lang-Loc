#!/bin/bash
# Run midpoint baseline localization evaluator.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT" || exit 1

python -m langloc.localization.baseline_midpoint \
    "$@"
