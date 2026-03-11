#!/bin/bash
# Helper script to run localization evaluation with sensible defaults.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT" || exit 1

python -m langloc.localization.cli \
    localization.show_heatmap=true \
    localization.show_arrows=true \
    "$@"
