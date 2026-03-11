#!/bin/bash
# Run fine localization evaluation (standard mode).
#
# Evaluates position error, angular error, Hit@r, mass-radius, and 3D View IoU
# across all scenes found in the configured data directories.
#
# Usage:
#   bash scripts/localization/run_eval.sh
#
# Override any Hydra parameter via CLI:
#   bash scripts/localization/run_eval.sh localization.grid_step=0.5 localization.dataset=scannet

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT" || exit 1

python -m langloc.localization.cli \
    localization.mode=standard \
    "$@"
