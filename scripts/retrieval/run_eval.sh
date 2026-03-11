#!/bin/bash
# Run scene retrieval evaluation (BigGNN graph matching).
#
# Evaluates Recall@k on ScanScribe, human, and ScanNet test sets.
# Requires a trained model checkpoint in data/model_checkpoints/graph2graph/.
#
# Usage:
#   bash scripts/retrieval/run_eval.sh
#
# Override model name or parameters:
#   bash scripts/retrieval/run_eval.sh eval.model_name=my_model eval.eval_iters=20

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT" || exit 1

python3 -m langloc.graph_matching.eval \
    eval.eval_iters=10 \
    'eval.valid_top_k=[1,2,3,5]' \
    eval.model_name=model_NO_subg_100_epochs_entire_training_set_epoch_30_checkpoint \
    "$@"
