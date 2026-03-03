#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT" || exit 1

CUDA_VISIBLE_DEVICES=0 python3 -m langloc.graph_matching.eval \
    eval.eval_iters=10 \
    'eval.valid_top_k=[1,2,3,5]' \
    eval.model_name=model_NO_subg_100_epochs_entire_training_set_epoch_30_checkpoint
