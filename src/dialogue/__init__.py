"""Pose-level dialogue system for interactive localization.

This package provides Bayesian dialogue backends and evaluation tools
for refining camera pose estimates through yes/no/unknown questions
about visible scene semantics.

Modules:
    math_utils: Vector math, Bayesian updates, and pose error computation.
    semantics: Frame label salience and spatial relation extraction.
    candidates: Candidate pose extraction from evaluation entries.
    likelihood: Yes/no/unknown likelihood model for labels and relations.
    question_pool: Question representation, pool building, and IDF weighting.
    backends: Bayesian inference backends (candidate, particle, frame).
    question_selection: Information-gain question policy and display helpers.
    dialogue_runner: Per-backend dialogue loop (interactive and oracle).
    eval_runner: Batch evaluation entry point with aggregation.
    render_gt: Ground-truth image rendering and index generation.
    dialogue_config: Typed configuration dataclass for dialogue parameters.
"""
