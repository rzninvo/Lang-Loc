"""Typed configuration dataclass for the dialogue system.

Follows the ``NBVConfig`` pattern from ``langloc.utils.nbv_config`` — a
dataclass consolidating all runtime parameters with an
``extract_dialogue_config`` factory that marshals a Hydra
``DictConfig`` or ``argparse.Namespace`` into the typed container.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Any, List, Union

from omegaconf import DictConfig


@dataclass
class DialogueConfig:
    """Configuration parameters for the pose-level dialogue system.

    Attributes:
        candidates_json: Path to the evaluation candidates JSON file.
        dataset_root: Root directory of the 3RScan (or similar) dataset.
        only_scene_id: If non-empty, restrict evaluation to this scene.
        limit: Maximum number of entries to evaluate (0 = unlimited).

        eval_mode: ``"sequential"`` (each backend independent) or
            ``"shared"`` (single driver updates all).
        backend_order: Order in which backends are run in sequential mode.
        cache_answers: Reuse answers for identical questions across backends.

        candidate_set: Which candidate set to use
            (``"auto"`` | ``"grid"`` | ``"fov"`` | ``"both"``).
        include_predicted_pose: Append predicted pose as extra candidate.
        pred_candidate_prior: Prior weight for the predicted pose candidate.

        k_nn: Number of nearest-neighbour frames for candidate→frame mapping.
        sigma: Gaussian kernel bandwidth for candidate→frame weighting.
        use_direction: Include direction similarity in mapping.
        dir_temp: Temperature for direction-based mapping.

        max_rounds: Maximum dialogue rounds per backend.
        min_rounds: Minimum rounds before early stopping.
        conf_threshold: Posterior confidence threshold for early stopping.
        auto_relax: Progressively widen thresholds if no questions pass.
        ask_min_p: Minimum P(yes) threshold for asking a question.
        ask_max_p: Maximum P(yes) threshold for asking a question.

        question_strategy: Selection strategy
            (``"ig"`` | ``"binary"`` | ``"least_first"``).
        question_driver: Driver backend for shared mode.
        rel_min_answerable: Minimum answerable probability for relations.
        rel_bonus: Information-gain bonus multiplier for relations.
        rel_prefer_margin: Margin for preferring relations over labels.

        idf_weight: Weight applied to IDF-based label scoring.
        ignore_labels: Labels to exclude from questioning.

        rel_min_salience: Minimum salience for relation pool inclusion.
        rel_unique_only: Keep only unique relation triples.
        max_pool_frames: Maximum number of frames in the pool.
        max_rel_pool: Maximum number of relations in the pool.
        allowed_rels: If non-empty, only allow these relation predicates.

        alpha_label: Likelihood calibration for label questions.
        alpha_rel: Likelihood calibration for relation questions.
        p_u_label: Base unknown probability for labels.
        p_u_rel: Base unknown probability for relations.
        p_u_unanswerable: Unknown probability for unanswerable hypotheses.
        vis_tau: Salience → visibility scale parameter.
        ans_tau: Salience → answerable scale parameter.

        n_particles: Number of particles for the A2 backend.
        p_k_nn: KNN parameter for particle-frame lookups.
        p_sigma: Gaussian bandwidth for particle-frame weighting.
        p_jitter: Positional jitter added to particles at resampling.
        seed: Random seed for the particle filter.

        answer_mode: ``"interactive"`` or ``"oracle"``.
        oracle_ansable_min: Below this answerable threshold the oracle
            responds ``"u"`` (unknown).

        show_top_n: Number of top entries to display.
        show_gt_debug: Print ground-truth debug information.
    """

    # --- I/O ---
    candidates_json: str = ""
    dataset_root: str = ""
    only_scene_id: str = ""
    limit: int = 1

    # --- Evaluation mode ---
    eval_mode: str = "sequential"
    backend_order: List[str] = field(default_factory=lambda: ["a1", "a2", "a3"])
    cache_answers: bool = False

    # --- Candidates ---
    candidate_set: str = "both"
    include_predicted_pose: bool = False
    pred_candidate_prior: float = 0.35

    # --- Mapping ---
    k_nn: int = 15
    sigma: float = 0.25
    use_direction: bool = False
    dir_temp: float = 0.25

    # --- Dialogue loop ---
    max_rounds: int = 12
    min_rounds: int = 2
    conf_threshold: float = 0.85
    auto_relax: bool = False
    ask_min_p: float = 0.01
    ask_max_p: float = 0.99

    # --- Question selection ---
    question_strategy: str = "ig"
    question_driver: str = "a3"
    rel_min_answerable: float = 0.10
    rel_bonus: float = 0.25
    rel_prefer_margin: float = 0.05

    # --- Label filtering / IDF ---
    idf_weight: float = 1.0
    ignore_labels: List[str] = field(
        default_factory=lambda: ["floor", "wall", "ceiling", "room", "baseboard", "carpet"],
    )

    # --- Pools ---
    rel_min_salience: float = 0.0
    rel_unique_only: bool = False
    max_pool_frames: int = 30
    max_rel_pool: int = 600
    allowed_rels: List[str] = field(default_factory=list)

    # --- Likelihood calibration ---
    alpha_label: float = 0.82
    alpha_rel: float = 0.70
    p_u_label: float = 0.05
    p_u_rel: float = 0.15
    p_u_unanswerable: float = 0.90
    vis_tau: float = 0.20
    ans_tau: float = 0.10

    # --- A2 particle filter ---
    n_particles: int = 256
    p_k_nn: int = 10
    p_sigma: float = 0.25
    p_jitter: float = 0.07
    seed: int = 42  # canonical project seed

    # --- Answering ---
    answer_mode: str = "interactive"
    oracle_ansable_min: float = 0.25

    # --- Display ---
    show_top_n: int = 5
    show_gt_debug: bool = False


def extract_dialogue_config(
    source: Union[DictConfig, argparse.Namespace, Any],
) -> DialogueConfig:
    """Create a :class:`DialogueConfig` from a Hydra config or argparse namespace.

    All fields are read from *source* by name.  Missing attributes fall
    back to the dataclass defaults.  Supports both ``DictConfig`` (from
    Hydra) and ``argparse.Namespace`` (from standalone CLI).

    Args:
        source: Hydra ``DictConfig`` section (e.g. ``cfg.dialogue``),
            or parsed ``argparse.Namespace``.

    Returns:
        Populated :class:`DialogueConfig` instance.
    """
    kwargs = {}
    for f in DialogueConfig.__dataclass_fields__:
        if isinstance(source, DictConfig):
            if f in source:
                val = source[f]
                # OmegaConf lists → plain Python lists for dataclass fields
                if isinstance(val, (list, DictConfig)):
                    from omegaconf import OmegaConf
                    val = OmegaConf.to_container(val, resolve=True)
                kwargs[f] = val
        else:
            if hasattr(source, f):
                kwargs[f] = getattr(source, f)
    return DialogueConfig(**kwargs)
