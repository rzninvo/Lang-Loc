"""Information-gain question selection policy and display helpers.

Provides the strategy for choosing the next question to ask during a
dialogue round.  Supports information-gain (IG), binary-uncertainty,
and least-first strategies, with IDF weighting and relation preference.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.dialogue.likelihood import ynu_likelihood_from_prob
from src.dialogue.math_utils import bayes_update, entropy
from src.dialogue.question_pool import Question
from src.dialogue.semantics import rel_item_to_tuple


# ---------------------------------------------------------------------------
# Question key (for dedup / caching)
# ---------------------------------------------------------------------------
def question_key(
    q: Question,
    label_pool: List[str],
    rel_pool: List[Any],
) -> Tuple:
    """Return a canonical hashable key for a question.

    Args:
        q: The question.
        label_pool: Label pool for index resolution.
        rel_pool: Relation pool for index resolution.

    Returns:
        Tuple suitable for use as a dictionary key.
    """
    if q.qtype == "label":
        return ("label", label_pool[q.idx])
    s, r, o = rel_item_to_tuple(rel_pool[q.idx])
    return ("rel", s, r, o)


# ---------------------------------------------------------------------------
# Expected information gain
# ---------------------------------------------------------------------------
def compute_ig_for_question(
    p: np.ndarray,
    p_true: np.ndarray,
    p_ans: np.ndarray,
    alpha: float,
    p_u_base: float,
    p_u_unanswerable: float,
) -> float:
    """Compute expected information gain for a single question.

    Args:
        p: Current posterior vector.
        p_true: Per-hypothesis truth probability.
        p_ans: Per-hypothesis answerable probability.
        alpha: Likelihood calibration parameter.
        p_u_base: Base unknown probability.
        p_u_unanswerable: Unknown probability for unanswerable hypotheses.

    Returns:
        Expected information gain in nats.
    """
    H0 = entropy(p)
    Py, Pn, Pu = ynu_likelihood_from_prob(p_true, p_ans, alpha, p_u_base, p_u_unanswerable)
    # P(answer)
    P_y = float(np.dot(p, Py))
    P_n = float(np.dot(p, Pn))
    P_u = float(np.dot(p, Pu))

    Hexp = 0.0
    if P_y > 1e-12:
        Hexp += P_y * entropy(bayes_update(p, Py))
    if P_n > 1e-12:
        Hexp += P_n * entropy(bayes_update(p, Pn))
    if P_u > 1e-12:
        Hexp += P_u * entropy(bayes_update(p, Pu))

    return H0 - Hexp


# ---------------------------------------------------------------------------
# System-level question picker
# ---------------------------------------------------------------------------
def pick_next_question_system(
    backend_name: str,
    backend: Any,
    questions: List[Question],
    label_pool: List[str],
    rel_pool: List[Any],
    idf: Dict[str, float],
    cfg: Any,
) -> Optional[Question]:
    """Select the best next question for a given backend.

    Strategies:
      - ``ig``: maximum expected information gain (recommended).
      - ``binary``: closest ``P(yes)`` to 0.5.
      - ``least_first``: most extreme ``P(yes)``.

    Labels in ``cfg.ignore_labels`` are skipped.  Relations are
    preferred over labels when their score is within
    ``cfg.rel_prefer_margin`` of the best label score.

    Args:
        backend_name: Name of the backend (for logging only).
        backend: Backend instance exposing ``posterior_vector``,
            ``label_prob_yes``, and ``rel_prob_true_and_answerable``.
        questions: Remaining candidate questions.
        label_pool: Label pool for index resolution.
        rel_pool: Relation pool for index resolution.
        idf: Per-label IDF weights.
        cfg: Dialogue configuration (``DialogueConfig`` or argparse
            namespace) with strategy parameters.

    Returns:
        The best :class:`Question`, or ``None`` if no question passes
        the threshold filters.
    """
    st = cfg.question_strategy.lower().strip()
    p = backend.posterior_vector()

    ignore = set([x.strip().lower() for x in cfg.ignore_labels])

    best_rel: Tuple[Optional[Question], float] = (None, -1e18)
    best_lab: Tuple[Optional[Question], float] = (None, -1e18)

    def passes_thresholds(p_yes: float, p_ans: float, is_rel: bool) -> bool:
        if not (cfg.ask_min_p <= p_yes <= cfg.ask_max_p):
            return False
        if is_rel and p_ans < cfg.rel_min_answerable:
            return False
        return True

    for q in questions:
        if q.qtype == "label":
            lab = label_pool[q.idx]
            if lab in ignore:
                continue

            p_true, p_ans = backend.label_prob_yes(lab)
            p_yes = float(np.dot(p, p_true))
            p_ans_avg = float(np.dot(p, p_ans))
            if not passes_thresholds(p_yes, 1.0, False):
                continue

            if st == "ig":
                score = compute_ig_for_question(
                    p=p,
                    p_true=p_true,
                    p_ans=p_ans,
                    alpha=cfg.alpha_label,
                    p_u_base=cfg.p_u_label,
                    p_u_unanswerable=cfg.p_u_unanswerable,
                )
                # IDF boost
                score *= (1.0 + cfg.idf_weight * float(idf.get(lab, 0.0)))
            elif st == "binary":
                score = -abs(p_yes - 0.5)
            elif st in ("least_first", "least-first", "least"):
                score = -min(p_yes, 1.0 - p_yes)
            else:
                raise ValueError(f"Unknown question_strategy: {cfg.question_strategy}")

            if score > best_lab[1]:
                best_lab = (q, score)

        else:
            s, r, o = rel_item_to_tuple(rel_pool[q.idx])
            p_true, p_ans = backend.rel_prob_true_and_answerable((s, r, o))
            p_yes = float(np.dot(p, p_true))
            p_ans_avg = float(np.dot(p, p_ans))
            if not passes_thresholds(p_yes, p_ans_avg, True):
                continue

            if st == "ig":
                score = compute_ig_for_question(
                    p=p,
                    p_true=p_true,
                    p_ans=p_ans,
                    alpha=cfg.alpha_rel,
                    p_u_base=cfg.p_u_rel,
                    p_u_unanswerable=cfg.p_u_unanswerable,
                )
                score *= (1.0 + cfg.rel_bonus)
            elif st == "binary":
                score = -abs(p_yes - 0.5)
            elif st in ("least_first", "least-first", "least"):
                score = -min(p_yes, 1.0 - p_yes)
            else:
                raise ValueError(f"Unknown question_strategy: {cfg.question_strategy}")

            if score > best_rel[1]:
                best_rel = (q, score)

    # prefer relation if it exists and not terrible
    if best_rel[0] is not None and best_rel[1] >= best_lab[1] - cfg.rel_prefer_margin:
        return best_rel[0]
    if best_lab[0] is not None:
        return best_lab[0]
    if best_rel[0] is not None:
        return best_rel[0]
    return None


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------
def show_top_frames(
    frames: Sequence[Any],
    pf: np.ndarray,
    top_n: int = 5,
    title: str = "Top frames",
) -> None:
    """Print the top frames by posterior probability.

    Args:
        frames: Frame objects in the pool.
        pf: Posterior vector over frames.
        top_n: Number of frames to display.
        title: Header string.
    """
    order = np.argsort(-pf)[: min(top_n, len(pf))]
    print(title + ":")
    for k, j in enumerate(order, 1):
        fr = frames[int(j)]
        fid = getattr(fr, "frame_id", str(j))
        pos = getattr(fr, "position", getattr(fr, "pos", None))
        pos_list = pos.tolist() if hasattr(pos, "tolist") else (list(pos) if pos is not None else None)
        print(f"  {k:>2}. {fid:<12} P={pf[int(j)]:.3f} pos={pos_list}")


def show_top_candidates(
    cand_pos: np.ndarray,
    cand_dir: Optional[np.ndarray],
    pc: np.ndarray,
    top_n: int = 5,
) -> None:
    """Print the top candidates by posterior probability.

    Args:
        cand_pos: ``(N, 3)`` candidate positions.
        cand_dir: ``(N, 3)`` candidate directions, or ``None``.
        pc: Posterior vector over candidates.
        top_n: Number of candidates to display.
    """
    order = np.argsort(-pc)[: min(top_n, len(pc))]
    print("Top candidates:")
    for k, i in enumerate(order, 1):
        d = None if cand_dir is None else cand_dir[int(i)].tolist()
        print(f"  {k:>2}. idx={int(i):<4} P={pc[int(i)]:.3f} pos={cand_pos[int(i)].tolist()} dir={d}")


def show_top_particles(
    p_pos: np.ndarray,
    p_dir: np.ndarray,
    pw: np.ndarray,
    top_n: int = 5,
) -> None:
    """Print the top particles by weight.

    Args:
        p_pos: ``(P, 3)`` particle positions.
        p_dir: ``(P, 3)`` particle directions.
        pw: Particle weight vector.
        top_n: Number of particles to display.
    """
    order = np.argsort(-pw)[: min(top_n, len(pw))]
    print("Top particles:")
    for k, i in enumerate(order, 1):
        print(f"  {k:>2}. idx={int(i):<4} w={pw[int(i)]:.3f} pos={p_pos[int(i)].tolist()} dir={p_dir[int(i)].tolist()}")
