"""Question representation, pool building, and IDF weighting.

Defines the ``Question`` dataclass, builds label and relation pools from
a set of frames, and computes inverse-document-frequency weights to
down-weight common labels during question selection.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import pose_level_dialogue_semantic_fallback as dsf

from src.dialogue.math_utils import call_with_supported_kwargs
from src.dialogue.semantics import _rel_to_tuple, frame_label_salience


@dataclass(frozen=True)
class Question:
    """A candidate question to ask during dialogue.

    Attributes:
        qtype: Question type — ``"label"`` or ``"rel"``.
        idx: Index into the corresponding pool (label pool or relation pool).
    """

    qtype: str  # "label" or "rel"
    idx: int    # index into pool


HELP_TEXT = """Commands:
  y  yes
  n  no
  u  unknown/skip
  q  quit this entry
  tf show top frames posterior (this backend)
  tc show top candidates (A1 only)
  tp show top particles (A2 only)
  o  show pool sizes
  h  help
"""


def build_pools(
    frames_all: Sequence[Any],
    frame_subset: Sequence[int],
    max_rel_pool: int,
    rel_min_salience: float,
    rel_unique_only: bool,
    allowed_rels: Sequence[str],
) -> Tuple[List[str], List[Any]]:
    """Build label and relation pools from a subset of frames.

    Labels are extracted via ``dsf.label_pool_from_frames`` (with a local
    fallback).  Relations are extracted via ``dsf.rel_pool_from_frames``
    and optionally filtered by an allow-list of relation types.

    Args:
        frames_all: All frames in the scene.
        frame_subset: Indices into *frames_all* for the pooled frames.
        max_rel_pool: Maximum number of relations to include.
        rel_min_salience: Minimum salience for relation inclusion.
        rel_unique_only: If ``True``, keep only unique relation triples.
        allowed_rels: If non-empty, only keep relations whose predicate is
            in this set.

    Returns:
        Tuple of ``(label_pool, rel_pool)`` — lists of label strings and
        raw relation objects respectively.
    """
    frames_sub = [frames_all[i] for i in frame_subset]

    # labels
    try:
        label_pool = list(dsf.label_pool_from_frames(frames_all, frame_subset))
        label_pool = [str(x).strip().lower() for x in label_pool]
    except Exception:
        s: set = set()
        for fr in frames_sub:
            s |= set(frame_label_salience(fr).keys())
        label_pool = sorted(list(s))

    # relations
    rel_pool = call_with_supported_kwargs(
        getattr(dsf, "rel_pool_from_frames"),
        frames_all,
        frame_subset,
        max_rel=max_rel_pool,
        min_salience=rel_min_salience,
        unique_only=rel_unique_only,
    )
    rel_pool = list(rel_pool)

    if allowed_rels:
        allow = set(map(lambda x: str(x).strip().lower(), allowed_rels))
        filtered = []
        for t in rel_pool:
            tup = _rel_to_tuple(t)
            if tup and tup[1] in allow:
                filtered.append(t)
        rel_pool = filtered

    return label_pool, rel_pool


def compute_label_idf(
    label_pool: List[str],
    frame_label_dicts: List[Dict[str, float]],
) -> Dict[str, float]:
    """Compute inverse-document-frequency weights for each label.

    Uses the formula ``IDF(l) = log((F + 1) / (df(l) + 1))`` where *F* is
    the number of frames and *df(l)* is the number of frames containing
    label *l*.

    Args:
        label_pool: List of label strings to score.
        frame_label_dicts: Per-frame ``label → salience`` dictionaries.

    Returns:
        Dictionary mapping each label in *label_pool* to its IDF score.
    """
    F = len(frame_label_dicts)
    df: Dict[str, int] = {lab: 0 for lab in label_pool}
    for d in frame_label_dicts:
        for lab in d.keys():
            if lab in df:
                df[lab] += 1
    # idf = log((F+1)/(df+1))
    out: Dict[str, float] = {}
    for lab in label_pool:
        out[lab] = float(math.log((F + 1.0) / (df.get(lab, 0) + 1.0)))
    return out
