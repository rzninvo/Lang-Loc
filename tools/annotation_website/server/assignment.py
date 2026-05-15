"""Frame assignment policy.

Picks the next ``(scene_id, frame_id)`` the annotator should describe
within their chosen ``dataset``.

Constraints:
  * filter to keyframes that belong to the requested dataset,
  * never re-assign a frame this annotator already completed,
  * never give a frame currently leased by someone else,
  * never give a frame past the redundancy target unless every
    eligible frame has hit it,
  * **concentrate annotators on a single scene until it's covered**
    (closure-first; prevents fragmented per-scene coverage),
  * **prefer the easiest fresh scene** (lowest GPT-pipeline median Pos
    error) when opening a new one, so we close out the high-signal
    scenes first.
"""
from __future__ import annotations

import random
from collections import defaultdict
from typing import List, Optional, Tuple

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from .config import get_settings
from .lease import acquire, held_by
from .models import FrameCompletion, Keyframe, Lease, Scene


_MAX_RETRIES = 5

# Row tuple layout returned by _candidate_frames:
#   (scene_id, frame_id, dataset, difficulty_rank, frame_completions, my_scene_count, scene_total_count)
_IDX_SCENE = 0
_IDX_FRAME = 1
_IDX_DATASET = 2
_IDX_RANK = 3
_IDX_FRAME_COMP = 4
_IDX_MY_SCENE = 5
_IDX_SCENE_TOTAL = 6


def _candidate_frames(
    session: Session,
    annotator_id: str,
    dataset: str,
    allow_over_target: bool,
) -> List[Tuple[str, str, str, int, int, int, int]]:
    s = get_settings()

    seen_subq = (
        select(FrameCompletion.scene_id, FrameCompletion.frame_id)
        .where(FrameCompletion.annotator_id == annotator_id)
        .subquery()
    )
    leased_subq = (
        select(Lease.scene_id, Lease.frame_id)
        .where(Lease.annotator_id != annotator_id)
        .subquery()
    )
    frame_comp_subq = (
        select(
            FrameCompletion.scene_id,
            FrameCompletion.frame_id,
            func.count().label("n"),
        )
        .group_by(FrameCompletion.scene_id, FrameCompletion.frame_id)
        .subquery()
    )
    my_scene_subq = (
        select(
            FrameCompletion.scene_id,
            func.count().label("n"),
        )
        .where(FrameCompletion.annotator_id == annotator_id)
        .group_by(FrameCompletion.scene_id)
        .subquery()
    )
    scene_total_subq = (
        select(
            FrameCompletion.scene_id,
            func.count().label("n"),
        )
        .group_by(FrameCompletion.scene_id)
        .subquery()
    )

    stmt = (
        select(
            Keyframe.scene_id,
            Keyframe.frame_id,
            Scene.dataset,
            Scene.difficulty_rank,
            func.coalesce(frame_comp_subq.c.n, 0).label("frame_completions"),
            func.coalesce(my_scene_subq.c.n, 0).label("my_scene_count"),
            func.coalesce(scene_total_subq.c.n, 0).label("scene_total"),
        )
        .join(Scene, Scene.id == Keyframe.scene_id)
        .outerjoin(
            seen_subq,
            (seen_subq.c.scene_id == Keyframe.scene_id)
            & (seen_subq.c.frame_id == Keyframe.frame_id),
        )
        .outerjoin(
            leased_subq,
            (leased_subq.c.scene_id == Keyframe.scene_id)
            & (leased_subq.c.frame_id == Keyframe.frame_id),
        )
        .outerjoin(
            frame_comp_subq,
            (frame_comp_subq.c.scene_id == Keyframe.scene_id)
            & (frame_comp_subq.c.frame_id == Keyframe.frame_id),
        )
        .outerjoin(
            my_scene_subq,
            my_scene_subq.c.scene_id == Keyframe.scene_id,
        )
        .outerjoin(
            scene_total_subq,
            scene_total_subq.c.scene_id == Keyframe.scene_id,
        )
        .where(Scene.dataset == dataset)
        .where(seen_subq.c.scene_id.is_(None))
        .where(leased_subq.c.scene_id.is_(None))
    )
    if not allow_over_target:
        stmt = stmt.where(func.coalesce(frame_comp_subq.c.n, 0) < s.redundancy_target)

    rows = session.execute(stmt).all()
    return [
        (r[0], r[1], r[2], int(r[3]), int(r[4]), int(r[5]), int(r[6])) for r in rows
    ]


def _pick_within_scene(
    rows: List[Tuple], rng: random.Random
) -> Tuple[str, str]:
    """When all candidates belong to one scene (or one bucket of scenes
    of equal priority), pick a frame inside the scene that has the
    fewest existing completions, weighted-random tie-break."""
    weights = [1.0 / (r[_IDX_FRAME_COMP] + 1) for r in rows]
    pick = rng.choices(rows, weights=weights, k=1)[0]
    return pick[_IDX_SCENE], pick[_IDX_FRAME]


def _easiest_scene_first(
    rows: List[Tuple], rng: random.Random
) -> Tuple[str, str]:
    """Pick a frame from the scene with the lowest difficulty_rank.
    Ties broken by random scene; within the chosen scene, weighted
    random by 1 / (frame_completions + 1)."""
    scenes: dict[str, List[Tuple]] = defaultdict(list)
    rank_of: dict[str, int] = {}
    for r in rows:
        scenes[r[_IDX_SCENE]].append(r)
        rank_of[r[_IDX_SCENE]] = r[_IDX_RANK]

    # lowest difficulty_rank wins; random tie-break across scenes with the
    # same rank prevents simultaneous-arrival clumping
    sorted_scene_ids = sorted(scenes.keys(), key=lambda sid: (rank_of[sid], rng.random()))
    target_scene = sorted_scene_ids[0]
    return _pick_within_scene(scenes[target_scene], rng)


def _pick_phase(
    rows: List[Tuple], rng: random.Random
) -> Tuple[str, str]:
    """Three-phase prioritisation:
       Phase 1: scenes the annotator has already started (continue them).
       Phase 2: scenes any annotator has started but not finished
                (close them out before opening fresh ones).
       Phase 3: easiest fresh scene by difficulty_rank.
    """
    if not rows:
        raise ValueError("empty candidate list")

    in_progress = [r for r in rows if r[_IDX_MY_SCENE] > 0]
    if in_progress:
        # finishing my own scene: pick a frame inside it
        scenes: dict[str, List[Tuple]] = defaultdict(list)
        for r in in_progress:
            scenes[r[_IDX_SCENE]].append(r)
        # if I have multiple in-progress scenes, finish the most-progressed first
        target_scene = max(
            scenes.keys(),
            key=lambda sid: (scenes[sid][0][_IDX_MY_SCENE], rng.random()),
        )
        return _pick_within_scene(scenes[target_scene], rng)

    partial = [r for r in rows if r[_IDX_SCENE_TOTAL] > 0]
    if partial:
        # close someone else's partial scene; if multiple exist, the one
        # with the most existing completions (closer to fully covered)
        scenes = defaultdict(list)
        for r in partial:
            scenes[r[_IDX_SCENE]].append(r)
        target_scene = max(
            scenes.keys(),
            key=lambda sid: (scenes[sid][0][_IDX_SCENE_TOTAL], rng.random()),
        )
        return _pick_within_scene(scenes[target_scene], rng)

    return _easiest_scene_first(rows, rng)


def assign_next(
    session: Session,
    annotator_id: str,
    dataset: str,
    rng: Optional[random.Random] = None,
) -> Optional[Tuple[str, str]]:
    """Pick and lock the next (scene_id, frame_id) for this annotator
    within ``dataset``. Returns ``None`` if there is nothing left."""
    if rng is None:
        rng = random.Random()

    held = held_by(session, annotator_id)
    if held is not None:
        # only resume the lease if it belongs to the requested dataset
        kf_scene = session.get(Scene, held.scene_id)
        if kf_scene is not None and kf_scene.dataset == dataset:
            return held.scene_id, held.frame_id
        # different dataset → release the stale lease and reassign
        from .lease import release as _release
        _release(session, held.scene_id, held.frame_id, annotator_id)

    for _ in range(_MAX_RETRIES):
        rows = _candidate_frames(session, annotator_id, dataset, allow_over_target=False)
        if not rows:
            rows = _candidate_frames(session, annotator_id, dataset, allow_over_target=True)
            if not rows:
                return None
        scene_id, frame_id = _pick_phase(rows, rng)
        if acquire(session, scene_id, frame_id, annotator_id):
            return scene_id, frame_id
    return None
