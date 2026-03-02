"""Candidate pose extraction from evaluation entries.

Parses the raw JSON entry structure and produces clean numpy arrays
of candidate positions, directions, and prior probabilities.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def extract_candidates(
    entry: Dict[str, Any],
    candidate_set: str,
    include_predicted_pose: bool,
    pred_prior: float,
) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    """Extract candidate positions, directions, and prior from an entry dict.

    Supports ``"grid"``, ``"fov"``, ``"both"``, or ``"auto"`` (falls back to
    grid if available, else fov) candidate sets.  Optionally appends the
    predicted pose as an extra hypothesis.

    Args:
        entry: Single scene entry from the candidates JSON.
        candidate_set: Which candidate set to use
            (``"auto"`` | ``"grid"`` | ``"fov"`` | ``"both"``).
        include_predicted_pose: If ``True``, append the predicted pose as an
            additional candidate.
        pred_prior: Prior probability weight assigned to the predicted pose
            candidate.

    Returns:
        Tuple of ``(positions, directions, prior)`` where:

        - *positions*: ``(N, 3)`` float32 array.
        - *directions*: ``(N, 3)`` float32 array or ``None`` if all directions
          are zero.
        - *prior*: ``(N,)`` float64 normalised prior vector.

    Raises:
        ValueError: If no candidates are found or *candidate_set* is unknown.
    """
    cs = (candidate_set or "auto").lower().strip()
    grid = entry.get("grid_point_candidates") or []
    fov = entry.get("fov_pose_candidates") or []

    if cs == "auto":
        cs = "grid" if len(grid) else "fov"

    if cs == "grid":
        cands = list(grid)
        mode = "grid"
    elif cs == "fov":
        cands = list(fov)
        mode = "fov"
    elif cs == "both":
        cands = list(fov) + list(grid)
        mode = "both"
    else:
        raise ValueError(f"Unknown candidate_set: {candidate_set}")

    # add predicted pose as extra hypothesis
    if include_predicted_pose:
        pp = entry.get("predicted_pose", None)
        if isinstance(pp, dict) and "position" in pp:
            cands.append(
                {
                    "position": pp.get("position"),
                    "direction": pp.get("direction", None),
                    "prob": float(pred_prior),
                    "visible_count": float(pred_prior),
                    "_is_predicted_pose": True,
                }
            )

    if not cands:
        raise ValueError(f"No candidates found (grid={len(grid)}, fov={len(fov)}).")

    pos_list: List[List[float]] = []
    dir_list: List[List[float]] = []
    prior_list: List[float] = []

    for c in cands:
        if not isinstance(c, dict) or "position" not in c:
            continue
        pos_list.append(list(c["position"]))

        d = c.get("direction", None)
        if d is None:
            dir_list.append([0.0, 0.0, 0.0])
        else:
            dir_list.append(list(d))

        if c.get("_is_predicted_pose", False):
            prior_list.append(float(pred_prior))
        else:
            if mode == "grid":
                prior_list.append(float(c.get("prob", 1.0)))
            elif mode == "fov":
                prior_list.append(float(c.get("visible_count", c.get("prob", 1.0))))
            else:
                prior_list.append(float(c.get("visible_count", c.get("prob", 1.0))))

    cand_pos = np.asarray(pos_list, dtype=np.float32)
    cand_dir = np.asarray(dir_list, dtype=np.float32)

    if float(np.linalg.norm(cand_dir, axis=1).max()) < 1e-6:
        cand_dir_out = None
    else:
        cand_dir_out = cand_dir / np.maximum(np.linalg.norm(cand_dir, axis=1, keepdims=True), 1e-6)

    prior = np.asarray(prior_list, dtype=np.float64)
    prior = np.maximum(prior, 0.0)
    prior = prior / max(float(prior.sum()), 1e-12)
    return cand_pos, cand_dir_out, prior
