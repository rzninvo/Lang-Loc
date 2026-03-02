"""Camera prediction strategies and candidate building for localization.

Provides methods for selecting a final predicted camera position from a
set of weighted candidates, including argmax, random sampling, and a
cluster-aware Gaussian-weighted strategy.  Also includes utilities for
building ranked candidate lists for export.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
#  Core prediction strategies
# ---------------------------------------------------------------------------

def _cluster_weighted_prediction(positions: np.ndarray,
                                 weights: np.ndarray,
                                 bandwidth: float,
                                 max_points: int) -> Tuple[np.ndarray, List[int], np.ndarray]:
    """Compute a cluster-aware weighted-average camera prediction.

    Selects the top candidates by weight, applies a Gaussian kernel to
    emphasise spatially clustered high-weight positions, and returns
    the density-weighted centroid.

    Args:
        positions: Candidate positions, shape ``(N, 2)`` or ``(N, 3)``.
        weights: Non-negative weight per candidate, shape ``(N,)``.
        bandwidth: Gaussian kernel bandwidth in metres.
        max_points: Maximum number of top candidates to consider.

    Returns:
        A 3-tuple ``(pred, indices, cluster_weights)`` where

        - **pred** is the predicted position array.
        - **indices** lists the candidate indices that contributed.
        - **cluster_weights** is the normalised weight array over those
          candidates.

    Raises:
        ValueError: If *positions* is empty.
    """
    if len(positions) == 0:
        raise ValueError("No candidate positions available for prediction.")

    weights = np.clip(np.asarray(weights, dtype=np.float64), 0.0, None)
    if not np.any(weights > 0):
        weights = np.ones_like(weights)

    bandwidth = max(float(bandwidth), 1e-6)
    max_points = max(1, int(max_points))

    idx_sorted = np.argsort(weights)
    if len(idx_sorted) > max_points:
        idx_sorted = idx_sorted[-max_points:]

    subset_positions = positions[idx_sorted]
    subset_weights = weights[idx_sorted]
    subset_weights /= subset_weights.sum()

    if len(subset_positions) == 1:
        return subset_positions[0], [int(idx_sorted[0])], np.asarray([1.0], dtype=np.float64)

    diff = subset_positions[:, None, :] - subset_positions[None, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    kernel = np.exp(-dist2 / (2.0 * bandwidth * bandwidth))
    density = kernel @ subset_weights
    cluster_weights = subset_weights * density
    total = cluster_weights.sum()
    if total <= 0:
        cluster_weights = subset_weights
        total = cluster_weights.sum()
    cluster_weights /= total
    pred = np.sum(cluster_weights[:, None] * subset_positions, axis=0)

    return pred, [int(idx) for idx in idx_sorted], cluster_weights


def select_prediction_point(positions: np.ndarray,
                            weights: np.ndarray,
                            strategy: str,
                            rng: np.random.Generator,
                            bandwidth: float,
                            max_points: int) -> Tuple[np.ndarray, List[int], np.ndarray]:
    """Select a predicted camera position according to the requested strategy.

    Args:
        positions: Candidate positions, shape ``(N, D)``.
        weights: Non-negative weight per candidate, shape ``(N,)``.
        strategy: One of ``"argmax"``, ``"random"``, or ``"weighted"``
            (cluster-aware).
        rng: NumPy random generator (used for ``"random"`` strategy).
        bandwidth: Gaussian kernel bandwidth for the ``"weighted"``
            strategy, in metres.
        max_points: Maximum candidates for the ``"weighted"`` strategy.

    Returns:
        A 3-tuple ``(pred, indices, weights)`` — see
        :func:`_cluster_weighted_prediction`.

    Raises:
        ValueError: If *positions* is empty.
    """
    if len(positions) == 0:
        raise ValueError("No candidate positions available for prediction.")

    weights = np.clip(np.asarray(weights, dtype=np.float64), 0.0, None)
    if not np.any(weights > 0):
        weights = np.ones_like(weights)
    total = weights.sum()

    if strategy == "argmax" or len(positions) == 1:
        idx = int(np.argmax(weights))
        return positions[idx], [idx], np.asarray([1.0], dtype=np.float64)

    if strategy == "random":
        probs = weights / total
        idx = int(rng.choice(len(positions), p=probs))
        return positions[idx], [idx], np.asarray([1.0], dtype=np.float64)

    # Default: weighted cluster-aware prediction.
    return _cluster_weighted_prediction(positions,
                                        weights,
                                        bandwidth=bandwidth,
                                        max_points=max_points)


def top_n_fov_poses(positions: np.ndarray,
                    weights: np.ndarray,
                    n: int,
                    rng: np.random.Generator,
                    directions: Optional[np.ndarray] = None) -> List[Dict[str, object]]:
    """Return up to *n* pose/direction pairs ranked by FOV-weighted probability.

    Selects the highest-weighted candidates (breaking ties randomly)
    and pairs each position with its direction vector if available.

    Args:
        positions: Candidate positions, shape ``(M, 2+)``.
        weights: Non-negative weight per candidate, shape ``(M,)``.
        n: Maximum number of poses to return.
        rng: NumPy random generator for tie-breaking.
        directions: Optional unit direction vectors, shape ``(M, 3)``.

    Returns:
        A list of dicts, each with keys ``"pose"`` (``[x, y]``) and
        ``"direction"`` (unit vector list or ``None``).

    Raises:
        ValueError: If *positions* and *weights* have mismatched lengths,
            or if *directions* is provided but has a different length.
    """
    if n <= 0:
        return []
    if len(positions) == 0 or len(weights) == 0:
        return []
    if len(positions) != len(weights):
        raise ValueError("Positions and weights must have the same length.")
    if directions is not None and len(directions) != len(positions):
        raise ValueError("Directions must align with positions.")

    weights = np.clip(np.asarray(weights, dtype=np.float64), 0.0, None)
    if not np.any(weights > 0):
        weights = np.ones_like(weights)

    max_w = float(weights.max())
    top_idx = np.where(weights == max_w)[0]

    if len(top_idx) > n:
        chosen = rng.choice(top_idx, size=n, replace=False)
    else:
        sorted_idx = np.argsort(-weights)
        chosen = sorted_idx[: min(n, len(sorted_idx))]

    results: List[Dict[str, object]] = []
    for i in chosen:
        pose_xy = [float(positions[i][0]), float(positions[i][1])]
        if directions is not None:
            dir_vec = np.asarray(directions[i], dtype=np.float64)
            norm = float(np.linalg.norm(dir_vec))
            dir_out = (dir_vec / norm).tolist() if norm > 1e-6 else None
        else:
            dir_out = None
        results.append({
            "pose": pose_xy,
            "direction": dir_out,
        })
    return results


# ---------------------------------------------------------------------------
#  Candidate building (used by candidates export mode)
# ---------------------------------------------------------------------------

def build_grid_candidates(cams: np.ndarray,
                          counts: np.ndarray,
                          probs: np.ndarray,
                          top_n: int) -> List[Dict[str, object]]:
    """Build a ranked list of top-N grid-point candidates.

    Candidates are ranked by descending visibility count.

    Args:
        cams: Camera positions, shape ``(N, 3)``.
        counts: Visibility counts per camera, shape ``(N,)``.
        probs: Normalised probabilities per camera, shape ``(N,)``.
        top_n: Number of candidates to return.

    Returns:
        A list of dicts with keys ``"position"``, ``"visible_count"``,
        and ``"prob"``.
    """
    if top_n <= 0 or len(cams) == 0:
        return []
    order = np.argsort(-counts)
    top_idx = order[: min(top_n, len(order))]
    results: List[Dict[str, object]] = []
    for idx in top_idx:
        pos = cams[int(idx)]
        results.append({
            "position": [float(pos[0]), float(pos[1]), float(pos[2])],
            "visible_count": int(counts[int(idx)]),
            "prob": float(probs[int(idx)]),
        })
    return results


def build_pose_candidates(positions: List[np.ndarray],
                          directions: List[np.ndarray],
                          weights: List[float],
                          top_n: int) -> List[Dict[str, object]]:
    """Build a ranked list of top-N FOV pose candidates.

    Candidates are ranked by descending FOV-weighted count.

    Args:
        positions: Arrow-field positions (list of ``(3,)`` arrays).
        directions: Arrow-field unit directions (list of ``(3,)`` arrays).
        weights: FOV-weighted counts per position.
        top_n: Number of candidates to return.

    Returns:
        A list of dicts with keys ``"position"``, ``"direction"``, and
        ``"visible_count"``.
    """
    if top_n <= 0 or not positions or not weights:
        return []
    weights_np = np.asarray(weights, dtype=np.float64)
    order = np.argsort(-weights_np)
    top_idx = order[: min(top_n, len(order))]
    results: List[Dict[str, object]] = []
    for idx in top_idx:
        pos = positions[int(idx)]
        dir_vec = directions[int(idx)] if directions else None
        dir_out = None
        if dir_vec is not None:
            norm = float(np.linalg.norm(dir_vec))
            if norm > 1e-6:
                dir_out = (dir_vec / norm).tolist()
        results.append({
            "position": [float(pos[0]), float(pos[1]), float(pos[2])],
            "direction": dir_out,
            "visible_count": int(weights_np[int(idx)]),
        })
    return results
