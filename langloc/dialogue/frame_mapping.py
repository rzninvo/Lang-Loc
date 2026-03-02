"""Candidate-to-frame nearest-neighbour mapping and pool extraction.

Provides the KD-tree--backed ``NNIndex`` for spatial nearest-neighbour
queries, the ``CandToFrameMap`` dataclass that stores Gaussian-weighted
candidate→frame associations, and helper functions to build the mapping
and extract label/relation pools from a frame subset.

Key exports:
    NNIndex, CandToFrameMap, build_cand_to_frame_map,
    top_frames_by_mapping, label_pool_from_frames, rel_pool_from_frames.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Set, Tuple

import numpy as np

from langloc.dialogue.scene_data import FrameInfo


# ---------------------------------------------------------------------------
# Nearest-neighbour index (KD-tree with brute-force fallback)
# ---------------------------------------------------------------------------

class NNIndex:
    """Nearest-neighbour index over a point cloud.

    Uses ``scipy.spatial.cKDTree`` when available, falling back to a
    brute-force NumPy implementation otherwise.

    Attributes:
        X: Point cloud of shape ``(M, D)`` stored as float32.
    """

    def __init__(self, X: np.ndarray) -> None:
        self.X = X.astype(np.float32)
        self._kdtree = None
        try:
            from scipy.spatial import cKDTree  # type: ignore
            self._kdtree = cKDTree(self.X)
        except Exception:
            self._kdtree = None

    def query(self, q: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Find the *k* nearest neighbours for each query point.

        Args:
            q: Query points of shape ``(Q, D)``.
            k: Number of neighbours to return.

        Returns:
            Tuple of ``(indices, distances)`` with shapes ``(Q, k)``.
        """
        q = np.asarray(q, dtype=np.float32)
        k = min(int(k), self.X.shape[0])
        if self._kdtree is not None:
            d, idx = self._kdtree.query(q, k=k)
            return np.asarray(idx, dtype=np.int64), np.asarray(d, dtype=np.float32)

        # brute-force fallback
        diff = self.X[None, :, :] - q[:, None, :]
        d2 = np.sum(diff * diff, axis=-1)
        idx = np.argsort(d2, axis=1)[:, :k]
        d = np.take_along_axis(np.sqrt(d2), idx, axis=1)
        return idx.astype(np.int64), d.astype(np.float32)


# ---------------------------------------------------------------------------
# Candidate → frame mapping
# ---------------------------------------------------------------------------

@dataclass
class CandToFrameMap:
    """Gaussian-weighted mapping from candidates to their nearest frames.

    Attributes:
        idx: Frame indices, shape ``(N, K)`` — the *K* nearest frame indices
            for each of the *N* candidates.
        w: Normalised weights, shape ``(N, K)`` — Gaussian weight of each
            candidate→frame association (rows sum to 1).
    """

    idx: np.ndarray  # (N, K)
    w: np.ndarray    # (N, K)


def build_cand_to_frame_map(
    cand_pos: np.ndarray,
    cand_dir: Optional[np.ndarray],
    frame_pos: np.ndarray,
    frame_dir: np.ndarray,
    k_nn: int,
    sigma: float,
    use_direction: bool,
    dir_temp: float = 0.25,
) -> CandToFrameMap:
    """Build a candidate→frame mapping using KNN + Gaussian weighting.

    For each candidate pose, finds the *k_nn* nearest frames by position,
    weights them with a Gaussian kernel (``exp(-d^2 / 2σ^2)``), and
    optionally multiplies in a direction-cosine weight.

    Args:
        cand_pos: Candidate positions, shape ``(N, 3)``.
        cand_dir: Candidate directions, shape ``(N, 3)`` or ``None``.
        frame_pos: Frame positions, shape ``(F, 3)``.
        frame_dir: Frame directions, shape ``(F, 3)``.
        k_nn: Number of nearest frames per candidate.
        sigma: Gaussian kernel bandwidth (metres).
        use_direction: Whether to include direction-cosine weighting.
        dir_temp: Temperature for the direction weight (lower = sharper).

    Returns:
        ``CandToFrameMap`` with row-normalised weights.
    """
    nn = NNIndex(frame_pos)
    idx, dist = nn.query(cand_pos, k=k_nn)

    sigma = float(sigma) if sigma and sigma > 0 else 0.6
    w = np.exp(-(dist ** 2) / (2.0 * sigma * sigma)).astype(np.float32)

    if use_direction and cand_dir is not None:
        cd = cand_dir.astype(np.float32)
        cd = cd / np.maximum(np.linalg.norm(cd, axis=1, keepdims=True), 1e-6)
        fd = frame_dir[idx]
        fd = fd / np.maximum(np.linalg.norm(fd, axis=2, keepdims=True), 1e-6)
        cos = np.sum(cd[:, None, :] * fd, axis=2)
        dir_w = np.exp((cos - 1.0) / float(dir_temp)).astype(np.float32)
        w *= dir_w

    w = w / np.maximum(np.sum(w, axis=1, keepdims=True), 1e-12)
    return CandToFrameMap(idx=idx, w=w)


def top_frames_by_mapping(c2f: CandToFrameMap, max_frames: int = 30) -> List[int]:
    """Return the most frequently referenced frame indices from a mapping.

    Args:
        c2f: Candidate-to-frame mapping.
        max_frames: Maximum number of frame indices to return.

    Returns:
        Frame indices sorted by descending reference count.
    """
    flat = c2f.idx.reshape(-1)
    uniq, cnt = np.unique(flat, return_counts=True)
    order = uniq[np.argsort(-cnt)]
    return [int(x) for x in order[: min(int(max_frames), len(order))]]


# ---------------------------------------------------------------------------
# Pool builders
# ---------------------------------------------------------------------------

def label_pool_from_frames(
    frames: List[FrameInfo],
    frame_indices: Sequence[int],
) -> List[str]:
    """Collect all unique visible labels from a subset of frames.

    Args:
        frames: Full frame list.
        frame_indices: Indices into *frames* to include.

    Returns:
        Sorted list of unique canonical label strings.
    """
    pool: Set[str] = set()
    for j in frame_indices:
        pool |= frames[int(j)].visible_labels
    return sorted(pool)


def rel_pool_from_frames(
    frames: List[FrameInfo],
    frame_indices: Sequence[int],
    max_rel: int = 600,
    min_salience: float = 0.0,
    unique_only: bool = False,
) -> List[Tuple[str, str, str]]:
    """Collect unique relation triples from a subset of frames.

    Args:
        frames: Full frame list.
        frame_indices: Indices into *frames* to include.
        max_rel: Maximum number of relation triples to return.
        min_salience: Minimum salience for inclusion.  Currently unused
            because ``FrameInfo.rel_triples`` carries no per-relation
            salience — accepted for forward-compatibility with the config.
        unique_only: If ``True``, deduplicate across frames.  Relations
            are stored as sets so uniqueness is already guaranteed;
            accepted for config compatibility.

    Returns:
        Sorted list of ``(subject, relation, object)`` tuples, truncated to
        *max_rel*.
    """
    pool: Set[Tuple[str, str, str]] = set()
    for j in frame_indices:
        pool |= frames[int(j)].rel_triples
    rels = sorted(pool)
    return rels[:max_rel]
