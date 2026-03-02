"""Math utilities for the dialogue system.

Pure math helpers: vector operations, Bayesian updates, pose error
computation, and robust aggregation.  These functions have no
domain-specific dependencies and are shared across backends, question
selection, and evaluation.
"""

from __future__ import annotations

import inspect
import math
import statistics
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Signature-safe call
# ---------------------------------------------------------------------------
def call_with_supported_kwargs(fn: Callable, *args: Any, **kwargs: Any) -> Any:
    """Call *fn* forwarding only keyword arguments it actually accepts.

    If *fn* accepts ``**kwargs``, all keyword arguments are passed through
    unchanged.

    Args:
        fn: The callable to invoke.
        *args: Positional arguments forwarded to *fn*.
        **kwargs: Keyword arguments; those not in *fn*'s signature are
            silently dropped.

    Returns:
        The return value of *fn*.
    """
    sig = inspect.signature(fn)
    if any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values()):
        return fn(*args, **kwargs)
    filtered = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return fn(*args, **filtered)


# ---------------------------------------------------------------------------
# Candidate-to-frame mapping
# ---------------------------------------------------------------------------
def c2f_to_dense(
    c2f_map: Any,
    num_cands: int,
    num_frames: int,
    eps: float = 1e-12,
) -> np.ndarray:
    """Convert a candidate-to-frame map to a dense row-normalised matrix.

    Supports plain ``np.ndarray``, objects exposing a ``matrix``/``dense``/
    ``to_dense`` attribute or method, and sparse representations with
    ``idx``/``weights`` arrays.

    Args:
        c2f_map: Candidate-to-frame mapping in any supported representation.
        num_cands: Number of candidates (rows).
        num_frames: Number of frames (columns).
        eps: Small constant to prevent division by zero.

    Returns:
        Row-normalised ``(num_cands, num_frames)`` dense matrix.

    Raises:
        TypeError: If *c2f_map* cannot be converted to a dense matrix.
    """
    if isinstance(c2f_map, np.ndarray):
        M = np.asarray(c2f_map, dtype=np.float64)
    else:
        M = None

        # matrix-like
        for name in ("matrix", "mat", "M", "dense", "c2f", "weights_matrix", "to_dense", "as_matrix", "to_matrix", "toarray"):
            if hasattr(c2f_map, name):
                obj = getattr(c2f_map, name)
                try:
                    M = obj() if callable(obj) else obj
                    M = np.asarray(M, dtype=np.float64)
                    break
                except Exception:
                    M = None

        # sparse-like (idx + weights)
        if M is None:
            idx = None
            wts = None
            for iname in ("idx", "indices", "nn_idx", "knn_idx", "frame_idx", "frame_indices"):
                if hasattr(c2f_map, iname):
                    idx = getattr(c2f_map, iname)
                    break
            for wname in ("w", "weights", "nn_w", "knn_w", "vals", "values", "p"):
                if hasattr(c2f_map, wname):
                    wts = getattr(c2f_map, wname)
                    break

            if idx is not None and wts is not None:
                idx = np.asarray(idx, dtype=np.int64)
                wts = np.asarray(wts, dtype=np.float64)
                if idx.ndim == 1:
                    idx = idx[:, None]
                if wts.ndim == 1:
                    wts = wts[:, None]
                M = np.zeros((num_cands, num_frames), dtype=np.float64)
                n = min(num_cands, idx.shape[0])
                K = idx.shape[1]
                for i in range(n):
                    for k in range(K):
                        j = int(idx[i, k])
                        if 0 <= j < num_frames:
                            M[i, j] += float(wts[i, k])

    if M is None:
        raise TypeError(f"Could not convert CandToFrameMap to dense matrix: type={type(c2f_map)}")

    if M.shape == (num_frames, num_cands):
        M = M.T

    M = M[:num_cands, :num_frames]
    rs = M.sum(axis=1, keepdims=True)
    M = M / np.maximum(rs, eps)
    return M


# ---------------------------------------------------------------------------
# Vector / pose helpers
# ---------------------------------------------------------------------------
def _normalize(v: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Return the unit-length version of 3-D vector *v*.

    Args:
        v: Input vector (any shape broadcastable to ``(3,)``).
        eps: Norm threshold below which a zero vector is returned.

    Returns:
        Unit vector with shape ``(3,)``, or zeros if the norm is below *eps*.
    """
    v = np.asarray(v, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(v))
    return v / n if n > eps else np.zeros(3, dtype=np.float64)


def angle_deg(u: np.ndarray, v: np.ndarray) -> float:
    """Return the angle in degrees between vectors *u* and *v*.

    Args:
        u: First 3-D vector.
        v: Second 3-D vector.

    Returns:
        Angle in degrees, or ``nan`` if either vector has near-zero norm.
    """
    u = _normalize(u)
    v = _normalize(v)
    if float(np.linalg.norm(u)) < 1e-9 or float(np.linalg.norm(v)) < 1e-9:
        return float("nan")
    c = float(np.clip(np.dot(u, v), -1.0, 1.0))
    return float(math.degrees(math.acos(c)))


def pose_errors(
    pred_pos: Any,
    pred_dir: Any,
    gt_pos: Any,
    gt_dir: Any,
) -> Tuple[float, float]:
    """Compute position and rotation errors between predicted and ground-truth poses.

    Args:
        pred_pos: Predicted position (array-like, length 3).
        pred_dir: Predicted direction (array-like, length 3, or ``None``).
        gt_pos: Ground-truth position (array-like, length 3).
        gt_dir: Ground-truth direction (array-like, length 3, or ``None``).

    Returns:
        Tuple of ``(position_error_metres, rotation_error_degrees)``.
        Rotation error is ``nan`` if either direction is ``None``.
    """
    pred_pos = np.asarray(pred_pos, dtype=np.float64).reshape(3)
    gt_pos = np.asarray(gt_pos, dtype=np.float64).reshape(3)
    pos_err = float(np.linalg.norm(pred_pos - gt_pos))
    if pred_dir is None or gt_dir is None:
        rot_err = float("nan")
    else:
        rot_err = angle_deg(pred_dir, gt_dir)
    return pos_err, rot_err


# ---------------------------------------------------------------------------
# Probability / information theory
# ---------------------------------------------------------------------------
def entropy(p: np.ndarray, eps: float = 1e-12) -> float:
    """Shannon entropy (in nats) of a discrete probability vector.

    Args:
        p: Probability vector (need not sum to 1; will be normalised).
        eps: Floor applied before taking the log.

    Returns:
        Scalar entropy value.
    """
    p = np.asarray(p, dtype=np.float64)
    p = np.clip(p, eps, 1.0)
    p = p / float(p.sum())
    return float(-np.sum(p * np.log(p)))


def bayes_update(p: np.ndarray, like: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Bayesian update: ``posterior ∝ prior * likelihood``.

    Args:
        p: Prior distribution vector.
        like: Likelihood vector (same length as *p*).
        eps: Floor applied to the likelihood and to the normalisation constant.

    Returns:
        Normalised posterior distribution vector.
    """
    post = p * np.clip(like, eps, None)
    s = float(post.sum())
    if s <= 0:
        return np.ones_like(p) / len(p)
    return post / s


# ---------------------------------------------------------------------------
# Pose getters
# ---------------------------------------------------------------------------
def get_pose(
    entry: Dict[str, Any],
    key: str,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Dict[str, Any]]:
    """Extract position and direction from an entry dict.

    Args:
        entry: Dictionary containing pose information under *key*.
        key: Top-level key (e.g. ``"gt_pose"`` or ``"predicted_pose"``).

    Returns:
        Tuple of ``(position, direction, raw_dict)`` where *position* and
        *direction* are ``(3,)`` arrays or ``None``.
    """
    d = entry.get(key, None) or {}
    pos = d.get("position", None)
    direc = d.get("direction", None)
    if pos is None:
        return None, None, d
    pos = np.asarray(pos, dtype=np.float64).reshape(3)
    if direc is None:
        return pos, None, d
    direc = np.asarray(direc, dtype=np.float64).reshape(3)
    return pos, direc, d


# ---------------------------------------------------------------------------
# Salience → probability
# ---------------------------------------------------------------------------
def _to_prob01(x: float) -> float:
    """Clamp a value to the ``[0, 1]`` range.

    If *x* > 1 it is assumed to be a percentage and divided by 100.

    Args:
        x: Raw numeric value.

    Returns:
        Value in ``[0.0, 1.0]``.
    """
    x = float(x)
    if x > 1.0:
        # likely percentage
        x = x / 100.0
    return float(np.clip(x, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------
def safe_mean(xs: List[float]) -> float:
    """Mean of finite values, returning ``nan`` if the list is empty.

    Args:
        xs: List of float values (``None`` and non-finite entries are skipped).

    Returns:
        Mean of the finite values.
    """
    xs = [x for x in xs if x is not None and np.isfinite(x)]
    return float(statistics.fmean(xs)) if xs else float("nan")


def safe_median(xs: List[float]) -> float:
    """Median of finite values, returning ``nan`` if the list is empty.

    Args:
        xs: List of float values (``None`` and non-finite entries are skipped).

    Returns:
        Median of the finite values.
    """
    xs = [x for x in xs if x is not None and np.isfinite(x)]
    return float(statistics.median(xs)) if xs else float("nan")
