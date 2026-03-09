"""Evaluation metrics: SceneMetrics dataclass, Hit@r, mass-radius, IoU.

Provides a unified metrics container and two computation variants:

- **Standard mode**: multi-radius Hit@r curve, mass-radius percentiles,
  Top-K minimum distance, angular error, and view IoU.
- **Simple mode** (coarse-to-fine): ground-truth probability, NLL,
  single-radius Hit@r, and distance error.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import open3d as o3d


# ---------------------------------------------------------------------------
#  Unified metrics container
# ---------------------------------------------------------------------------

@dataclass
class SceneMetrics:
    """Per-scene localization metrics.

    All fields not applicable to the current evaluation mode default to
    ``None`` or appropriate zero values.

    Attributes:
        scene_id: Scene identifier string.
        frame_id: Frame identifier string.
        distance_error: Euclidean distance (m) between predicted and
            ground-truth camera positions.
        grid_points: Number of grid camera positions sampled.
        matched_objects: Number of matched object IDs used.
        hit_masses: (Standard) Dict mapping radius *r* to the probability
            mass within *r* metres of ground truth.
        mass_radii: (Standard) Dict mapping percentile to the radius (m)
            enclosing that fraction of probability mass.
        topk_min_dist: (Standard) Minimum distance (m) among the top-K
            probability grid points to ground truth.
        angular_error_deg: (Standard) Angular error in degrees between
            predicted and ground-truth viewing directions.
        iou_error: (Standard) ``1 - IoU`` between predicted and
            ground-truth view frustum triangle sets.
        gt_prob: (Simple) Probability at the ground-truth grid cell.
        nll: (Simple) Negative log-likelihood at ground truth.
        hit_mass: (Simple) Single-radius Hit@r mass.
    """
    scene_id: str = ""
    frame_id: str = ""
    distance_error: float = 0.0
    grid_points: int = 0
    matched_objects: int = 0

    # Standard-mode fields
    hit_masses: Optional[Dict[float, float]] = None
    mass_radii: Optional[Dict[float, float]] = None
    topk_min_dist: Optional[float] = None
    angular_error_deg: Optional[float] = None
    iou_error: Optional[float] = None

    # Simple-mode fields
    gt_prob: Optional[float] = None
    nll: Optional[float] = None
    hit_mass: Optional[float] = None


# ---------------------------------------------------------------------------
#  Standard-mode metrics
# ---------------------------------------------------------------------------

def compute_metrics_standard(cams: np.ndarray,
                             probs: np.ndarray,
                             gt_cam: np.ndarray,
                             hit_radii: List[float],
                             mass_percentiles: List[float],
                             topk_k: int) -> Tuple[int, SceneMetrics]:
    """Compute standard localization metrics with multi-radius Hit@r and mass-radius.

    Args:
        cams: Camera positions, shape ``(N, 3)``.
        probs: Normalised probability per camera, shape ``(N,)``.
        gt_cam: Ground-truth camera position, shape ``(3,)``.
        hit_radii: List of radii (m) for Hit@r computation.
        mass_percentiles: List of percentiles for mass-radius (e.g. 50, 90).
        topk_k: Number of top-probability points for Top-K min distance.

    Returns:
        A 2-tuple ``(pred_idx, metrics)`` where *pred_idx* is the argmax
        probability index and *metrics* is the populated
        :class:`SceneMetrics`.
    """
    pred_idx = int(np.argmax(probs))
    distances = np.linalg.norm(cams - gt_cam[None, :], axis=1)

    hit_masses: Dict[float, float] = {}
    for r in sorted(set(float(r) for r in hit_radii)):
        hit_masses[r] = float(probs[distances <= r].sum())

    mass_radii: Dict[float, float] = {}
    order = np.argsort(distances)
    cum = np.cumsum(probs[order])
    for p in sorted(set(float(p) for p in mass_percentiles)):
        target = max(0.0, min(p / 100.0, 1.0))
        idx = int(np.searchsorted(cum, target, side="left"))
        if idx >= len(order):
            mass_radii[p] = float(distances[order[-1]])
        else:
            mass_radii[p] = float(distances[order[idx]])

    topk_k = max(1, int(topk_k))
    k = min(topk_k, len(probs))
    top_idx = np.argpartition(probs, -k)[-k:]
    topk_min_dist = float(distances[top_idx].min()) if len(top_idx) else float("nan")

    dist_err = float(np.linalg.norm(cams[pred_idx] - gt_cam))

    return pred_idx, SceneMetrics(
        scene_id="",
        frame_id="",
        hit_masses=hit_masses,
        mass_radii=mass_radii,
        topk_min_dist=topk_min_dist,
        distance_error=dist_err,
        grid_points=len(cams),
    )


def build_metrics_table_standard(metrics_list: List[SceneMetrics],
                                 hit_radii: List[float],
                                 mass_percentiles: List[float],
                                 topk_k: int) -> str:
    """Format a multi-column metrics table for standard-mode evaluation.

    Args:
        metrics_list: Per-scene metrics from standard evaluation.
        hit_radii: Radii used in the Hit@r columns.
        mass_percentiles: Percentiles used in the mass-radius columns.
        topk_k: K value for the Top-K min distance column.

    Returns:
        A formatted table string, or ``""`` if *metrics_list* is empty.
    """
    hit_radii = sorted(set(float(r) for r in hit_radii))
    mass_percentiles = sorted(set(float(p) for p in mass_percentiles))
    headers = [
        "Scene",
        "Frame",
        *[f"Hit@{r:.2f}m" for r in hit_radii],
        *[f"R{p:.0f}%" for p in mass_percentiles],
        f"TopK{topk_k} (m)",
        "Err (m)",
        "Ang err (deg)",
        "IoU err",
        "Matches",
        "Grid pts",
    ]
    rows: List[List[str]] = []
    for m in metrics_list:
        hit_vals = [m.hit_masses.get(r, 0.0) for r in hit_radii] if m.hit_masses else [0.0] * len(hit_radii)
        rad_vals = [m.mass_radii.get(p, float("nan")) for p in mass_percentiles] if m.mass_radii else [float("nan")] * len(mass_percentiles)
        ang_err = "-" if m.angular_error_deg is None else f"{m.angular_error_deg:.2f}"
        rows.append([
            m.scene_id,
            m.frame_id,
            *[f"{v:.3f}" for v in hit_vals],
            *[f"{v:.3f}" if np.isfinite(v) else "-" for v in rad_vals],
            f"{m.topk_min_dist:.3f}" if m.topk_min_dist is not None else "-",
            f"{m.distance_error:.3f}",
            ang_err,
            "-" if m.iou_error is None else f"{m.iou_error:.3f}",
            str(m.matched_objects),
            str(m.grid_points),
        ])

    if not rows:
        return ""

    col_widths = [len(h) for h in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            col_widths[idx] = max(col_widths[idx], len(cell))

    def fmt_row(cells: List[str]) -> str:
        """Format a single row with left-justified, padded cells."""
        return " | ".join(cell.ljust(col_widths[i]) for i, cell in enumerate(cells))

    separator = "-+-".join("-" * w for w in col_widths)
    lines = [fmt_row(headers), separator]
    for row in rows:
        lines.append(fmt_row(row))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
#  Simple-mode metrics (coarse-to-fine)
# ---------------------------------------------------------------------------

def compute_metrics_simple(cams: np.ndarray,
                           probs: np.ndarray,
                           gt_cam: np.ndarray,
                           eps: float,
                           hit_radius: float) -> Tuple[int, SceneMetrics]:
    """Compute simple localization metrics: GT probability, NLL, single Hit@r.

    Args:
        cams: Camera positions, shape ``(N, 3)``.
        probs: Normalised probability per camera, shape ``(N,)``.
        gt_cam: Ground-truth camera position, shape ``(3,)``.
        eps: Floor for log-probability to avoid ``-inf``.
        hit_radius: Radius (m) for the single Hit@r metric.

    Returns:
        A 2-tuple ``(pred_idx, metrics)`` where *pred_idx* is the argmax
        probability index and *metrics* is the populated
        :class:`SceneMetrics`.
    """
    pred_idx = int(np.argmax(probs))
    pred_cam = cams[pred_idx]

    distances = np.linalg.norm(cams - gt_cam[None, :], axis=1)
    gt_idx = int(np.argmin(distances))
    gt_prob = float(probs[gt_idx])
    gt_prob_clamped = max(gt_prob, eps)
    nll = float(-math.log(gt_prob_clamped))

    hit_mass = float(probs[distances <= hit_radius].sum())
    dist_err = float(np.linalg.norm(pred_cam - gt_cam))

    return pred_idx, SceneMetrics(
        scene_id="",
        frame_id="",
        gt_prob=gt_prob,
        nll=nll,
        hit_mass=hit_mass,
        distance_error=dist_err,
        grid_points=len(cams),
    )


def build_metrics_table_simple(metrics_list: List[SceneMetrics],
                               hit_radius: float) -> str:
    """Format a simple metrics table for coarse-to-fine evaluation.

    Args:
        metrics_list: Per-scene metrics from simple evaluation.
        hit_radius: Radius used for the Hit@r column.

    Returns:
        A formatted table string, or ``""`` if *metrics_list* is empty.
    """
    headers = [
        "Scene",
        "Frame",
        "GT Prob",
        "NLL",
        f"Hit@{hit_radius:.2f}m",
        "Err (m)",
        "Matches",
        "Grid pts",
    ]
    rows: List[List[str]] = []
    for m in metrics_list:
        rows.append([
            m.scene_id,
            m.frame_id,
            f"{m.gt_prob:.4f}" if m.gt_prob is not None else "-",
            f"{m.nll:.3f}" if m.nll is not None else "-",
            f"{m.hit_mass:.3f}" if m.hit_mass is not None else "-",
            f"{m.distance_error:.3f}",
            str(m.matched_objects),
            str(m.grid_points),
        ])

    if not rows:
        return ""

    col_widths = [len(h) for h in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            col_widths[idx] = max(col_widths[idx], len(cell))

    def fmt_row(cells: List[str]) -> str:
        """Format a single row with left-justified, padded cells."""
        return " | ".join(cell.ljust(col_widths[i]) for i, cell in enumerate(cells))

    separator = "-+-".join("-" * w for w in col_widths)
    lines = [fmt_row(headers), separator]
    for row in rows:
        lines.append(fmt_row(row))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
#  View IoU computation (standard mode only)
#  Core frustum visibility lives in langloc.eval.view_iou.
# ---------------------------------------------------------------------------

from langloc.eval.view_iou import (
    _visible_triangles_from_view as _vis_tris,
)


def compute_view_iou_error(gt_cam: np.ndarray,
                           gt_dir: Optional[np.ndarray],
                           pred_cam: np.ndarray,
                           pred_dir: Optional[np.ndarray],
                           hfov: float,
                           vfov: float,
                           rc: o3d.t.geometry.RaycastingScene,
                           geom_id: int,
                           tri_pts: np.ndarray,
                           tri_centroids: np.ndarray,
                           tri_areas: np.ndarray,
                           near: float = 0.05,
                           far: Optional[float] = None) -> Tuple[Optional[float], Optional[float], set[int], set[int]]:
    """Compute view IoU and IoU error between ground-truth and predicted cameras.

    The IoU is the area-weighted intersection-over-union of the triangle
    sets visible from each camera frustum.

    Args:
        gt_cam: Ground-truth camera position, shape ``(3,)``.
        gt_dir: Ground-truth viewing direction, shape ``(3,)`` or ``None``.
        pred_cam: Predicted camera position, shape ``(3,)``.
        pred_dir: Predicted viewing direction, shape ``(3,)`` or ``None``.
        hfov: Horizontal field-of-view in radians.
        vfov: Vertical field-of-view in radians.
        rc: Pre-built Open3D raycasting scene.
        geom_id: Geometry ID of the mesh within *rc*.
        tri_pts: Per-triangle vertex positions, shape ``(T, 3, 3)``.
        tri_centroids: Per-triangle centroids, shape ``(T, 3)``.
        tri_areas: Per-triangle areas, shape ``(T,)``.
        near: Near-plane distance in metres.
        far: Optional far-plane distance in metres.

    Returns:
        A 4-tuple ``(iou, iou_error, gt_set, pred_set)`` where *iou* and
        *iou_error* are ``None`` if insufficient data is available, and
        *gt_set* / *pred_set* are the triangle-index sets.
    """
    if gt_dir is None or pred_dir is None:
        return None, None, set(), set()
    gt_vis = _vis_tris(gt_cam, gt_dir, hfov, vfov,
                       rc, geom_id, tri_pts, tri_centroids,
                       near, far)
    pred_vis = _vis_tris(pred_cam, pred_dir, hfov, vfov,
                         rc, geom_id, tri_pts, tri_centroids,
                         near, far)
    if not gt_vis and not pred_vis:
        return None, None, gt_vis, pred_vis

    inter = gt_vis & pred_vis
    union = gt_vis | pred_vis
    if not union:
        return None, None, gt_vis, pred_vis

    inter_area = float(tri_areas[list(inter)].sum()) if inter else 0.0
    union_area = float(tri_areas[list(union)].sum())
    if union_area <= 1e-9:
        return None, None, gt_vis, pred_vis
    iou = inter_area / union_area
    return iou, 1.0 - iou, gt_vis, pred_vis
