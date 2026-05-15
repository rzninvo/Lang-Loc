"""Human-as-localizer: pose-error math + 3-D View IoU + assignment.

The annotator reads a description and places a first-person camera at
their best guess of where it was written from. We compare against the
keyframe's stored ``scene_pose`` and compute:

  * ``distance_error`` — 3-D distance from predicted to GT camera
    centre (matches the paper Tab. 4 ``Pos.`` column).
  * ``angular_error_deg`` — 3-D angle between predicted forward (a
    horizontal unit vector from yaw) and GT forward (``R @ [0,0,1]``).
    Matches ``langloc.localization.evaluation``'s definition.
  * ``iou_error`` — 1 − 3-D View IoU at the per-dataset evaluation
    FoV, ported from Abu's
    ``LangLoc-human-localisation-tool/metrics.py`` which itself is
    adapted from ``eval_pose_iou_ang.py``. Open3D raycasts visible
    triangles from both GT and predicted camera centres; IoU =
    intersection area / union area. Cached per scene (mesh loading is
    expensive).

Assignment policy mirrors the description side: continue mine →
close someone's partial scene → fresh easiest scene by
``difficulty_rank``. We only assign frames that already have at least
one description.
"""
from __future__ import annotations

import json
import math
import threading
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from sqlalchemy import Integer, func, literal, select
from sqlalchemy.orm import Session

from .models import Description, Keyframe, Scene, HumanLocalization, LocalizationSkip


_HUMAN_EYE_HEIGHT_M = 1.6

# Per-dataset evaluation FoV (paper supp Tab. 7 / configs/localization/*.yaml).
# These are the values the LangLoc localizer uses when scoring; we use
# them here so the human IoU number is directly comparable to Tab. 4.
_DATASET_FOV_DEG = {
    "scannet": (58.30, 45.33),
    "3rscan":  (39.31, 64.76),
}

# Mesh filename per dataset (relative to dataset root / scene_id)
_DATASET_MESH = {
    "scannet": lambda sid: f"{sid}_vh_clean_2.ply",
    "3rscan":  lambda _:  "labels.instances.annotated.v2.ply",
}

# Lazy-built IoU context cache: scene_id -> (rc_scene, mesh_id,
# tri_points, tri_centroids, tri_areas) or None on failure.
_iou_cache: dict = {}
_iou_cache_lock = threading.Lock()


def compute_pose_errors(
    gt_pose: list,
    pred_x: float,
    pred_y: float,
    pred_z: float,
    pred_yaw: float,
) -> Tuple[float, float]:
    """Return (distance_error, angular_error_deg) for a 4x4 GT pose +
    a predicted (x, y, z, yaw). Same convention as
    ``langloc.localization.evaluation``: GT forward = ``R @ [0,0,1]``;
    predicted forward = ``(cos yaw, sin yaw, 0)``.
    """
    pose_arr = np.asarray(gt_pose, dtype=np.float64)
    gt_cam = pose_arr[:3, 3]
    rot = pose_arr[:3, :3]
    gt_fwd_raw = rot @ np.array([0.0, 0.0, 1.0])
    gt_fwd_norm = np.linalg.norm(gt_fwd_raw)
    gt_fwd = gt_fwd_raw / gt_fwd_norm if gt_fwd_norm > 1e-6 else None

    pred_cam = np.array([pred_x, pred_y, pred_z], dtype=np.float64)
    distance_error = float(np.linalg.norm(pred_cam - gt_cam))

    pred_fwd = np.array([math.cos(pred_yaw), math.sin(pred_yaw), 0.0])
    if gt_fwd is None:
        return distance_error, float("nan")
    cos_a = float(np.clip(float(np.dot(gt_fwd, pred_fwd)), -1.0, 1.0))
    angle_deg = math.degrees(math.acos(cos_a))
    return distance_error, angle_deg


# ---------------------------------------------------------------------------
# 3-D View IoU (ported from Abu/LangLoc-human-localisation-tool/metrics.py)
# ---------------------------------------------------------------------------
_NEAR = 0.05  # near plane (m) — must match the paper's eval


def _normalise(v: np.ndarray, eps: float = 1e-9) -> Optional[np.ndarray]:
    n = float(np.linalg.norm(v))
    return v / n if n >= eps else None


def _camera_axes(forward: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    fwd = _normalise(np.asarray(forward, dtype=np.float64))
    if fwd is None:
        return None
    up = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(fwd, up))) > 0.95:
        up = np.array([0.0, 1.0, 0.0])
    right = _normalise(np.cross(fwd, up))
    if right is None:
        return None
    up_ortho = _normalise(np.cross(right, fwd))
    return (fwd, right, up_ortho) if up_ortho is not None else None


def _build_iou_context(dataset: str, scene_id: str, dataset_root: Path):
    """Build the Open3D raycasting scene + triangle arrays for a mesh.
    Cached per ``scene_id``. Returns ``None`` on failure (mesh missing,
    Open3D not installed, etc.) — IoU is then skipped in that submit.
    """
    try:
        import open3d as o3d
    except ImportError:
        print("[WARN] open3d not installed → skipping View IoU", flush=True)
        return None
    mesh_name = _DATASET_MESH.get(dataset, lambda s: None)(scene_id)
    if mesh_name is None:
        print(f"[WARN] no mesh recipe for dataset={dataset!r} → skipping IoU", flush=True)
        return None
    ply = dataset_root / scene_id / mesh_name
    if not ply.is_file():
        print(f"[WARN] mesh not on disk: {ply} → skipping IoU", flush=True)
        return None
    mesh = o3d.io.read_triangle_mesh(str(ply))
    if mesh.is_empty():
        print(f"[WARN] empty mesh: {ply} → skipping IoU", flush=True)
        return None
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    tris = np.asarray(mesh.triangles, dtype=np.int64)
    tri_points = verts[tris]
    edge_a = tri_points[:, 1] - tri_points[:, 0]
    edge_b = tri_points[:, 2] - tri_points[:, 0]
    tri_areas = 0.5 * np.linalg.norm(np.cross(edge_a, edge_b), axis=1)
    tri_centroids = tri_points.mean(axis=1)
    rc_scene = o3d.t.geometry.RaycastingScene()
    mesh_id = int(rc_scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh)))
    return rc_scene, mesh_id, tri_points, tri_centroids, tri_areas


def _get_iou_context(dataset: str, scene_id: str, dataset_root: Path):
    key = f"{dataset}:{scene_id}"
    with _iou_cache_lock:
        if key in _iou_cache:
            return _iou_cache[key]
    ctx = _build_iou_context(dataset, scene_id, dataset_root)
    with _iou_cache_lock:
        _iou_cache[key] = ctx
    return ctx


def _visible_triangles(
    cam: np.ndarray, forward: np.ndarray, h_fov_rad: float, v_fov_rad: float,
    rc_scene, mesh_id: int, tri_points: np.ndarray, tri_centroids: np.ndarray,
) -> set:
    """Frustum-cull triangles by the camera FoV, then raycast from
    camera to each surviving triangle's centroid; keep only those whose
    primary hit is the same triangle (i.e. nothing occludes it)."""
    try:
        import open3d as o3d
    except ImportError:
        return set()
    axes = _camera_axes(forward)
    if axes is None:
        return set()
    fwd, right, up = axes
    cam = np.asarray(cam, dtype=np.float64)

    rel = tri_points - cam[None, None, :]
    fwd_c = rel @ fwd
    right_c = rel @ right
    up_c = rel @ up
    tan_h = math.tan(0.5 * h_fov_rad)
    tan_v = math.tan(0.5 * v_fov_rad)
    mask = (
        np.all(fwd_c > _NEAR, axis=1)
        & np.all(np.abs(right_c) <= fwd_c * tan_h, axis=1)
        & np.all(np.abs(up_c) <= fwd_c * tan_v, axis=1)
    )
    if not np.any(mask):
        return set()
    idxs = np.nonzero(mask)[0]
    vecs = tri_centroids[idxs] - cam
    dists = np.linalg.norm(vecs, axis=1)
    valid = dists > 1e-6
    idxs = idxs[valid]
    if len(idxs) == 0:
        return set()
    dirs = vecs[valid] / dists[valid, None]
    rays = np.concatenate(
        [np.repeat(cam[None], len(idxs), axis=0), dirs], axis=1
    ).astype(np.float32)
    cast = rc_scene.cast_rays(o3d.core.Tensor(rays))
    prim_ids = np.asarray(cast["primitive_ids"].numpy())
    geom_ids = np.asarray(cast["geometry_ids"].numpy())
    hit = (prim_ids == idxs) & (geom_ids == mesh_id)
    return {int(i) for i in idxs[hit]}


def compute_view_iou(
    dataset: str,
    scene_id: str,
    dataset_root: Path,
    gt_pose: list,
    pred_x: float,
    pred_y: float,
    pred_z: float,
    pred_yaw: float,
) -> Optional[float]:
    """Return the 3-D View IoU for the given GT pose vs predicted
    (x, y, z, yaw), or ``None`` if Open3D / mesh / FoV is unavailable.
    """
    fov = _DATASET_FOV_DEG.get(dataset)
    if fov is None:
        return None
    h_deg, v_deg = fov
    h_rad = math.radians(h_deg)
    v_rad = math.radians(v_deg)

    ctx = _get_iou_context(dataset, scene_id, dataset_root)
    if ctx is None:
        return None
    rc_scene, mesh_id, tri_points, tri_centroids, tri_areas = ctx

    pose_arr = np.asarray(gt_pose, dtype=np.float64)
    gt_cam = pose_arr[:3, 3]
    gt_fwd_raw = pose_arr[:3, :3] @ np.array([0.0, 0.0, 1.0])
    gt_fwd = _normalise(gt_fwd_raw)
    if gt_fwd is None:
        return None
    pred_cam = np.array([pred_x, pred_y, pred_z], dtype=np.float64)
    pred_fwd = np.array([math.cos(pred_yaw), math.sin(pred_yaw), 0.0])

    gt_vis = _visible_triangles(gt_cam, gt_fwd, h_rad, v_rad, rc_scene, mesh_id, tri_points, tri_centroids)
    pr_vis = _visible_triangles(pred_cam, pred_fwd, h_rad, v_rad, rc_scene, mesh_id, tri_points, tri_centroids)
    union = gt_vis | pr_vis
    if not union:
        return None
    inter = gt_vis & pr_vis
    inter_area = float(tri_areas[list(inter)].sum()) if inter else 0.0
    union_area = float(tri_areas[list(union)].sum())
    if union_area < 1e-9:
        return None
    return inter_area / union_area


def gt_pose_for(scene_id: str, frame_id: str, data_root: Path) -> Optional[list]:
    """Read the 4x4 ``scene_pose`` from the GT per-frame JSON. Used
    when a localization is submitted — the server reads the GT, the
    client never sees it (otherwise the user could cheat by inspecting
    the response)."""
    p = data_root / scene_id / "output" / "descriptions" / f"{frame_id}.json"
    if not p.exists():
        return None
    try:
        d = json.loads(p.read_text())
        return d.get("scene_pose")
    except Exception:
        return None


def pick_prompt_description(
    sess: Session, scene_id: str, frame_id: str, viewer_id: str
) -> Optional[Description]:
    """Pick which existing description the localizer should read.

    Rules:
    * Prefer descriptions from a DIFFERENT annotator than the viewer
      (avoid self-bias — they already know the scene if they wrote it).
    * Prefer unflagged descriptions over flagged ones.
    * Among matching candidates, pick the longest (most informative).
    """
    rows = sess.scalars(
        select(Description).where(
            Description.scene_id == scene_id,
            Description.frame_id == frame_id,
        )
    ).all()
    if not rows:
        return None

    def keyfn(d: Description) -> tuple:
        # lower tuple wins (sorted ascending)
        not_self = 0 if d.annotator_id != viewer_id else 1
        is_flagged = int(d.flagged or 0)
        neg_len = -int(d.word_count or 0)
        return (not_self, is_flagged, neg_len)

    return min(rows, key=keyfn)


# ---------------------------------------------------------------------------
# Localization assignment policy
# ---------------------------------------------------------------------------
def candidate_frames_for_localization(
    sess: Session,
    annotator_id: str,
    dataset: str,
    scene_id: Optional[str] = None,
) -> list:
    """Frames eligible for THIS annotator to localize.

    Eligibility (current policy, May 2026):
    * Scene is in the requested dataset.
    * If ``scene_id`` is given, restrict to that one scene (used by
      the scene-selector page).
    * Frame has at least one human description on file (the localizer
      has something to read).
    * NO ONE has localized this frame yet — strict global per-frame
      exclusion. Once any annotator submits a localization for
      ``(scene, frame)`` it disappears from the assignment pool for
      everyone.
    * The annotator has NOT skipped this specific (scene, frame).

    Annotators ARE free to localize many frames in the same scene
    (only the per-frame global cap applies).

    Returned rows: (scene_id, frame_id, difficulty_rank, n_descs,
    my_scene_count, scene_total).
    """
    sub_descs = (
        select(Description.scene_id, Description.frame_id,
               func.count().label("n_descs"))
        .group_by(Description.scene_id, Description.frame_id)
        .subquery()
    )
    # Frames anyone has localized — these are removed from the pool
    # for everyone (redundancy_target = 1 for localizations).
    sub_done = (
        select(HumanLocalization.scene_id, HumanLocalization.frame_id)
        .subquery()
    )
    # Specific (scene, frame) tuples this annotator skipped.
    sub_my_skips = (
        select(LocalizationSkip.scene_id, LocalizationSkip.frame_id)
        .where(LocalizationSkip.annotator_id == annotator_id)
        .subquery()
    )
    # How many localizations a scene has — used by Phase 2 (close
    # the most-partially-completed scene first).
    sub_scene_total = (
        select(HumanLocalization.scene_id, func.count().label("scene_n"))
        .group_by(HumanLocalization.scene_id)
        .subquery()
    )
    stmt = (
        select(
            Keyframe.scene_id,
            Keyframe.frame_id,
            Scene.difficulty_rank,
            sub_descs.c.n_descs,
            literal(0, type_=Integer).label("my_scene_count"),
            func.coalesce(sub_scene_total.c.scene_n, 0).label("scene_total"),
        )
        .join(Scene, Scene.id == Keyframe.scene_id)
        .join(
            sub_descs,
            (sub_descs.c.scene_id == Keyframe.scene_id)
            & (sub_descs.c.frame_id == Keyframe.frame_id),
        )
        .outerjoin(
            sub_done,
            (sub_done.c.scene_id == Keyframe.scene_id)
            & (sub_done.c.frame_id == Keyframe.frame_id),
        )
        .outerjoin(
            sub_my_skips,
            (sub_my_skips.c.scene_id == Keyframe.scene_id)
            & (sub_my_skips.c.frame_id == Keyframe.frame_id),
        )
        .outerjoin(
            sub_scene_total, sub_scene_total.c.scene_id == Keyframe.scene_id
        )
        .where(Scene.dataset == dataset)
        .where(sub_done.c.scene_id.is_(None))             # frame untouched globally
        .where(sub_my_skips.c.scene_id.is_(None))         # not skipped by me
    )
    if scene_id is not None:
        stmt = stmt.where(Keyframe.scene_id == scene_id)
    return list(sess.execute(stmt).all())


def assign_localization_frame(
    sess: Session,
    annotator_id: str,
    dataset: str,
    scene_id: Optional[str] = None,
) -> Optional[Tuple[str, str]]:
    """Pick a (scene_id, frame_id) for this annotator under the
    current policy:

    1. If ``scene_id`` is given (scene picker), restrict to that
       scene and pick its lowest-frame_id remaining frame.
    2. Otherwise: prefer to close someone else's partial scene
       (Phase 2 — scenes with the most localizations win, by
       ``difficulty_rank`` tie-break).
    3. Fall back to opening the lowest-``difficulty_rank`` fresh
       scene that still has unlocalized frames (Phase 3).

    Annotators are free to keep doing frames in the same scene; the
    per-frame global exclusion is what stops them from repeating
    other annotators' work.
    """
    rows = candidate_frames_for_localization(
        sess, annotator_id, dataset, scene_id=scene_id
    )
    if not rows:
        return None

    if scene_id is not None:
        # scene-picker mode: just pick the next remaining frame
        rows.sort(key=lambda r: r[1])  # frame_id asc
        return rows[0][0], rows[0][1]

    # Phase 2: close someone else's partial scene
    partial = [r for r in rows if r[5] > 0]  # scene_total > 0
    if partial:
        partial.sort(key=lambda r: (-r[5], r[2]))  # most-done scene first
        return partial[0][0], partial[0][1]

    # Phase 3: easiest fresh scene
    rows.sort(key=lambda r: r[2])  # difficulty_rank ascending
    return rows[0][0], rows[0][1]
