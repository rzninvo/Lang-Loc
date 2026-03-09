"""Camera setup, rasterization, object visibility computation, and spatial relations.

Provides GPU-accelerated mesh rasterization via PyTorch3D to determine
per-pixel face visibility, per-object pixel counts, depth consistency
checks, and heuristic spatial relation computation for 3D indoor scenes.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from pytorch3d.renderer import (
    MeshRasterizer,
    PerspectiveCameras,
    RasterizationSettings,
)
from pytorch3d.structures import Meshes
from pytorch3d.utils import cameras_from_opencv_projection


# -------------------------------- Constants -----------------------------------

# Background/unlabeled value written to exported masks.
VOID_ID: int = 0


# ----------------------------- Cameras & Rasterizer ---------------------------


def make_p3d_camera_from_opencv(
    R_cv: np.ndarray,
    t_cv: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    H: int,
    W: int,
    device: torch.device,
) -> PerspectiveCameras:
    """
    Build a PyTorch3D camera that exactly matches an OpenCV pinhole camera.

    This uses PyTorch3D's `cameras_from_opencv_projection`, which:
    - Expects OpenCV world->camera (R, t) and a 3x3 intrinsics matrix K.
    - Takes image_size=(H, W) in pixels.
    - Returns a PerspectiveCameras that renders in the same pixel space as the RGB.

    Args:
        R_cv, t_cv: OpenCV world->cam rotation/translation.
        fx, fy, cx, cy: intrinsics (pixels).
        H, W: image size in pixels (height, width).
        device: CUDA or CPU device.

    Returns:
        PerspectiveCameras instance consistent with the inputs.
    """
    K = torch.tensor(
        [[[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]],
        dtype=torch.float32,
        device=device,
    )  # (1,3,3)
    R = torch.from_numpy(R_cv).float()[None].to(device)  # (1,3,3)
    t = torch.from_numpy(t_cv).float()[None].to(device)  # (1,3)
    image_size = torch.tensor([[H, W]], dtype=torch.float32, device=device)  # (1,2) = (H,W)
    cams = cameras_from_opencv_projection(R=R, tvec=t, camera_matrix=K, image_size=image_size)
    return cams


def make_p3d_cameras_batched(
    Rs_cv: list[np.ndarray],
    ts_cv: list[np.ndarray],
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    H: int,
    W: int,
    device: torch.device,
) -> PerspectiveCameras:
    """
    Build a batched PyTorch3D PerspectiveCameras from N OpenCV cameras.

    All cameras share the same intrinsics and image size.

    Args:
        Rs_cv: List of N (3,3) OpenCV world->cam rotation matrices.
        ts_cv: List of N (3,) OpenCV world->cam translation vectors.
        fx, fy, cx, cy: Shared intrinsics (pixels).
        H, W: Image size in pixels.
        device: CUDA or CPU device.

    Returns:
        PerspectiveCameras with batch dimension N.
    """
    N = len(Rs_cv)
    K = torch.tensor(
        [[[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]],
        dtype=torch.float32, device=device,
    ).expand(N, -1, -1)  # (N, 3, 3)

    R = torch.from_numpy(np.stack(Rs_cv, axis=0)).float().to(device)  # (N,3,3)
    t = torch.from_numpy(np.stack(ts_cv, axis=0)).float().to(device)  # (N,3)
    image_size = torch.tensor(
        [[H, W]], dtype=torch.float32, device=device,
    ).expand(N, -1)  # (N,2)

    cams = cameras_from_opencv_projection(
        R=R, tvec=t, camera_matrix=K, image_size=image_size,
    )
    return cams


@torch.no_grad()
def rasterize_visibility_batched(
    meshes: Meshes,
    cameras: PerspectiveCameras,
    rasterizer: MeshRasterizer,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Batched rasterization for N cameras at once.

    Args:
        meshes: PyTorch3D Meshes (single mesh, extended internally to batch N).
        cameras: PerspectiveCameras with batch size N.
        rasterizer: MeshRasterizer configured to the target image size.

    Returns:
        (pix_to_face, zbuf) tensors with shape (N, H, W).
    """
    N = cameras.R.shape[0]
    meshes_batch = meshes.extend(N)
    fragments = rasterizer(meshes_world=meshes_batch, cameras=cameras)
    p2f = fragments.pix_to_face[..., 0].to(torch.int64)  # (N, H, W)
    zbuf = fragments.zbuf[..., 0]  # (N, H, W)

    # Convert packed face indices to local (per-mesh) indices.
    # meshes.extend(N) replicates the mesh, so mesh i's faces start at i*F_orig.
    # Downstream code indexes face_obj_ids with shape (F_orig,), so we must
    # subtract the per-mesh offset to get local face indices.
    F_orig = meshes.num_faces_per_mesh()[0].item()
    for i in range(N):
        valid = p2f[i] >= 0
        p2f[i][valid] -= i * F_orig

    return p2f, zbuf


def make_rasterizer(
    H: int,
    W: int,
    faces_per_pixel: int = 1,
    bin_size: int | None = None,
    max_faces_per_bin: int | None = None,
    blur_radius: float = 0.0,
) -> MeshRasterizer:
    """
    Construct a MeshRasterizer with the given image size and options.

    Args:
        H, W: raster image size (pixels).
        faces_per_pixel: how many faces to keep per pixel.
        bin_size, max_faces_per_bin: tiling knobs for speed/memory on large meshes.
        blur_radius: soft rasterization blur; 0.0 = hard rasterization.

    Returns:
        MeshRasterizer configured with the given settings.
    """
    settings = RasterizationSettings(
        image_size=(H, W),
        blur_radius=blur_radius,
        faces_per_pixel=faces_per_pixel,
        bin_size=bin_size,
        max_faces_per_bin=max_faces_per_bin,
        perspective_correct=True,
        cull_backfaces=True,  # Cull backfaces for efficiency and correctness
    )
    return MeshRasterizer(raster_settings=settings)


# --------------------------- Visibility & Selection ---------------------------


def per_face_object_ids(
    F: np.ndarray, vert_seg: np.ndarray, seg_to_obj: Dict[int, int]
) -> np.ndarray:
    """
    Assign each face an object id via majority vote of its 3 vertices' segment ids.

    Faces whose majority segment is not present in seg_to_obj are set to -1.

    Args:
        F:        (Nf,3) face indices.
        vert_seg: (Nv,) segment id per vertex.
        seg_to_obj: mapping from segment id to object id.

    Returns:
        face_obj_ids: (Nf,) int32 array of object ids per face (-1 = unlabeled).
    """
    face_segs = vert_seg[F]  # (Nf,3)
    seg_mode = []
    for s0, s1, s2 in face_segs:
        vals = [int(s0), int(s1), int(s2)]
        seg_mode.append(max(vals, key=vals.count))
    seg_mode = np.asarray(seg_mode, dtype=np.int32)

    face_obj_ids = np.full(len(seg_mode), -1, dtype=np.int32)
    for i, seg in enumerate(seg_mode):
        face_obj_ids[i] = seg_to_obj.get(int(seg), -1)
    return face_obj_ids


def _compute_obb_from_points(points_world: np.ndarray) -> Dict[str, np.ndarray]:
    """Compute an oriented bounding box (OBB) from a 3D point cloud.

    Uses PCA to find the principal axes, then computes the tight bounding
    box in that rotated frame.

    Args:
        points_world: ``(N, 3)`` point cloud in world coordinates.

    Returns:
        Dict with ``center``, ``axes`` (3x3, rows are axis vectors),
        and ``extents`` (half-extents along each axis).
    """
    if points_world.size == 0:
        zero3 = np.zeros(3, dtype=np.float64)
        return {
            "center": zero3,
            "axes": np.eye(3, dtype=np.float64),
            "extents": zero3,
        }

    pts = np.asarray(points_world, dtype=np.float64)
    if pts.shape[0] < 3:
        bbox_min = pts.min(axis=0)
        bbox_max = pts.max(axis=0)
        return {
            "center": (bbox_min + bbox_max) / 2.0,
            "axes": np.eye(3, dtype=np.float64),
            "extents": (bbox_max - bbox_min) / 2.0,
        }

    mean = pts.mean(axis=0)
    centered = pts - mean
    cov = np.cov(centered, rowvar=False)

    # PCA axes (columns), sorted by descending variance.
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    axes_cols = eigvecs[:, order]
    if np.linalg.det(axes_cols) < 0.0:
        axes_cols[:, -1] *= -1.0

    projected = centered @ axes_cols
    pmin = projected.min(axis=0)
    pmax = projected.max(axis=0)
    extents = (pmax - pmin) / 2.0
    center_local = (pmax + pmin) / 2.0
    center_world = mean + axes_cols @ center_local

    return {
        "center": center_world.astype(np.float64),
        "axes": axes_cols.T.astype(np.float64),  # rows are axis vectors
        "extents": extents.astype(np.float64),
    }


def precompute_object_geometry(
    V: np.ndarray,
    F: np.ndarray,
    face_obj_ids: np.ndarray,
) -> Dict[int, Dict[str, np.ndarray]]:
    """Precompute full-mesh object geometry (centroid, AABB, OBB) per object id.

    Args:
        V: ``(Nv, 3)`` vertex positions.
        F: ``(Nf, 3)`` face indices.
        face_obj_ids: ``(Nf,)`` object id per face (-1 = unlabeled).

    Returns:
        Dict mapping object id to geometry dict with keys:
        ``centroid_world``, ``bbox_min``, ``bbox_max``,
        ``obb_center``, ``obb_axes``, ``obb_extents``.
    """
    geometry: Dict[int, Dict[str, np.ndarray]] = {}
    object_ids = np.unique(face_obj_ids[face_obj_ids >= 0])

    for oid in object_ids:
        obj_face_ids = np.where(face_obj_ids == oid)[0]
        if obj_face_ids.size == 0:
            continue

        vert_ids = np.unique(F[obj_face_ids].reshape(-1))
        verts_obj = V[vert_ids]
        if verts_obj.size == 0:
            continue

        centroid_world = verts_obj.mean(axis=0).astype(np.float64)
        bbox_min = verts_obj.min(axis=0).astype(np.float64)
        bbox_max = verts_obj.max(axis=0).astype(np.float64)
        obb = _compute_obb_from_points(verts_obj)

        geometry[int(oid)] = {
            "centroid_world": centroid_world,
            "bbox_min": bbox_min,
            "bbox_max": bbox_max,
            "obb_center": obb["center"],
            "obb_axes": obb["axes"],
            "obb_extents": obb["extents"],
        }

    return geometry


@torch.no_grad()
def rasterize_visibility(
    meshes: Meshes,
    cameras: PerspectiveCameras,
    rasterizer: MeshRasterizer,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Rasterize once to get:
        - Per-pixel face index for the closest face (pix_to_face[..., 0]).
        - Z-buffer (depth) of that closest face.

    Args:
        meshes: PyTorch3D Meshes (single mesh).
        cameras: PerspectiveCameras (from OpenCV projection).
        rasterizer: MeshRasterizer configured to the image size.

    Returns:
        (pix_to_face, zbuf) tensors with shape (H, W).
    """
    fragments = rasterizer(meshes_world=meshes, R=cameras.R, T=cameras.T, cameras=cameras)
    p2f = fragments.pix_to_face[0, ..., 0].to(torch.int64)  # (H,W)
    zbuf = fragments.zbuf[0, ..., 0]  # (H,W)
    return p2f, zbuf


def depth_consistency_mask(
    zbuf: torch.Tensor,
    sensor_depth: np.ndarray,
    vis_thres: float = 0.20,
    ignore_invalid_depth: bool = True,
) -> np.ndarray:
    """
    Compute a boolean mask of depth-consistent pixels.

    A pixel is depth-consistent when the sensor (measured) depth agrees with
    the rendered (mesh-rasterized) depth within a relative tolerance::

        |sensor_depth - zbuf| <= vis_thres * sensor_depth

    This mirrors the occlusion check in Open3DSG's ``compute_mapping()``.

    Args:
        zbuf: (H, W) rendered depth from the rasterizer (meters, >0 for valid).
        sensor_depth: (H, W) sensor depth in meters (>0 for valid measurements).
        vis_thres: Relative depth tolerance.  Default 0.20 (20 %) matches
            the Open3DSG production threshold.
        ignore_invalid_depth: If True, pixels where ``sensor_depth <= 0``
            are marked *inconsistent*.

    Returns:
        (H, W) bool ndarray — ``True`` for depth-consistent pixels.
    """
    zbuf_np = zbuf.cpu().numpy() if isinstance(zbuf, torch.Tensor) else zbuf

    consistent = np.abs(sensor_depth - zbuf_np) <= vis_thres * sensor_depth

    # Background pixels (no rendered face) are always inconsistent.
    consistent &= zbuf_np > 0

    if ignore_invalid_depth:
        consistent &= sensor_depth > 0

    return consistent


def compute_depth_consistent_counts(
    pix_to_face: torch.Tensor,
    face_obj_ids: np.ndarray,
    depth_mask: np.ndarray,
) -> Dict[int, int]:
    """
    Count depth-consistent pixels per object id.

    This is analogous to :func:`compute_image_visibility` but only counts
    pixels that pass the depth-consistency mask produced by
    :func:`depth_consistency_mask`.

    Args:
        pix_to_face: (H, W) face-index tensor from the rasterizer.
        face_obj_ids: (Nf,) object id per face (-1 = unlabeled).
        depth_mask: (H, W) boolean mask (``True`` = depth-consistent).

    Returns:
        Dict mapping ``object_id -> depth_consistent_pixel_count``.
    """
    p2f_flat = pix_to_face.cpu().numpy().reshape(-1)
    mask_flat = depth_mask.reshape(-1)

    valid = (p2f_flat >= 0) & mask_flat
    faces = p2f_flat[valid]
    if faces.size == 0:
        return {}

    obj_ids = face_obj_ids[faces]
    labeled = obj_ids >= 0
    obj_ids = obj_ids[labeled]

    counts: Dict[int, int] = {}
    if obj_ids.size > 0:
        unique, cnts = np.unique(obj_ids, return_counts=True)
        for u, c in zip(unique, cnts):
            counts[int(u)] = int(c)
    return counts


def compute_image_visibility(
    pix_to_face: torch.Tensor, face_obj_ids: np.ndarray
) -> Tuple[Dict[int, int], int]:
    """
    Accumulate labeled pixel counts per object id for one image.

    Args:
        pix_to_face: (H,W) tensor of face indices (-1 for background).
        face_obj_ids: (Nf,) object id per face (-1 for unlabeled).

    Returns:
        obj_px: dict {objectId: pixel_count}
        total_labeled_px: int total labeled pixels.
    """
    p2f_flat = pix_to_face.cpu().numpy().reshape(-1)
    valid = p2f_flat >= 0
    faces = p2f_flat[valid]
    if faces.size == 0:
        return {}, 0

    obj_ids = face_obj_ids[faces]
    labeled = obj_ids >= 0
    obj_ids = obj_ids[labeled]
    total = int(labeled.sum())

    vis: Dict[int, int] = defaultdict(int)
    if total > 0:
        unique, counts = np.unique(obj_ids, return_counts=True)
        for u, c in zip(unique, counts):
            vis[int(u)] += int(c)
    return dict(vis), total


def compute_visible_objects(
    V: np.ndarray,
    F: np.ndarray,
    vert_seg: np.ndarray,
    seg_to_obj: Dict[int, int],
    obj_to_label: Dict[int, str],
    obj_px: Dict[int, int],
    face_obj_ids: np.ndarray,
    pix_to_face: torch.Tensor,
    pose: np.ndarray,
    R_cv: np.ndarray,
    t_cv: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    image_width: int,
    image_height: int,
    fov_depth_clip: Tuple[float, float] = (0.2, 10.0),
    coverage_threshold: float = 0.05,
    min_pixel_count: int = 50,
    full_object_geometry: Optional[Dict[int, Dict[str, np.ndarray]]] = None,
    object_attributes: Optional[Dict[int, Dict[str, object]]] = None,
    depth_consistent_counts: Optional[Dict[int, int]] = None,
    depth_consistent_ratio_threshold: float = 0.0,
    min_depth_consistent_pixels: int = 0,
    filtered_reasons: Optional[Dict[int, str]] = None,
) -> Dict[int, Dict[str, object]]:
    """
    Compute per-object metadata for objects visible in this frame.

    Visibility gating is based on rendered pixels/visible faces, but geometric
    descriptors (`centroid_world`, `bbox_world`, `obb_world`) come from the
    complete object mesh.

    This function extracts geometric properties (centroid, bounding box, distance)
    for all objects that are visible in the current view (as determined by `obj_px`).

    The returned information is later used by GPT-based description models or
    downstream spatial reasoning modules.

    Coordinate conventions:
      - `V` and `pose` are in **world coordinates**.
      - `R_cv`, `t_cv` represent the OpenCV-style world->camera transform.
      - The camera coordinate system follows OpenCV / PyTorch3D conventions:
            X -> right,  Y -> down,  Z -> forward (depth).

    Args:
        F (np.ndarray):         (Nf, 3) array of face indices.
        V (np.ndarray):         (Nv, 3) array of vertex positions in world coordinates.
        vert_seg (np.ndarray):  (Nv,) integer array mapping each vertex to a segment ID.
        seg_to_obj (dict):      Mapping from segment ID -> object ID (instance).
        obj_to_label (dict):    Mapping from object ID -> raw label string (lowercased).
        obj_px (dict):          Mapping from visible object ID -> number of visible pixels.
        face_obj_ids (np.ndarray): (Nf,) integer array mapping each face to an object ID (-1 = unlabeled).
        pix_to_face (torch.Tensor): (H,W) tensor of face indices per pixel.
        pose (np.ndarray):      (4,4) camera-to-world SE(3) matrix for this frame.
        R_cv (np.ndarray):      (3,3) world->camera rotation matrix (OpenCV convention).
        t_cv (np.ndarray):      (3,) world->camera translation vector (OpenCV convention).
        fx, fy (float):         Focal lengths of the rasterization camera (pixels).
        cx, cy (float):         Principal point of the rasterization camera (pixels).
        image_width (int):      Rasterized image width (pixels).
        image_height (int):     Rasterized image height (pixels).
        fov_depth_clip (tuple): Min/max depth (m) for an object centroid to be considered visible.
        coverage_threshold (float): Minimum percent of TOTAL IMAGE pixels (0.0-1.0) for an object to be considered visible.
        min_pixel_count (int):  Minimum absolute pixel count for an object to be considered visible.
        full_object_geometry: Optional precomputed object geometry from
            `precompute_object_geometry(...)`. If None, computed on-demand.
        object_attributes: Optional mapping `objectId -> attributes` to attach
            to each visible object entry (e.g., loaded from 3RScan `objects.json`).
        depth_consistent_counts: Optional per-object depth-consistent pixel
            counts from :func:`compute_depth_consistent_counts`. When provided,
            enables depth-based gating and adds ``depth_consistent_pixels``
            and ``depth_consistent_ratio`` to each output entry.
        depth_consistent_ratio_threshold: Minimum depth-consistent ratio
            (depth_consistent_pixels / visible_pixels) to keep an object.
            Set to 0.0 (default) for log-only mode without gating.
        min_depth_consistent_pixels: Minimum absolute depth-consistent pixel
            count to keep an object.  Set to 0 (default) for no gating.
        filtered_reasons: Optional dict that, if provided, will be populated
            with ``{object_id: reason_string}`` for every object that was
            filtered out. Reason strings include: ``"too_few_pixels"``,
            ``"low_coverage"``, ``"depth_few_consistent_px"``,
            ``"depth_low_ratio"``, ``"no_visible_faces"``,
            ``"no_visible_verts"``, ``"behind_camera"``,
            ``"outside_depth_range"``, ``"outside_image_bounds"``,
            ``"no_geometry"``.

    Returns:
        Dict[int, Dict[str, object]]:
        Mapping from object ID to a metadata dictionary with fields:
            {
                "label": str,                     # object name (e.g., "chair")
                "pixel_percent": float,           # percent of image pixels
                "centroid_world": [x, y, z],      # world-space centroid (meters)
                "centroid_ndc": [ndc_x, ndc_y],   # NDC coords of centroid in image frame
                "bbox_world": [[minx, miny, minz], [maxx, maxy, maxz]],
                "obb_world": {
                    "center": [x, y, z],
                    "axes": [[ax1x, ax1y, ax1z], [ax2x, ax2y, ax2z], [ax3x, ax3y, ax3z]],
                    "extents": [ex, ey, ez],
                },
                "centroid_cam": [x, y, z],        # camera-space centroid
                "distance_from_camera": float     # Euclidean distance (m) from camera to centroid
                "visible_faces": int,             # number of visible faces for this object
                "visible_pixels": int,            # number of visible pixels for this object
                "depth_consistent_pixels": int,   # (if depth_consistent_counts provided)
                "depth_consistent_ratio": float,  # (if depth_consistent_counts provided)
                "attributes": dict                # optional semantic attributes (if provided)
            }
    """
    cam_pos = pose[:3, 3]
    visible_objects = {}

    # Determine which faces were visible in the image
    p2f_np = pix_to_face.cpu().numpy()
    visible_face_ids = np.unique(p2f_np[p2f_np >= 0])

    if full_object_geometry is None:
        full_object_geometry = precompute_object_geometry(V, F, face_obj_ids)

    # Calculate total image pixels for correct percentage computation
    total_image_pixels = image_width * image_height
    if total_image_pixels <= 0:
        return visible_objects

    for oid, px_count in obj_px.items():

        if px_count < min_pixel_count:
            if filtered_reasons is not None:
                filtered_reasons[oid] = "too_few_pixels"
            continue

        coverage = px_count / total_image_pixels

        # Filter out low-coverage objects
        if coverage < coverage_threshold:
            if filtered_reasons is not None:
                filtered_reasons[oid] = "low_coverage"
            continue

        # Depth consistency gating (optional)
        if depth_consistent_counts is not None:
            dc_px = depth_consistent_counts.get(oid, 0)
            dc_ratio = dc_px / px_count if px_count > 0 else 0.0
            if dc_px < min_depth_consistent_pixels:
                if filtered_reasons is not None:
                    filtered_reasons[oid] = "depth_few_consistent_px"
                continue
            if dc_ratio < depth_consistent_ratio_threshold:
                if filtered_reasons is not None:
                    filtered_reasons[oid] = "depth_low_ratio"
                continue

        # Faces belonging to this object that are actually visible
        mask = face_obj_ids[visible_face_ids] == oid
        if not np.any(mask):
            if filtered_reasons is not None:
                filtered_reasons[oid] = "no_visible_faces"
            continue
        obj_face_ids = visible_face_ids[mask]
        if obj_face_ids.size == 0:
            if filtered_reasons is not None:
                filtered_reasons[oid] = "no_visible_faces"
            continue

        # Visible-surface centroid is used for in-frame/depth gating only.
        visible_verts_idx = np.unique(F[obj_face_ids].reshape(-1))
        visible_verts = V[visible_verts_idx]
        if visible_verts.size == 0:
            if filtered_reasons is not None:
                filtered_reasons[oid] = "no_visible_verts"
            continue

        visible_centroid_world = visible_verts.mean(axis=0)
        visible_centroid_cam = R_cv @ visible_centroid_world + t_cv
        if visible_centroid_cam[2] <= 0:
            if filtered_reasons is not None:
                filtered_reasons[oid] = "behind_camera"
            continue

        # Filter: in front of camera & within depth range
        if visible_centroid_cam[2] < fov_depth_clip[0] or visible_centroid_cam[2] > fov_depth_clip[1]:
            if filtered_reasons is not None:
                filtered_reasons[oid] = "outside_depth_range"
            continue
        # Filter: visible centroid should project inside image bounds (with slack)
        z = visible_centroid_cam[2]
        inv_z = 1.0 / z
        u = fx * (visible_centroid_cam[0] * inv_z) + cx
        v = fy * (visible_centroid_cam[1] * inv_z) + cy
        pad_w = 0.02 * image_width
        pad_h = 0.02 * image_height
        if (u < -pad_w) or (u > image_width + pad_w) or (v < -pad_h) or (v > image_height + pad_h):
            if filtered_reasons is not None:
                filtered_reasons[oid] = "outside_image_bounds"
            continue

        geom = full_object_geometry.get(int(oid))
        if geom is None:
            if filtered_reasons is not None:
                filtered_reasons[oid] = "no_geometry"
            continue

        centroid_world = geom["centroid_world"]
        centroid_cam = R_cv @ centroid_world + t_cv
        z_full = max(float(centroid_cam[2]), 1e-8)
        u_full = fx * (float(centroid_cam[0]) / z_full) + cx
        v_full = fy * (float(centroid_cam[1]) / z_full) + cy
        ndc_x = (u_full / image_width) * 2 - 1
        ndc_y = 1 - (v_full / image_height) * 2
        dist_from_cam = float(np.linalg.norm(centroid_world - cam_pos))
        bbox_min, bbox_max = geom["bbox_min"], geom["bbox_max"]

        object_entry = {
            "label": obj_to_label.get(int(oid), ""),
            "pixel_percent": float(round(coverage * 100.0, 3)),
            "centroid_world": centroid_world.tolist(),
            "centroid_ndc": [ndc_x, ndc_y],
            "bbox_world": [bbox_min.tolist(), bbox_max.tolist()],
            "obb_world": {
                "center": geom["obb_center"].tolist(),
                "axes": geom["obb_axes"].tolist(),
                "extents": geom["obb_extents"].tolist(),
            },
            "centroid_cam": centroid_cam.tolist(),
            "distance_from_camera": dist_from_cam,
            "visible_faces": int(obj_face_ids.size),
            "visible_pixels": int(px_count),
        }
        if depth_consistent_counts is not None:
            dc_px = depth_consistent_counts.get(oid, 0)
            object_entry["depth_consistent_pixels"] = dc_px
            object_entry["depth_consistent_ratio"] = (
                round(dc_px / px_count, 4) if px_count > 0 else 0.0
            )
        if object_attributes is not None:
            attrs = object_attributes.get(int(oid), {})
            if isinstance(attrs, dict):
                object_entry["attributes"] = attrs
            else:
                object_entry["attributes"] = {}
        visible_objects[int(oid)] = object_entry

    return visible_objects


def compute_spatial_relations(
    visible_objects: Dict[int, Dict[str, object]],
    max_distance: float = 2.0,
    size_ratio_threshold: float = 5.0,
    eps: float = 0.1,
) -> List[Dict[str, object]]:
    """
    Compute heuristic spatial relations between all pairs of visible objects.

    Emits multiple relations per pair. Relation names match the 3DSSG predicate
    vocabulary (``relationships.txt``).

    **Directional** (camera-frame, dominant axis):
      left, right, front, behind, higher than, lower than

    **Comparative** (world-frame bbox):
      bigger than, smaller than

    **Proximity / support** (world-frame):
      close by, standing on, supported by

    Camera coordinate frame follows OpenCV: X->right, Y->down, Z->forward.

    Args:
        visible_objects: Object ID -> metadata dict. Must include
            ``centroid_cam``, ``centroid_world``, ``bbox_world``.
        max_distance: Max world distance (m) between centroids.
        size_ratio_threshold: Max bbox diagonal ratio to consider a pair.
        eps: Min displacement (m) for directional relations.

    Returns:
        List of relation dicts with keys:
        ``subject``, ``object``, ``relation``, ``distance``.
    """

    # Semantic constraints: restrict which relations certain labels
    # can be the SUBJECT of.
    SEMANTIC_RULES = {
        "ceiling": {"higher than"},
        "floor": {"lower than", "supported by"},
        "wall": {"behind", "front", "close by"},
        "carpet": {"lower than", "supported by"},
        "rug": {"lower than", "supported by"},
    }

    spatial_relations: List[Dict[str, object]] = []
    visible_ids = list(visible_objects.keys())

    # Pre-calculate bbox diagonal and volume per object
    sizes = {}
    volumes = {}
    for oid in visible_ids:
        bbox = visible_objects[oid]["bbox_world"]
        bmin = np.array(bbox[0])
        bmax = np.array(bbox[1])
        dims = np.maximum(bmax - bmin, 1e-8)
        sizes[oid] = float(np.linalg.norm(dims))
        volumes[oid] = float(dims[0] * dims[1] * dims[2])

    def _add(subj_label, subj_id, obj_label, obj_id, relation, dist):
        sl = subj_label.lower() if subj_label else f"id_{subj_id}"
        allowed = SEMANTIC_RULES.get(sl)
        if allowed is not None and relation not in allowed:
            return
        spatial_relations.append({
            "subject": sl,
            "object": obj_label.lower() if obj_label else f"id_{obj_id}",
            "relation": relation,
            "distance": float(dist),
        })

    for i in range(len(visible_ids)):
        for j in range(i + 1, len(visible_ids)):
            id_a, id_b = visible_ids[i], visible_ids[j]
            obj_a = visible_objects[id_a]
            obj_b = visible_objects[id_b]
            label_a = obj_a.get("label", "")
            label_b = obj_b.get("label", "")

            # World-space distance filter
            cw_a = np.array(obj_a["centroid_world"])
            cw_b = np.array(obj_b["centroid_world"])
            dist_world = float(np.linalg.norm(cw_a - cw_b))
            if dist_world > max_distance:
                continue

            # Size ratio filter
            size_a, size_b = sizes[id_a], sizes[id_b]
            if size_a > 0 and size_b > 0:
                if max(size_a, size_b) / min(size_a, size_b) > size_ratio_threshold:
                    continue

            # Camera-space centroids
            ca = np.array(obj_a["centroid_cam"])
            cb = np.array(obj_b["centroid_cam"])
            delta = ca - cb  # vector from B to A in camera frame

            # ── 1. DIRECTIONAL (dominant axis, camera-frame) ──
            axis = int(np.argmax(np.abs(delta)))
            # OpenCV: X=right, Y=down, Z=forward
            if axis == 0 and abs(delta[0]) > eps:
                if delta[0] > 0:
                    _add(label_a, id_a, label_b, id_b, "right", dist_world)
                    _add(label_b, id_b, label_a, id_a, "left", dist_world)
                else:
                    _add(label_a, id_a, label_b, id_b, "left", dist_world)
                    _add(label_b, id_b, label_a, id_a, "right", dist_world)
            elif axis == 1 and abs(delta[1]) > eps:
                if delta[1] > 0:
                    _add(label_a, id_a, label_b, id_b, "lower than", dist_world)
                    _add(label_b, id_b, label_a, id_a, "higher than", dist_world)
                else:
                    _add(label_a, id_a, label_b, id_b, "higher than", dist_world)
                    _add(label_b, id_b, label_a, id_a, "lower than", dist_world)
            elif axis == 2 and abs(delta[2]) > eps:
                if delta[2] > 0:
                    _add(label_a, id_a, label_b, id_b, "behind", dist_world)
                    _add(label_b, id_b, label_a, id_a, "front", dist_world)
                else:
                    _add(label_a, id_a, label_b, id_b, "front", dist_world)
                    _add(label_b, id_b, label_a, id_a, "behind", dist_world)

            # ── 2. COMPARATIVE: bigger than / smaller than ──
            vol_a, vol_b = volumes[id_a], volumes[id_b]
            if vol_a > 0 and vol_b > 0:
                vol_ratio = vol_a / vol_b
                if vol_ratio > 1.5:
                    _add(label_a, id_a, label_b, id_b, "bigger than", dist_world)
                    _add(label_b, id_b, label_a, id_a, "smaller than", dist_world)
                elif vol_ratio < 1.0 / 1.5:
                    _add(label_a, id_a, label_b, id_b, "smaller than", dist_world)
                    _add(label_b, id_b, label_a, id_a, "bigger than", dist_world)

            # ── 3. PROXIMITY: close by ──
            if dist_world < 0.5:
                _add(label_a, id_a, label_b, id_b, "close by", dist_world)
                _add(label_b, id_b, label_a, id_a, "close by", dist_world)

            # ── 4. SUPPORT: standing on / supported by ──
            # A standing on B: A is higher (smaller cam Y), horizontally close
            dy_cam = delta[1]  # positive = A lower, negative = A higher
            if abs(dy_cam) > eps and dist_world < 1.0:
                horiz_dist = float(np.sqrt(delta[0]**2 + delta[2]**2))
                if horiz_dist < max(size_a, size_b) * 0.7:
                    if dy_cam < -eps:
                        # A is higher → A standing on B
                        _add(label_a, id_a, label_b, id_b, "standing on", dist_world)
                        _add(label_b, id_b, label_a, id_a, "supported by", dist_world)
                    else:
                        _add(label_b, id_b, label_a, id_a, "standing on", dist_world)
                        _add(label_a, id_a, label_b, id_b, "supported by", dist_world)

    return spatial_relations


# ----------------------------- Debug Visualization ----------------------------


def save_depth_debug_panel(
    rgb_path,
    sensor_depth: np.ndarray,
    zbuf,
    depth_mask: np.ndarray,
    pix_to_face,
    face_obj_ids: np.ndarray,
    visible_objects: Dict[int, Dict],
    out_path,
    fid: str = "",
    vis_thres: float = 0.20,
    obj_px: Optional[Dict[int, int]] = None,
    obj_to_label: Optional[Dict[int, str]] = None,
    filtered_reasons: Optional[Dict[int, str]] = None,
) -> None:
    """
    Save a 3x3 diagnostic figure for depth-consistency validation.

    Panels:
        (0,0) RGB image
        (0,1) Sensor depth (viridis colormap)
        (0,2) Rendered depth / zbuf (same colormap)
        (1,0) |sensor - rendered| absolute difference
        (1,1) Consistency mask overlay (green = pass, red = fail)
        (1,2) Per-object depth-consistent ratio (green = high, red = low)
        (2,0) All raster-visible objects (before filtering)
        (2,1) Surviving objects after filtering (filtered objects in gray)
        (2,2) Summary statistics

    Args:
        rgb_path: Path to the RGB image for this frame.
        sensor_depth: (H, W) float32 sensor depth in meters.
        zbuf: (H, W) rendered depth (torch Tensor or ndarray).
        depth_mask: (H, W) bool mask from ``depth_consistency_mask()``.
        pix_to_face: (H, W) face-index tensor from the rasterizer.
        face_obj_ids: (Nf,) object id per face.
        visible_objects: Output of ``compute_visible_objects()`` (with dc fields).
        out_path: Destination PNG path.
        fid: Frame id string (used in the title).
        vis_thres: Threshold used (shown in title).
        obj_px: Raw per-object pixel counts from ``compute_image_visibility()``
            (before depth filtering). Used for before/after comparison.
        obj_to_label: Object id to label name mapping.
        filtered_reasons: Optional dict ``{object_id: reason_string}`` from
            ``compute_visible_objects()``. Displayed next to each filtered
            object in the summary panel.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pathlib import Path
    from PIL import Image

    zbuf_np = zbuf.cpu().numpy() if isinstance(zbuf, torch.Tensor) else np.asarray(zbuf)
    p2f_np = pix_to_face.cpu().numpy() if isinstance(pix_to_face, torch.Tensor) else np.asarray(pix_to_face)
    H, W = sensor_depth.shape[:2]

    # Load and resize RGB to match depth resolution
    rgb = np.array(Image.open(rgb_path).resize((W, H), Image.BILINEAR))

    # Per-pixel object id map (shared across panels)
    obj_ids_px = np.full((H, W), -1, dtype=np.int32)
    valid_faces = p2f_np >= 0
    if valid_faces.any():
        obj_ids_px[valid_faces] = face_obj_ids[p2f_np[valid_faces]]

    all_raster_oids = set(int(x) for x in np.unique(obj_ids_px[obj_ids_px >= 0]))
    surviving_oids = set(
        int(k) if isinstance(k, str) else k for k in visible_objects.keys()
    )
    filtered_oids = all_raster_oids - surviving_oids

    # Stable per-object colour via tab20 colormap
    cmap = plt.cm.get_cmap("tab20")

    def _obj_color(oid: int):
        return np.array(cmap(hash(oid) % 20)[:3])

    # Shared depth colour range
    valid_depths = sensor_depth[sensor_depth > 0]
    vmin = 0.0
    vmax = float(np.percentile(valid_depths, 98)) if valid_depths.size > 0 else 5.0

    fig, axes = plt.subplots(3, 3, figsize=(18, 16))
    fig.suptitle(
        f"Depth Consistency — Frame {fid}  (threshold = {vis_thres:.0%})",
        fontsize=14,
    )

    # ======================= ROW 0: Depth data =======================

    # (0,0) RGB
    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title("RGB")
    axes[0, 0].axis("off")

    # (0,1) Sensor depth
    sd = np.where(sensor_depth > 0, sensor_depth, np.nan)
    im1 = axes[0, 1].imshow(sd, cmap="viridis", vmin=vmin, vmax=vmax)
    axes[0, 1].set_title("Sensor Depth (m)")
    axes[0, 1].axis("off")
    fig.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

    # (0,2) Rendered depth (zbuf)
    zb = np.where(zbuf_np > 0, zbuf_np, np.nan)
    im2 = axes[0, 2].imshow(zb, cmap="viridis", vmin=vmin, vmax=vmax)
    axes[0, 2].set_title("Rendered Depth / zbuf (m)")
    axes[0, 2].axis("off")
    fig.colorbar(im2, ax=axes[0, 2], fraction=0.046, pad=0.04)

    # ===================== ROW 1: Consistency ========================

    # (1,0) |sensor − rendered| difference
    diff = np.abs(sensor_depth - zbuf_np)
    diff_display = np.where((sensor_depth > 0) & (zbuf_np > 0), diff, np.nan)
    im3 = axes[1, 0].imshow(
        diff_display, cmap="hot", vmin=0, vmax=max(vmax * vis_thres * 2, 0.5)
    )
    axes[1, 0].set_title("|Sensor − Rendered| (m)")
    axes[1, 0].axis("off")
    fig.colorbar(im3, ax=axes[1, 0], fraction=0.046, pad=0.04)

    # (1,1) Consistency mask overlay on RGB
    overlay = rgb.astype(np.float32) / max(float(rgb.max()), 1.0)
    has_data = (sensor_depth > 0) & (zbuf_np > 0)
    green = depth_mask & has_data
    red = ~depth_mask & has_data
    overlay[green] = overlay[green] * 0.4 + np.array([0, 0.8, 0]) * 0.6
    overlay[red] = overlay[red] * 0.4 + np.array([0.8, 0, 0]) * 0.6
    overlay[~has_data] *= 0.3
    axes[1, 1].imshow(np.clip(overlay, 0, 1))
    n_pass = int(green.sum())
    n_total = int(has_data.sum())
    pct = n_pass / n_total * 100 if n_total > 0 else 0
    axes[1, 1].set_title(f"Consistency Mask ({pct:.1f}% pass, {n_pass}/{n_total})")
    axes[1, 1].axis("off")

    # (1,2) Per-object depth-consistent ratio map
    obj_map = np.full((H, W, 3), 0.15, dtype=np.float32)
    for oid_key, meta in visible_objects.items():
        oid = int(oid_key) if isinstance(oid_key, str) else oid_key
        dc_ratio = meta.get("depth_consistent_ratio", 1.0)
        pmask = obj_ids_px == oid
        if not pmask.any():
            continue
        obj_map[pmask] = [1.0 - dc_ratio, dc_ratio, 0.1]

    axes[1, 2].imshow(np.clip(obj_map, 0, 1))
    axes[1, 2].set_title("Per-Object DC Ratio (green=high, red=low)")
    axes[1, 2].axis("off")

    if len(visible_objects) <= 15:
        for oid_key, meta in visible_objects.items():
            dc_ratio = meta.get("depth_consistent_ratio")
            if dc_ratio is None:
                continue
            label = meta.get("label", "")
            ndc = meta.get("centroid_ndc", [0, 0])
            px_x = (ndc[0] + 1) / 2 * W
            px_y = (1 - ndc[1]) / 2 * H
            if 0 <= px_x < W and 0 <= px_y < H:
                axes[1, 2].text(
                    px_x, px_y,
                    f"{label}\n{dc_ratio:.0%}",
                    fontsize=6, color="white", ha="center", va="center",
                    bbox=dict(
                        boxstyle="round,pad=0.2", facecolor="black", alpha=0.6
                    ),
                )

    # ================ ROW 2: Before / After filter ===================

    # (2,0) All raster-visible objects (before any filtering)
    before_map = np.full((H, W, 3), 0.1, dtype=np.float32)
    for oid in all_raster_oids:
        pmask = obj_ids_px == oid
        before_map[pmask] = _obj_color(oid)
    axes[2, 0].imshow(np.clip(before_map, 0, 1))
    axes[2, 0].set_title(f"All Raster Objects ({len(all_raster_oids)})")
    axes[2, 0].axis("off")

    # (2,1) Surviving objects (kept = colour, filtered = gray)
    after_map = np.full((H, W, 3), 0.1, dtype=np.float32)
    for oid in all_raster_oids:
        pmask = obj_ids_px == oid
        if oid in surviving_oids:
            after_map[pmask] = _obj_color(oid)
        else:
            after_map[pmask] = [0.35, 0.35, 0.35]
    axes[2, 1].imshow(np.clip(after_map, 0, 1))
    axes[2, 1].set_title(
        f"After Filtering ({len(surviving_oids)} kept, "
        f"{len(filtered_oids)} removed)"
    )
    axes[2, 1].axis("off")

    # (2,2) Summary statistics
    axes[2, 2].axis("off")
    _label = obj_to_label or {}
    lines = [
        f"Raster objects:  {len(all_raster_oids)}",
        f"Surviving:       {len(surviving_oids)}",
        f"Filtered out:    {len(filtered_oids)}",
        "",
    ]
    if filtered_oids:
        lines.append("--- Filtered objects ---")
        _reasons = filtered_reasons or {}
        for oid in sorted(filtered_oids):
            name = _label.get(oid, f"id_{oid}")
            raw_px = obj_px.get(oid, 0) if obj_px else "?"
            reason = _reasons.get(oid, "unknown")
            lines.append(f"  {name} (oid={oid}): {raw_px} px — {reason}")
    if surviving_oids:
        lines.append("")
        lines.append("--- Surviving objects ---")
        for oid in sorted(surviving_oids):
            meta = visible_objects.get(oid) or visible_objects.get(str(oid), {})
            name = meta.get("label", _label.get(oid, f"id_{oid}"))
            dc_r = meta.get("depth_consistent_ratio")
            dc_px = meta.get("depth_consistent_pixels", "?")
            vis_px = meta.get("visible_pixels", "?")
            dc_str = f"dc={dc_r:.0%}" if dc_r is not None else "dc=n/a"
            lines.append(f"  {name}: {vis_px} px, {dc_str} ({dc_px} dc_px)")

    axes[2, 2].text(
        0.02, 0.98,
        "\n".join(lines),
        transform=axes[2, 2].transAxes,
        fontsize=7, fontfamily="monospace",
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.9),
    )
    axes[2, 2].set_title("Filter Summary")

    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
