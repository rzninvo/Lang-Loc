"""Camera setup, rasterization, object visibility computation, and spatial relations."""
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
        seg_mode.append(max(vals, key=vals.count))  # majority vote over the three vertices
    seg_mode = np.asarray(seg_mode, dtype=np.int32)

    face_obj_ids = np.full(len(seg_mode), -1, dtype=np.int32)
    for i, seg in enumerate(seg_mode):
        face_obj_ids[i] = seg_to_obj.get(int(seg), -1)
    return face_obj_ids


def _compute_obb_from_points(points_world: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Compute an oriented bounding box (OBB) from a 3D point cloud.

    Returns center, orthonormal axes (rows), and half-extents.
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
    """
    Precompute full-mesh object geometry (centroid, AABB, OBB) per object id.
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
            continue

        coverage = px_count / total_image_pixels

        # Filter out low-coverage objects
        if coverage < coverage_threshold:
            continue

        # Faces belonging to this object that are actually visible
        mask = face_obj_ids[visible_face_ids] == oid
        if not np.any(mask):
            continue
        obj_face_ids = visible_face_ids[mask]
        if obj_face_ids.size == 0:
            continue

        # Visible-surface centroid is used for in-frame/depth gating only.
        visible_verts_idx = np.unique(F[obj_face_ids].reshape(-1))
        visible_verts = V[visible_verts_idx]
        if visible_verts.size == 0:
            continue

        visible_centroid_world = visible_verts.mean(axis=0)
        visible_centroid_cam = R_cv @ visible_centroid_world + t_cv
        if visible_centroid_cam[2] <= 0:
            continue

        # Filter: in front of camera & within depth range
        if visible_centroid_cam[2] < fov_depth_clip[0] or visible_centroid_cam[2] > fov_depth_clip[1]:
            continue
        # Filter: visible centroid should project inside image bounds (with slack)
        z = visible_centroid_cam[2]
        inv_z = 1.0 / z
        u = fx * (visible_centroid_cam[0] * inv_z) + cx
        v = fy * (visible_centroid_cam[1] * inv_z) + cy
        pad_w = 0.02 * image_width
        pad_h = 0.02 * image_height
        if (u < -pad_w) or (u > image_width + pad_w) or (v < -pad_h) or (v > image_height + pad_h):
            continue

        geom = full_object_geometry.get(int(oid))
        if geom is None:
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

        visible_objects[int(oid)] = {
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

    return visible_objects


def compute_spatial_relations(
    visible_objects: Dict[int, Dict[str, object]],
    max_distance: float = 2.0,
    size_ratio_threshold: float = 5.0,
    eps: float = 0.1,
) -> List[Dict[str, object]]:
    """
    Compute symbolic *spatial relations* between all pairs of visible objects
    based on their 3D centroids in **camera coordinates**.

    Instead of numeric distances, this function encodes *qualitative geometry*
    (e.g., "left_of", "above", "in_front_of"), which is more useful for language
    and scene description models such as GPT.

    Relations are determined by the **dominant displacement axis** between centroids:
      - X-axis -> "left_of" / "right_of"
      - Y-axis -> "above" / "below"
      - Z-axis -> "in_front_of" / "behind"

    The camera coordinate frame follows the OpenCV convention:
      X -> right,  Y -> down,  Z -> forward (depth).

    Args:
        visible_objects (dict): Mapping from object ID -> metadata dictionary.
                                Must include "centroid_cam" for each object.
        max_distance (float): Maximum distance (meters) between centroids to
                                consider a relation. Default = 2.0 m.
        size_ratio_threshold (float): Maximum size ratio between objects to
                                consider a relation. Default = 5.0.
        eps (float): Minimum displacement (meters) required to consider a
                     directional relation meaningful. Default = 0.1 m (10 cm).

    Returns:
        List[Dict[str, object]]: A list of pairwise relations, each entry
            contains the human-readable labels (lowercase, may be empty):
            [
              {"subject": "chair", "object": "table", "relation": "right_of", "distance": 0.5},
              {"subject": "chair", "object": "lamp", "relation": "in_front_of", "distance": 1.2},
                ...
            ]
    """

    # 1. Define Semantic Constraints
    # The list contains the allowed relations WHERE THE KEY IS THE SUBJECT.
    # e.g. "floor" can only be the SUBJECT of a "below" relation ("floor is below chair")
    SEMANTIC_RULES = {
        "ceiling": ["above"],
        "floor": ["below"],
        "wall": ["behind", "in_front_of"],  # Walls shouldn't be "left of" furniture
        "carpet": ["below", "under"],
        "rug": ["below", "under"],
    }

    spatial_relations = []
    visible_ids = list(visible_objects.keys())

    # Pre-calculate sizes (volume of bbox) for ratio checks
    # If bbox isn't perfect, we estimate size via diagonal length of bbox
    sizes = {}
    for oid in visible_ids:
        bbox = visible_objects[oid]["bbox_world"]
        # bbox is [[minx, miny, minz], [maxx, maxy, maxz]]
        diag = np.linalg.norm(np.array(bbox[1]) - np.array(bbox[0]))
        sizes[oid] = diag

    for i in range(len(visible_ids)):
        for j in range(i + 1, len(visible_ids)):
            id_a, id_b = visible_ids[i], visible_ids[j]
            obj_a = visible_objects[id_a]
            obj_b = visible_objects[id_b]

            # --- FILTER 1: Proximity (World Distance) ---
            # We use world coordinates for distance because camera depth can be misleading
            # (perspective distortion).
            center_a_world = np.array(obj_a["centroid_world"])
            center_b_world = np.array(obj_b["centroid_world"])
            dist_world = np.linalg.norm(center_a_world - center_b_world)

            if dist_world > max_distance:
                continue

            # --- FILTER 2: Size Ratio ---
            # Avoid relating tiny objects to massive structure unless necessary
            size_a = sizes[id_a]
            size_b = sizes[id_b]

            if size_a > 0 and size_b > 0:
                ratio = max(size_a, size_b) / min(size_a, size_b)
                if ratio > size_ratio_threshold:
                    continue

            # --- Geometric Calculation (Camera Coordinates) ---
            # We use Camera coords for Left/Right/Up/Down interpretation
            ca = np.array(obj_a["centroid_cam"])
            cb = np.array(obj_b["centroid_cam"])
            delta = ca - cb  # Vector from B to A

            axis = np.argmax(np.abs(delta))
            relation = None

            # OpenCV Coords: X (Right), Y (Down), Z (Forward)
            if axis == 0:  # X-axis
                if delta[0] > eps:
                    relation = "right_of"  # A is right of B
                elif delta[0] < -eps:
                    relation = "left_of"  # A is left of B
            elif axis == 1:  # Y-axis
                if delta[1] > eps:
                    relation = "below"  # A is below B (Y increases downwards)
                elif delta[1] < -eps:
                    relation = "above"  # A is above B
            elif axis == 2:  # Z-axis
                if delta[2] > eps:
                    relation = "behind"  # A is further (higher Z)
                elif delta[2] < -eps:
                    relation = "in_front_of"  # A is closer (lower Z)

            if not relation:
                continue

            # --- FILTER 3: Semantic Validity ---
            label_a = obj_a.get("label", "").lower()
            label_b = obj_b.get("label", "").lower()

            # Check if Subject (A) allows this relation. If not in rules, all relations allowed.
            if label_a in SEMANTIC_RULES:
                if relation not in SEMANTIC_RULES[label_a]:
                    continue

            spatial_relations.append(
                {
                    "subject": label_a or f"id_{id_a}",
                    "object": label_b or f"id_{id_b}",
                    "relation": relation,
                    "distance": float(dist_world),  # Helpful for debugging
                }
            )

    return spatial_relations
