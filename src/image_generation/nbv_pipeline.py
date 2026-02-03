"""
Shared Next-Best-View (NBV) Pipeline Components.

This module provides the common functionality used by both ScanNet++ and 3RScan
NBV selection pipelines. It includes:

- BRISQUE-based image quality filtering
- PyTorch3D camera and rasterizer utilities
- Visibility computation and analysis
- Greedy next-best-view selection algorithm
- Spatial relations computation
- Mask export utilities (instance and semantic)
"""
from __future__ import annotations

import multiprocessing as mp
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# PyTorch3D
from pytorch3d.renderer import (
    MeshRasterizer,
    PerspectiveCameras,
    RasterizationSettings,
)
from pytorch3d.structures import Meshes
from pytorch3d.utils import cameras_from_opencv_projection
from brisque import BRISQUE


# -------------------------------- Constants -----------------------------------

# Background/unlabeled value written to exported masks.
VOID_ID: int = 0


# ----------------------------- BRISQUE Filtering ------------------------------

_global_brisque = None


def _init_brisque_worker():
    """Create one BRISQUE instance per process and limit OpenCV threading."""
    global _global_brisque
    cv2.setNumThreads(1)  # avoid CPU oversubscription inside workers
    _global_brisque = BRISQUE()


def _score_one(image_path: Path):
    """Return (stem, score) for a single image; inf on failure."""
    global _global_brisque
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        return (image_path.stem, float("inf"))
    try:
        score = _global_brisque.score(img)  # no resizing, full-res BRISQUE
    except Exception:
        score = float("inf")
    return (image_path.stem, score)


def compute_sharpness(image_path: Path) -> float:
    """Single-image BRISQUE score (kept for compatibility)."""
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not read image from {image_path}")
    scorer = BRISQUE()
    return scorer.score(img)


def filter_sharp_images(
    color_dir: Path,
    threshold: float,
    workers: int | None = None,
    file_pattern: str = "*.jpg",
) -> list[str]:
    """
    Parallel BRISQUE filtering without caching or downsampling.

    Args:
        color_dir: directory containing image frames.
        threshold: keep frames with BRISQUE <= threshold (lower is better).
        workers: number of processes; default = max(1, cpu_count()-1).
        file_pattern: glob pattern for image files (default: "*.jpg").

    Returns:
        Ordered list of frame ids (stems) that pass the threshold.
    """
    image_files = sorted(color_dir.glob(file_pattern))
    if not image_files:
        return []

    if workers is None:
        default_workers = max(1, mp.cpu_count() - 1)
        print(f"[WARN] workers not set; using max(1, cpu_count()-1) which is {default_workers}")
        workers = default_workers

    sharp_set = {}
    try:
        ctx = mp.get_context("spawn")
    except ValueError:
        ctx = mp.get_context()
    with ctx.Pool(processes=workers, initializer=_init_brisque_worker) as pool:
        for stem, score in tqdm(
            pool.imap_unordered(_score_one, image_files, chunksize=8),
            total=len(image_files),
            desc="BRISQUE filtering",
            dynamic_ncols=True,
        ):
            sharp_set[stem] = score

    kept = [p.stem for p in image_files if sharp_set.get(p.stem, float("inf")) <= threshold]
    print(f"[INFO] {len(kept)}/{len(image_files)} images passed (threshold={threshold})")
    return kept


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
) -> Dict[int, Dict[str, object]]:
    """
    Compute per-object 3D and camera-space metadata for *actually visible surfaces*.
    The centroid and bbox are computed only from the triangles that were rendered,
    not the full mesh.

    This function extracts geometric properties (centroid, bounding box, distance)
    for all objects that are visible in the current view (as determined by `obj_px`).

    The returned information is later used by GPT-based description models or
    downstream spatial reasoning modules.

    Coordinate conventions:
      - `V` and `pose` are in **world coordinates**.
      - `R_cv`, `t_cv` represent the OpenCV-style world→camera transform.
      - The camera coordinate system follows OpenCV / PyTorch3D conventions:
            X → right,  Y → down,  Z → forward (depth).

    Args:
        F (np.ndarray):         (Nf, 3) array of face indices.
        V (np.ndarray):         (Nv, 3) array of vertex positions in world coordinates.
        vert_seg (np.ndarray):  (Nv,) integer array mapping each vertex to a segment ID.
        seg_to_obj (dict):      Mapping from segment ID → object ID (instance).
        obj_to_label (dict):    Mapping from object ID → raw label string (lowercased).
        obj_px (dict):          Mapping from visible object ID → number of visible pixels.
        face_obj_ids (np.ndarray): (Nf,) integer array mapping each face to an object ID (-1 = unlabeled).
        pix_to_face (torch.Tensor): (H,W) tensor of face indices per pixel.
        pose (np.ndarray):      (4,4) camera-to-world SE(3) matrix for this frame.
        R_cv (np.ndarray):      (3,3) world→camera rotation matrix (OpenCV convention).
        t_cv (np.ndarray):      (3,) world→camera translation vector (OpenCV convention).
        fx, fy (float):         Focal lengths of the rasterization camera (pixels).
        cx, cy (float):         Principal point of the rasterization camera (pixels).
        image_width (int):      Rasterized image width (pixels).
        image_height (int):     Rasterized image height (pixels).
        fov_depth_clip (tuple): Min/max depth (m) for an object centroid to be considered visible.
        coverage_threshold (float): Minimum percent of TOTAL IMAGE pixels (0.0-1.0) for an object to be considered visible.
        min_pixel_count (int):  Minimum absolute pixel count for an object to be considered visible.

    Returns:
        Dict[int, Dict[str, object]]:
        Mapping from object ID to a metadata dictionary with fields:
            {
                "label": str,                     # object name (e.g., "chair")
                "pixel_percent": float,           # percent of image pixels
                "centroid_world": [x, y, z],      # world-space centroid (meters)
                "centroid_ndc": [ndc_x, ndc_y],   # NDC coords of centroid in image frame
                "bbox_world": [[minx, miny, minz], [maxx, maxy, maxz]],
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

        # Collect all vertices of those faces
        verts_idx = np.unique(F[obj_face_ids].reshape(-1))
        verts_obj = V[verts_idx]
        if verts_obj.size == 0:
            continue

        centroid_world = verts_obj.mean(axis=0)
        centroid_cam = R_cv @ centroid_world + t_cv
        if centroid_cam[2] <= 0:
            continue

        # Filter: in front of camera & within depth range
        if centroid_cam[2] < fov_depth_clip[0] or centroid_cam[2] > fov_depth_clip[1]:
            continue
        # Filter: centroid should project inside the image bounds (with small slack)
        z = centroid_cam[2]
        inv_z = 1.0 / z
        u = fx * (centroid_cam[0] * inv_z) + cx
        v = fy * (centroid_cam[1] * inv_z) + cy
        pad_w = 0.02 * image_width
        pad_h = 0.02 * image_height
        if (u < -pad_w) or (u > image_width + pad_w) or (v < -pad_h) or (v > image_height + pad_h):
            continue

        # Calculating the NDC coordinates of the centroid (in the Image Frame)
        ndc_x = (u / image_width) * 2 - 1
        ndc_y = 1 - (v / image_height) * 2

        dist_from_cam = float(np.linalg.norm(centroid_world - cam_pos))
        bbox_min, bbox_max = verts_obj.min(axis=0), verts_obj.max(axis=0)

        visible_objects[int(oid)] = {
            "label": obj_to_label.get(int(oid), ""),
            "pixel_percent": float(round(coverage * 100.0, 3)),
            "centroid_world": centroid_world.tolist(),
            "centroid_ndc": [ndc_x, ndc_y],
            "bbox_world": [bbox_min.tolist(), bbox_max.tolist()],
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
      - X-axis → "left_of" / "right_of"
      - Y-axis → "above" / "below"
      - Z-axis → "in_front_of" / "behind"

    The camera coordinate frame follows the OpenCV convention:
      X → right,  Y → down,  Z → forward (depth).

    Args:
        visible_objects (dict): Mapping from object ID → metadata dictionary.
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


def greedy_next_best_views(
    image_stats: List[Dict],
    max_images: int | None = None,
    min_gain_pixels: int = 0,
    alpha: float = 0.5,  # 1.0 = old behavior (coverage only)
    min_obj_pixels_for_presence: int = 100,  # pixels to count an object as "visible" (diversity)
    camera_poses: Dict[str, np.ndarray] | None = None,  # fid -> (4,4) cam2world pose
    min_position_distance: float = 0.0,  # Min position distance between views (meters)
    min_angle_distance: float = 0.0,  # Min angle distance between views (degrees)
    enable_pose_filtering: bool = False,  # Enable spatial diversity filtering
) -> List[str]:
    """
    Balanced greedy NBV with optional spatial diversity filtering:
      score = alpha * normalized_coverage_gain + (1 - alpha) * normalized_diversity

    - Coverage gain: how many *new* pixels we add toward each object's cap.
    - Diversity: how many distinct objects in this image are clearly visible
      (>= min_obj_pixels_for_presence). This reflects "descriptiveness".
    - Spatial filtering: Optionally reject views too close to already-selected ones.

    Both terms are normalized (0..1) per-iteration across remaining candidates
    so alpha meaningfully trades off the two.

    Args:
        image_stats: Per-frame visibility stats (must contain "fid" and "obj_pixels").
        max_images: Maximum number of views to select.
        min_gain_pixels: Minimum pixel gain to continue selection.
        alpha: Balance between coverage (1.0) and diversity (0.0).
        min_obj_pixels_for_presence: Min pixels to count object as present.
        camera_poses: Optional dict mapping frame id -> camera pose matrix.
        min_position_distance: Min distance (m) between selected camera positions.
        min_angle_distance: Min angle (deg) between selected viewing directions.
        enable_pose_filtering: If True, apply spatial diversity filtering.

    Returns:
        List of selected frame ids in selection order.
    """
    from src.utils.camera_utils import is_pose_too_similar

    covered: Dict[int, int] = defaultdict(int)  # covered pixels toward each object's cap
    remaining = set(range(len(image_stats)))
    selected: List[str] = []
    selected_poses: List[np.ndarray] = []  # Track poses of selected views for spatial filtering

    # Per-object caps: max contribution any single image can provide for that object.
    obj_caps: Dict[int, int] = defaultdict(int)
    for s in image_stats:
        for oid, c in s["obj_pixels"].items():
            obj_caps[oid] = max(obj_caps[oid], c)

    while remaining:
        # Compute raw terms for all remaining candidates this round
        cov_gains = {}
        diversities = {}

        for i in remaining:
            s = image_stats[i]

            # --- Coverage gain (with caps, only counts remaining headroom) ---
            gain_cov = 0
            for oid, c in s["obj_pixels"].items():
                cap = obj_caps[oid]
                if cap > covered[oid]:
                    gain_cov += min(c, cap - covered[oid])

            # --- Diversity (descriptiveness): # of objects clearly visible ---
            # Count objects with enough pixels in this image (regardless of novelty).
            gain_div = sum(1 for _, c in s["obj_pixels"].items() if c >= min_obj_pixels_for_presence)

            cov_gains[i] = gain_cov
            diversities[i] = gain_div

        if not cov_gains:
            break

        # Stop early if even the best raw coverage gain is below the floor
        best_raw_gain = max(cov_gains.values()) if cov_gains else 0
        if best_raw_gain < min_gain_pixels:
            break

        # --- Normalize terms to 0..1 so alpha is meaningful ---
        max_cov = max(cov_gains.values()) or 1
        max_div = max(diversities.values()) or 1

        best_idx = None

        # Sort candidates by score and try them in order (with pose filtering)
        scored_candidates = []
        for i in remaining:
            norm_cov = cov_gains[i] / max_cov
            norm_div = diversities[i] / max_div
            score = alpha * norm_cov + (1.0 - alpha) * norm_div
            scored_candidates.append((i, score))

        # Sort by score descending
        scored_candidates.sort(key=lambda x: x[1], reverse=True)

        # Try candidates in score order, applying pose filter if enabled
        for candidate_idx, _ in scored_candidates:
            fid = image_stats[candidate_idx]["fid"]

            # Check spatial diversity constraint if enabled
            if enable_pose_filtering and camera_poses is not None and fid in camera_poses:
                candidate_pose = camera_poses[fid]
                if is_pose_too_similar(
                    candidate_pose,
                    selected_poses,
                    min_position_distance,
                    min_angle_distance,
                ):
                    continue  # Skip this candidate, try next one

            # Accept this candidate
            best_idx = candidate_idx
            break

        if best_idx is None:
            break

        # Commit selection
        fid_selected = image_stats[best_idx]["fid"]
        selected.append(fid_selected)

        # Track pose for spatial filtering
        if enable_pose_filtering and camera_poses is not None and fid_selected in camera_poses:
            selected_poses.append(camera_poses[fid_selected])

        # Update covered toward caps using the selected frame's contributions
        for oid, c in image_stats[best_idx]["obj_pixels"].items():
            cap = obj_caps[oid]
            if covered[oid] < cap:
                covered[oid] = min(cap, covered[oid] + c)

        remaining.remove(best_idx)
        if max_images is not None and len(selected) >= max_images:
            break

    return selected


# ------------------------------ Mask rendering --------------------------------


def pix_to_instance_mask(
    pix_to_face_np: np.ndarray, face_obj_ids: np.ndarray, void_val: int = VOID_ID
) -> np.ndarray:
    """
    Convert per-pixel face index to a 16-bit instance mask.

    The value written is the objectId. Unlabeled faces map to `void_val`.

    Args:
        pix_to_face_np: (H,W) int64 array of face indices (-1 = no face).
        face_obj_ids:   (Nf,) int32 object id per face (-1 = unlabeled).
        void_val:       background value for unlabeled regions.

    Returns:
        (H,W) uint16 array with objectId per pixel (or void).
    """
    h, w = pix_to_face_np.shape
    mask = np.full((h, w), void_val, dtype=np.uint16)
    valid = pix_to_face_np >= 0
    faces = pix_to_face_np[valid]
    inst = face_obj_ids[faces]  # -1 for unlabeled
    inst[inst < 0] = void_val
    mask[valid] = inst.astype(np.uint16)
    return mask


def pix_to_semantic_mask(
    pix_to_face_np: np.ndarray,
    face_obj_ids: np.ndarray,
    obj_to_sem_id: Dict[int, int],
    void_val: int = VOID_ID,
) -> np.ndarray:
    """
    Convert per-pixel face index to a 16-bit semantic class mask.

    We map: face -> objectId -> semanticId, using obj_to_sem_id.
    Unmapped objects become `void_val`.

    Args:
        pix_to_face_np: (H,W) int64 face index per pixel.
        face_obj_ids:   (Nf,) object id per face (-1 = unlabeled).
        obj_to_sem_id:  dict {objectId: semanticId} (from TSV & aggregation).
        void_val:       background value.

    Returns:
        (H,W) uint16 semanticId mask.
    """
    h, w = pix_to_face_np.shape
    mask = np.full((h, w), void_val, dtype=np.uint16)
    valid = pix_to_face_np >= 0
    faces = pix_to_face_np[valid]
    obj_ids = face_obj_ids[faces]
    sem_vals = np.full_like(obj_ids, void_val, dtype=np.int32)
    for i in range(len(obj_ids)):
        oid = int(obj_ids[i])
        if oid >= 0:
            sem_vals[i] = int(obj_to_sem_id.get(oid, void_val))
    mask[valid] = sem_vals.astype(np.uint16)
    return mask


def save_png16(path: Path, arr_uint16: np.ndarray) -> None:
    """
    Save a single-channel 16-bit PNG (preserves large ids correctly).

    Args:
        path: output file path.
        arr_uint16: (H,W) uint16 array to save.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr_uint16, mode="I;16").save(str(path))


# ------------------------------ K-means Clustering ----------------------------


def cluster_camera_poses(
    camera_poses: List[np.ndarray],
    frame_ids: List[str],
    n_clusters: int,
    random_state: int = 42,
) -> Tuple[List[str], np.ndarray]:
    """
    Cluster camera poses using K-means and return representative frame IDs.

    Args:
        camera_poses: List of (4,4) camera-to-world pose matrices.
        frame_ids: List of frame IDs corresponding to each pose.
        n_clusters: Number of clusters to form.
        random_state: Random seed for reproducibility.

    Returns:
        cluster_representatives: List of frame IDs (one per cluster, sorted by original rank).
        cluster_labels: Array of cluster labels for each input pose.
    """
    from sklearn.cluster import KMeans

    n_candidates = len(camera_poses)

    if n_candidates <= 1 or n_clusters <= 1:
        print("[WARN] Not enough frames for meaningful clustering. Skipping clustering.")
        return frame_ids[:], np.zeros(n_candidates, dtype=int)

    # Use effective cluster count
    n_clusters = min(n_candidates, n_clusters)

    # Extract positions from poses
    positions = np.stack([pose[:3, 3] for pose in camera_poses], axis=0)

    # Run K-means
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    cluster_labels = kmeans.fit_predict(positions)
    print(f"[INFO] K-means clustering into {n_clusters} clusters done.")

    # Map frame id -> original rank (lower is better, preserves NBV order)
    rank = {fid: i for i, fid in enumerate(frame_ids)}

    # Group frames by cluster id
    clusters = {}
    for lbl, fid in zip(cluster_labels, frame_ids):
        clusters.setdefault(int(lbl), []).append(fid)

    # Pick the top-ranked (earliest in original order) frame per cluster
    cluster_representatives = []
    for lbl, fids in clusters.items():
        best_fid = min(fids, key=lambda x: rank[x])
        cluster_representatives.append(best_fid)

    # Sort chosen reps by their original rank for a nice, stable order
    cluster_representatives.sort(key=lambda x: rank[x])

    return cluster_representatives, cluster_labels
