"""
ScanNet++-style NBV selection + mask export (instance & semantic) + debug viz.

This script implements a compact pipeline for ScanNet scenes:

Pipeline
--------
1) Load one ScanNet scene: mesh (+ vertex colors), segmentation, aggregation,
   color frames, per-frame poses, and intrinsics.
2) Convert vertex-level segment ids to per-face object ids via majority vote.
3) Rasterize a *downsampled* visibility pass over a subset of frames and count,
   per image, how many pixels (front-most face only) belong to each object id.
4) Greedy "next-best-view" selection that balances coverage across instances.
5) Cache the per-image stats and the selected list; write a summary CSV; copy
   selected RGB/poses to a folder.
6) For the selected frames, re-rasterize (usually at full resolution) and save:
   - Instance mask as 16-bit PNG (pixel value = ScanNet objectId)
   - Semantic mask as 16-bit PNG (pixel value = semanticId from TSV)

Debug Visualization
-------------------
If --debug is enabled, for a few frames we show the dataset RGB (left) and
a PyTorch3D render from the exact same camera (right), using the vertex colors
stored in the *_vh_clean_2.ply mesh. This is a good sanity check that the
camera model and pose parsing line up in pixel space.

Key Design Choice
-----------------
We build the PyTorch3D camera using the official OpenCV bridge:
    cameras_from_opencv_projection(R, tvec, K, image_size=(H,W))
This removes the need for manual axis flips; you pass in OpenCV world->cam
(R, t) and the pixel intrinsics, and the bridge returns a PerspectiveCameras
that renders in *the same* pixel coordinates as your RGBs.

Author: Roham Zendehdel Nobari (rzendehdel@ethz.ch)
"""
from __future__ import annotations

import multiprocessing as mp
import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import open3d as o3d
import torch
import cv2
from PIL import Image
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend for matplotlib
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import argparse

# PyTorch3D
from pytorch3d.renderer import (
    MeshRasterizer, PerspectiveCameras, RasterizationSettings,
    MeshRenderer, HardPhongShader, PointLights
)
from pytorch3d.structures import Meshes
from pytorch3d.renderer.mesh import TexturesVertex
from pytorch3d.renderer import Textures
from pytorch3d.utils import cameras_from_opencv_projection
from brisque import BRISQUE

# Project imports
from src.utils.camera_utils import (
    load_cam2world,
    invert_se3_to_opencv,
    load_intrinsics_txt,
    compute_pose_distance,
    is_pose_too_similar,
)
from src.utils.config_loader import load_config


# -------------------------------- Constants -----------------------------------

# Background/unlabeled value written to exported masks.
VOID_ID: int = 0


# ----------------------------- I/O & Utilities --------------------------------

# Note: load_intrinsics_txt, load_cam2world, and invert_se3_to_opencv
# are now imported from src.utils.camera_utils


def load_mesh_with_vertex_colors(scene_path: Path, scene_id: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load the *_vh_clean_2.ply mesh (vertices, faces, vertex colors).

    Args:
        scene_path: Root path of the scene (e.g., .../scene0000_00).
        scene_id: Scene id string ('scene0000_00').

    Returns:
        V: (Nv, 3) float32 vertices in world units.
        F: (Nf, 3) int64 face indices.
        VC: (Nv, 3) float32 vertex colors; in [0,1] if stored that way in PLY,
            else we normalize later.
    """
    ply = scene_path / f"{scene_id}_vh_clean_2.ply"
    if not ply.exists():
        raise FileNotFoundError(f"Mesh not found: {ply}")
    mesh = o3d.io.read_triangle_mesh(str(ply))
    V = np.asarray(mesh.vertices, dtype=np.float32)
    F = np.asarray(mesh.triangles, dtype=np.int64)
    VC = np.asarray(mesh.vertex_colors, dtype=np.float32) if mesh.has_vertex_colors() else np.ones_like(V, dtype=np.float32) * 0.7
    return V, F, VC


def load_segments_and_instances(scene_path: Path, scene_id: str) -> Tuple[np.ndarray, Dict[int, int], Dict[int, str]]:
    """
    Load segmentation + aggregation JSON and build mappings.

    - segs["segIndices"] provides a per-vertex segment id (same order as PLY).
    - aggregation groups segments into objects (instances) and provides labels.

    Args:
        scene_path: Root path of the scene.
        scene_id: Scene id string.

    Returns:
        vert_seg:   (Nv,) int32 segment id per vertex.
        seg_to_obj: dict mapping segment id -> object id (instance).
        obj_to_label: dict mapping object id -> raw label string (lowercased).
    """
    segs_json = scene_path / f"{scene_id}_vh_clean_2.0.010000.segs.json"
    if not segs_json.exists():
        segs_json = scene_path / f"{scene_id}_vh_clean_2.segs.json"
    if not segs_json.exists():
        raise FileNotFoundError(f"Segs JSON not found: {segs_json}")

    agg_json = scene_path / f"{scene_id}.aggregation.json"
    if not agg_json.exists():
        raise FileNotFoundError(f"Aggregation JSON not found: {agg_json}")

    segs = json.loads(segs_json.read_text())
    agg = json.loads(agg_json.read_text())

    vert_seg = np.array(segs["segIndices"], dtype=np.int32)

    seg_to_obj: Dict[int, int] = {}
    obj_to_label: Dict[int, str] = {}
    for g in agg["segGroups"]:
        oid = int(g["objectId"])
        obj_to_label[oid] = g.get("label", "").strip().lower()
        for s in g["segments"]:
            seg_to_obj[int(s)] = oid

    return vert_seg, seg_to_obj, obj_to_label


def per_face_object_ids(F: np.ndarray, vert_seg: np.ndarray, seg_to_obj: Dict[int, int]) -> np.ndarray:
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


# ---------------------------- Semantic label map ------------------------------

def load_scannet_labelmap_tsv(tsv_path: Path, semantic_id_key: str = "nyu40id") -> Dict[str, int]:
    """
    Parse 'scannetv2-labels.combined.tsv' and map raw labels to semantic ids.

    Args:
        tsv_path: Path to the combined TSV file.
        semantic_id_key: Which column to use for semantic ids (e.g., 'nyu40id').

    Returns:
        labelmap: dict raw_label_lowercase -> semantic_id (int).
    """
    if not tsv_path.exists():
        raise FileNotFoundError(f"Label map TSV not found: {tsv_path}")

    lines = tsv_path.read_text().splitlines()
    header = [h.strip() for h in lines[0].split("\t")]
    rows = [dict(zip(header, [c.strip() for c in L.split("\t")])) for L in lines[1:]]

    # Heuristically pick a column that holds the "raw" label string (robust to TSV variants).
    name_candidates = ["raw_category", "raw_name", "raw_label", "name", "raw_category_0"]
    raw_col = next((k for k in name_candidates if k in header), None)
    if raw_col is None:
        for h in header:
            hlow = h.lower()
            if ("raw" in hlow) and ("name" in hlow or "category" in hlow):
                raw_col = h
                break
    if raw_col is None:
        raise ValueError(f"Could not find a raw label column in TSV header: {header}")

    if semantic_id_key not in header:
        raise ValueError(f"semantic_id_key='{semantic_id_key}' not in TSV header: {header}")

    labelmap: Dict[str, int] = {}
    for r in rows:
        raw = (r.get(raw_col) or "").strip().lower()
        sid = (r.get(semantic_id_key) or "").strip()
        if not raw or not sid:
            continue
        try:
            labelmap[raw] = int(float(sid))
        except ValueError:
            continue
    return labelmap


# ----------------------------- Cameras & Rasterizer ---------------------------

def make_p3d_from_opencv(
    R_cv: np.ndarray,
    t_cv: np.ndarray,
    fx: float, fy: float, cx: float, cy: float,
    H: int, W: int,
    device: torch.device
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
    K = torch.tensor([[[fx, 0.0, cx],
                       [0.0, fy, cy],
                       [0.0, 0.0, 1.0]]], dtype=torch.float32, device=device)  # (1,3,3)
    R = torch.from_numpy(R_cv).float()[None].to(device)  # (1,3,3)
    t = torch.from_numpy(t_cv).float()[None].to(device)  # (1,3)
    image_size = torch.tensor([[H, W]], dtype=torch.float32, device=device)  # (1,2) = (H,W)
    cams = cameras_from_opencv_projection(R=R, tvec=t, camera_matrix=K, image_size=image_size)
    return cams


def make_rasterizer(
    H: int, W: int,
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

_global_brisque = None
def _init_brisque_worker():
    """Create one BRISQUE instance per process and limit OpenCV threading."""
    global _global_brisque
    cv2.setNumThreads(1)   # avoid CPU oversubscription inside workers
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

def filter_sharp_images(color_dir: Path, threshold: float, workers: int | None = None) -> list[str]:
    """
    Parallel BRISQUE filtering without caching or downsampling.

    Args:
        color_dir: directory containing *.jpg frames.
        threshold: keep frames with BRISQUE <= threshold (lower is better).
        workers: number of processes; default = max(1, cpu_count()-1).

    Returns:
        Ordered list of frame ids (stems) that pass the threshold.
    """
    image_files = sorted(color_dir.glob("*.jpg"))
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
    zbuf = fragments.zbuf[0, ..., 0]                        # (H,W)
    return p2f, zbuf

def compute_image_visibility(pix_to_face: torch.Tensor, face_obj_ids: np.ndarray) -> Tuple[Dict[int, int], int]:
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
    fov_depth_clip=(0.2, 10.0),
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
        pix_to_face (torch.Tensor): (H,W) tensor of face indices per pixel (-
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
    max_distance : float = 2.0, 
    size_ratio_threshold: float = 5.0,
    eps: float = 0.1
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
        "wall": ["behind", "in_front_of"], # Walls shouldn't be "left of" furniture
        "carpet": ["below", "under"],
        "rug": ["below", "under"]
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
            delta = ca - cb # Vector from B to A

            axis = np.argmax(np.abs(delta))
            relation = None
            
            # OpenCV Coords: X (Right), Y (Down), Z (Forward)
            if axis == 0:  # X-axis
                if delta[0] > eps: relation = "right_of" # A is right of B
                elif delta[0] < -eps: relation = "left_of" # A is left of B
            elif axis == 1:  # Y-axis
                if delta[1] > eps: relation = "below"    # A is below B (Y increases downwards)
                elif delta[1] < -eps: relation = "above" # A is above B
            elif axis == 2:  # Z-axis
                if delta[2] > eps: relation = "behind" # A is further (higher Z)
                elif delta[2] < -eps: relation = "in_front_of" # A is closer (lower Z)

            if not relation:
                continue

            # --- FILTER 3: Semantic Validity ---
            label_a = obj_a.get("label", "").lower()
            label_b = obj_b.get("label", "").lower()

            # Check if Subject (A) allows this relation. If not in rules, all relations allowed.
            if label_a in SEMANTIC_RULES:
                if relation not in SEMANTIC_RULES[label_a]:
                    continue

            spatial_relations.append({
                "subject": label_a or f"id_{id_a}",
                "object": label_b or f"id_{id_b}",
                "relation": relation,
                "distance": float(dist_world) # Helpful for debugging
            })

    return spatial_relations


# Note: compute_pose_distance and is_pose_too_similar
# are now imported from src.utils.camera_utils


def greedy_next_best_views(
    image_stats: List[Dict],
    max_images: int | None = None,
    min_gain_pixels: int = 0,
    alpha: float = 0.5,                      # 1.0 = old behavior (coverage only)
    min_obj_pixels_for_presence: int = 100,  # pixels to count an object as "visible" (diversity)
    camera_poses: Dict[str, np.ndarray] | None = None,  # fid -> (4,4) cam2world pose
    min_position_distance: float = 0.0,      # Min position distance between views (meters)
    min_angle_distance: float = 0.0,         # Min angle distance between views (degrees)
    enable_pose_filtering: bool = False,     # Enable spatial diversity filtering
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
    covered: Dict[int, int] = defaultdict(int)     # covered pixels toward each object's cap
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
            gain_div = sum(
                1 for _, c in s["obj_pixels"].items()
                if c >= min_obj_pixels_for_presence
            )

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

def pix_to_instance_mask(pix_to_face_np: np.ndarray, face_obj_ids: np.ndarray, void_val: int = VOID_ID) -> np.ndarray:
    """
    Convert per-pixel face index to a 16-bit instance mask.

    The value written is ScanNet's objectId. Unlabeled faces map to `void_val`.

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


# -------------------------- Debug visualization helper ------------------------

@torch.no_grad()
def render_rgb_with_camera(V, F, VC, cameras: PerspectiveCameras, H: int, W: int, device):
    """
    Render the mesh with vertex colors from the given camera.

    This is used for quick side-by-side debugging (RGB vs render).

    Args:
        V, F, VC: mesh vertices, faces, vertex colors.
        cameras: PerspectiveCameras built from the OpenCV bridge.
        H, W: image size in pixels.
        device: CUDA/CPU device.

    Returns:
        A float32 RGB image in [0,1] as np.ndarray with shape (H, W, 3).
    """
    verts = torch.from_numpy(V).float().to(device)
    faces = torch.from_numpy(F).long().to(device)

    vc = VC.astype(np.float32)
    if vc.max() > 1.0:
        vc = vc / 255.0  # normalize if colors were stored 0..255
    verts_rgb = torch.from_numpy(vc).float().unsqueeze(0).to(device)

    mesh = Meshes(verts=[verts], faces=[faces], textures=TexturesVertex(verts_features=verts_rgb))
    rasterizer = make_rasterizer(H, W, faces_per_pixel=1)

    # Ambient-only lighting: color == vertex color (diffuse/specular off).
    lights = PointLights(
        device=device,
        ambient_color=((1.0, 1.0, 1.0),),
        diffuse_color=((0.0, 0.0, 0.0),),
        specular_color=((0.0, 0.0, 0.0),),
        location=[[0.0, 0.0, 0.0]],
    )
    shader = HardPhongShader(device=device, cameras=cameras, lights=lights)
    renderer = MeshRenderer(rasterizer=rasterizer, shader=shader)

    img = renderer(mesh, cameras=cameras)[0, ..., :3].clamp(0, 1).cpu().numpy()
    return img


def show_side_by_side(fid: str, rgb_path: Path, rendered: np.ndarray):
    """
    Utility to display dataset RGB (left) vs. PyTorch3D render (right).

    Args:
        fid: frame id string for the title.
        rgb_path: path to the RGB image file.
        rendered: rendered RGB image (H,W,3) in [0,1].
    """
    rgb = np.array(Image.open(rgb_path))
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1); plt.imshow(rgb); plt.title(f"RGB {fid}"); plt.axis("off")
    plt.subplot(1, 2, 2); plt.imshow(rendered); plt.title("PyTorch3D render"); plt.axis("off")
    plt.tight_layout(); plt.show()

def plot_clusters(camera_poses: List[np.ndarray], cluster_labels: List[int]):
    """
    Plot the camera poses in 3D with color mapping for each cluster.
    """
    # Extract positions (translation vectors) from the camera poses
    positions = np.stack([pose[:3, 3] for pose in camera_poses], axis=0)  # (N,3)

    # Plot the clusters
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], c=cluster_labels, cmap='jet', s=50)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Clustered Camera Poses')

    plt.colorbar(scatter)
    plt.show()

# ----------------------------------- Main -------------------------------------

def main(scene_id: str, config_path: str, device_str: str | None = None,
         debug: bool = False, viz_first: int = 3, viz_selected: int = 0, auto_clean: bool = False, save_semantic_masks: bool = False, save_instance_masks: bool = False) -> None:
    """
    End-to-end pipeline orchestrator.

    It loads all assets, builds caches, performs NBV selection, copies selected
    frames, optionally visualizes, and finally exports instance/semantic masks.

    Args:
        scene_id: ScanNet scene id, e.g., 'scene0000_00'.
        config_path: path to a YAML with:
            paths.dataset_path: root folder containing ScanNet scenes.
            scannetpp.*: pipeline knobs (see code).
        device_str: override torch device string ('cuda:0', 'cpu', ...).
        debug: show RGB vs render for a few frames (sanity check).
        viz_first: how many initial frames to visualize.
        viz_selected: how many top selected frames to visualize.
    """
    cfg = load_config(config_path)

    # Paths
    dataset_path = Path(cfg["paths"]["scannet_dataset_path"])
    scan_path = dataset_path / scene_id
    intrinsic_path = scan_path / "intrinsic" / "intrinsic_color.txt"
    color_dir = scan_path / "color"
    pose_dir = scan_path / "pose"
    depth_dir = scan_path / "depth"
    label_dir = scan_path / "label"
    output_dir = scan_path / "output"
    sens_file = scan_path / f"{scene_id}.sens"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Rasterization / selection knobs
    spp = cfg.get("scannetpp", {})
    downsample = int(spp.get("image_downsample_factor", 2))
    subsample_factor = int(spp.get("subsample_factor", 5))
    faces_per_pixel = int(spp.get("faces_per_pixel", 1))
    bin_size = spp.get("bin_size", None)
    max_faces_per_bin = spp.get("max_faces_per_bin", None)
    blur_radius = float(spp.get("blur_radius", 0.0))
    limit_images = spp.get("limit_images", None)
    max_best = spp.get("max_best", None)
    min_gain_pixels = int(spp.get("min_gain_pixels", 0))
    kmeans_n_clusters = int(spp.get("kmeans_n_clusters", 10))
    imq_threshold = float(spp.get("imq_threshold", 40.0))

    # Object visibility thresholds
    coverage_threshold = float(spp.get("coverage_threshold", 0.05))
    min_pixel_count = int(spp.get("min_pixel_count", 50))
    min_obj_pixels_for_presence = int(spp.get("min_obj_pixels_for_presence", 100))

    # FOV and depth settings
    fov_depth_clip_min = float(spp.get("fov_depth_clip_min", 0.2))
    fov_depth_clip_max = float(spp.get("fov_depth_clip_max", 10.0))

    # NBV algorithm parameters
    nbv_alpha = float(spp.get("nbv_alpha", 0.5))
    nbv_min_position_distance = float(spp.get("nbv_min_position_distance", 0.0))
    nbv_min_angle_distance = float(spp.get("nbv_min_angle_distance", 0.0))
    nbv_enable_pose_filtering = bool(spp.get("nbv_enable_pose_filtering", False))

    # Spatial relations parameters
    spatial_max_distance = float(spp.get("spatial_max_distance", 2.0))
    spatial_size_ratio_threshold = float(spp.get("spatial_size_ratio_threshold", 5.0))
    spatial_eps = float(spp.get("spatial_eps", 0.1))

    # Mask export knobs
    mask_ds = int(spp.get("mask_downsample_factor", 1))
    semantic_id_key = str(spp.get("semantic_id_key", "nyu40id"))
    labelmap_tsv = Path(spp.get("labelmap_tsv", "data/scannetv2-labels.combined.tsv"))

    # Output / cache
    cache_dir = output_dir / Path(spp.get("cache_dir"))
    out_dir = output_dir / Path(spp.get("raster_out_dir"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Device
    device = torch.device(device_str) if device_str else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # Collect frames
    frame_ids = filter_sharp_images(color_dir, threshold=imq_threshold)
    if subsample_factor > 1:
        frame_ids = frame_ids[::subsample_factor]
    if limit_images is not None:
        frame_ids = frame_ids[:int(limit_images)]
    if not frame_ids:
        # progressively relax threshold up to a cap
        relax_seq = [imq_threshold + 10, 45, 50, 60]
        for thr in relax_seq:
            frame_ids = filter_sharp_images(color_dir, threshold=thr)
            if frame_ids:
                print(f"[WARN] Relaxed BRISQUE threshold to {thr}; kept {len(frame_ids)} frames.")
                break

    if not frame_ids:
        raise RuntimeError(f"No sharp images found in {color_dir} even after relaxing threshold.")
    if debug:
        print(f"[DEBUG] Using {len(frame_ids)} sharp frames after subsample/limit.")

    # Intrinsics
    fx, fy, cx, cy = load_intrinsics_txt(intrinsic_path)
    if debug:
        print(f"[DEBUG] Intrinsics: fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")

    # Mesh (+ vertex colors) and labels
    V, F, VC = load_mesh_with_vertex_colors(scan_path, scene_id)
    if debug:
        print(f"[DEBUG] Mesh: {V.shape[0]} vertices, {F.shape[0]} faces; colors: {VC.shape}")
    vert_seg, seg_to_obj, obj_to_label = load_segments_and_instances(scan_path, scene_id)
    face_obj_ids = per_face_object_ids(F, vert_seg, seg_to_obj)
    unique_face_objs = np.unique(face_obj_ids[face_obj_ids >= 0])
    if debug:
        print(f"[DEBUG] face_obj_ids: {len(unique_face_objs)} unique objects (non-void).")

    # Semantic label map (objectId -> semanticId)
    labelmap = load_scannet_labelmap_tsv(labelmap_tsv, semantic_id_key=semantic_id_key)
    obj_to_sem_id: Dict[int, int] = {oid: int(labelmap.get(lbl, VOID_ID)) for oid, lbl in obj_to_label.items()}

    # PyTorch3D mesh for visibility
    verts = torch.from_numpy(V).float().to(device)
    faces = torch.from_numpy(F).long().to(device)
    textures_white = Textures(verts_rgb=torch.ones_like(verts)[None, ...])  # visibility only
    meshes = Meshes(verts=[verts], faces=[faces], textures=textures_white)

    # Base image sizes
    sample_img = np.array(Image.open(color_dir / f"{frame_ids[0]}.jpg"))
    H0, W0 = map(int, sample_img.shape[:2])

    # ------------------------------- DEBUG VIZ --------------------------------
    if debug and viz_first > 0:
        print(f"[DEBUG] Visualizing first {viz_first} frame(s) (RGB vs PyTorch3D render)...")
        for fid in frame_ids[:viz_first]:
            pose = load_cam2world(pose_dir / f"{fid}.txt")
            R_cv, t_cv = invert_se3_to_opencv(pose)  # world->cam (OpenCV)

            # Full-res camera matches the RGB size.
            cams_dbg = make_p3d_from_opencv(R_cv, t_cv, fx, fy, cx, cy, H0, W0, device)

            rendered = render_rgb_with_camera(V, F, VC, cams_dbg, H0, W0, device)
            show_side_by_side(fid, color_dir / f"{fid}.jpg", rendered)

    # ----------------------- VISIBILITY (DOWNSAMPLED) -------------------------
    H_vis = max(1, H0 // max(1, downsample))
    W_vis = max(1, W0 // max(1, downsample))
    fx_vis, fy_vis, cx_vis, cy_vis = fx / downsample, fy / downsample, cx / downsample, cy / downsample

    rasterizer_vis = make_rasterizer(
        H_vis, W_vis,
        faces_per_pixel=faces_per_pixel,
        bin_size=bin_size,
        max_faces_per_bin=max_faces_per_bin,
        blur_radius=blur_radius,
    )

    cache_json = cache_dir / f"{scene_id}.json"
    if cache_json.exists():
        print(f"[INFO] Loading visibility cache: {cache_json}")
        image_stats = json.loads(cache_json.read_text())
    else:
        print(f"[INFO] Computing visibility for {len(frame_ids)} frames at {H_vis}x{W_vis}...")
        image_stats: List[Dict] = []

        for idx, fid in enumerate(tqdm(frame_ids)):
            pose = load_cam2world(pose_dir / f"{fid}.txt")
            R_cv, t_cv = invert_se3_to_opencv(pose)

            cams_vis = make_p3d_from_opencv(
                R_cv, t_cv, fx_vis, fy_vis, cx_vis, cy_vis, H_vis, W_vis, device
            )
            pix_to_face, _ = rasterize_visibility(meshes, cams_vis, rasterizer_vis)
            obj_px, total_px = compute_image_visibility(pix_to_face, face_obj_ids)

            visible_objects = compute_visible_objects(
                V, F, vert_seg, seg_to_obj, obj_to_label,
                obj_px, face_obj_ids, pix_to_face,
                pose, R_cv, t_cv,
                fx_vis, fy_vis, cx_vis, cy_vis,
                W_vis, H_vis,
                fov_depth_clip=(fov_depth_clip_min, fov_depth_clip_max),
                coverage_threshold=coverage_threshold,
                min_pixel_count=min_pixel_count,
            )
            spatial_relations = compute_spatial_relations(
                visible_objects,
                max_distance=spatial_max_distance,
                size_ratio_threshold=spatial_size_ratio_threshold,
                eps=spatial_eps,
            )

            image_entry = {
                "fid": fid,
                "obj_pixels": obj_px,
                "total_labeled_px": int(total_px),
                "visible_objects": visible_objects,
                "spatial_relations": spatial_relations,
            }
            image_stats.append(image_entry)

            if idx < 5 and debug:
                print(f"[DEBUG] Frame {fid}: {len(visible_objects)} objs, {len(spatial_relations)} relations")

        cache_json.write_text(json.dumps(image_stats, indent=2))
        print(f"[INFO] Saved visibility cache: {cache_json}")

    # ------------------------ GREEDY NEXT-BEST-VIEWS --------------------------
    # Load camera poses for spatial filtering (if enabled)
    camera_poses_dict = None
    if nbv_enable_pose_filtering:
        print("[INFO] Loading camera poses for spatial diversity filtering...")
        camera_poses_dict = {}
        for stat in image_stats:
            fid = stat["fid"]
            pose_path = pose_dir / f"{fid}.txt"
            if pose_path.exists():
                camera_poses_dict[fid] = load_cam2world(pose_path)
        print(f"[INFO] Loaded {len(camera_poses_dict)} camera poses.")

    order_cache = cache_dir / f"{scene_id}.pth"
    if order_cache.exists():
        print(f"[INFO] Loading best-views order: {order_cache}")
        best_views: List[str] = torch.load(order_cache)
    else:
        print("[INFO] Computing greedy next-best views...")
        best_views = greedy_next_best_views(
            image_stats,
            max_images=max_best,
            min_gain_pixels=min_gain_pixels,
            alpha=nbv_alpha,
            min_obj_pixels_for_presence=min_obj_pixels_for_presence,
            camera_poses=camera_poses_dict,
            min_position_distance=nbv_min_position_distance,
            min_angle_distance=nbv_min_angle_distance,
            enable_pose_filtering=nbv_enable_pose_filtering,
        )
        torch.save(best_views, order_cache)
        print(f"[INFO] Saved best-views order: {order_cache}")

   # --------------------------- K-means Clustering -----------------------------
    print("[INFO] Applying K-means clustering to selected camera poses...")

    # Build pose list and positions for only the NBV-ordered frames
    camera_poses = []
    camera_ids = []
    for fid in best_views:
        pose = load_cam2world(pose_dir / f"{fid}.txt")
        camera_poses.append(pose)
        camera_ids.append(fid)

    # Positions = translation vector from 4x4 cam2world
    positions = np.stack([pose[:3, 3] for pose in camera_poses], axis=0)  # (N,3)

    # Choose number of clusters robustly
    n_candidates = len(camera_ids)
    if isinstance(kmeans_n_clusters, int) and kmeans_n_clusters > 0:
        n_clusters = min(n_candidates, kmeans_n_clusters)
    else:
        # fallback: about one per ~12 views, clamped
        n_clusters = max(6, min(40, int(round(n_candidates / 12)))) if n_candidates >= 6 else n_candidates

    if n_candidates <= 1 or n_clusters <= 1:
        print("[WARNING] Not enough frames for meaningful clustering. Skipping clustering.")
        cluster_labels = np.zeros(n_candidates, dtype=int)
    else:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
        cluster_labels = kmeans.fit_predict(positions)
        print(f"[INFO] K-means clustering into {n_clusters} clusters done.")

    # Map frame id -> NBV rank (lower is better)
    rank = {fid: i for i, fid in enumerate(best_views)}

    # Group frames by cluster id
    clusters = {}
    for lbl, fid in zip(cluster_labels, camera_ids):
        clusters.setdefault(int(lbl), []).append(fid)

    # Pick the top-ranked (earliest in NBV order) frame per cluster
    cluster_representatives = []
    for lbl, fids in clusters.items():
        best_fid = min(fids, key=lambda x: rank[x])
        cluster_representatives.append(best_fid)

    # (Optional) sort chosen reps by their NBV rank for a nice, stable order
    cluster_representatives.sort(key=lambda x: rank[x])

    print(f"[INFO] Selected {len(cluster_representatives)} cluster representatives.")

    # --------------------------- Save selected frames ----------------------------
    color_output_dir = output_dir / "color"
    pose_output_dir = output_dir / "pose"
    depth_output_dir = output_dir / "depth"
    label_output_dir = output_dir / "label"
    for d in (color_output_dir, pose_output_dir, depth_output_dir, label_output_dir):
        d.mkdir(parents=True, exist_ok=True)

    camera_pose_json = {}
    for fid in cluster_representatives:
        pose = load_cam2world(pose_dir / f"{fid}.txt")
        camera_pose_json[fid] = pose.tolist()

        shutil.copy(color_dir / f"{fid}.jpg",  color_output_dir / f"{fid}.jpg")
        shutil.copy(depth_dir / f"{fid}.png",  depth_output_dir / f"{fid}.png")
        shutil.copy(pose_dir  / f"{fid}.txt",  pose_output_dir / f"{fid}.txt")
        lp = label_dir / f"{fid}.png"
        if lp.exists():
            shutil.copy(lp, label_output_dir / f"{fid}.png")
        print(f"[INFO] Saved cluster-representative frame {fid} to {output_dir}")

    with open(output_dir / "camera_pose.json", "w") as f:
        json.dump(camera_pose_json, f, indent=2)

    # --------------------------- (Optional) Debug plot ---------------------------
    if debug:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Plot all candidates, colored by cluster
        for lbl in np.unique(cluster_labels):
            idx = np.where(cluster_labels == lbl)[0]
            ax.scatter(positions[idx, 0], positions[idx, 1], positions[idx, 2], label=f"C{int(lbl)}", s=30)

        # Highlight chosen representatives (larger markers)
        rep_idx = [camera_ids.index(fid) for fid in cluster_representatives]
        ax.scatter(positions[rep_idx, 0], positions[rep_idx, 1], positions[rep_idx, 2], s=120, marker='*')

        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
        ax.set_title('Clustered Camera Poses (+ reps)')
        ax.legend()
        plt.show()

    # print("\n[DONE] Best views (first 50):")
    # for i, fid in enumerate(best_views[:50], start=1):
    #     print(f"{i:>3d}. {fid}")

    # --------------------------- Export instance/semantic masks -------------------
    if save_instance_masks or save_semantic_masks:
        print("[INFO] Rendering masks for selected frames...")
        # Build image size / intrinsics for mask resolution
        Hm = max(1, H0 // max(1, mask_ds))
        Wm = max(1, W0 // max(1, mask_ds))
        fx_m, fy_m, cx_m, cy_m = fx / mask_ds, fy / mask_ds, cx / mask_ds, cy / mask_ds

        rasterizer_mask = make_rasterizer(
            Hm, Wm,
            faces_per_pixel=1,
            bin_size=bin_size,
            max_faces_per_bin=max_faces_per_bin,
            blur_radius=0.0,
        )

        inst_dir = output_dir / "instance"
        sem_dir  = output_dir / "semantic"
        if save_instance_masks: inst_dir.mkdir(parents=True, exist_ok=True)
        if save_semantic_masks: sem_dir.mkdir(parents=True, exist_ok=True)

        # numpy copy of face->object ids for fast indexing
        face_obj_ids_np = np.asarray(face_obj_ids, dtype=np.int32)

        for fid in tqdm(cluster_representatives, desc="Masks", dynamic_ncols=True):
            pose = load_cam2world(pose_dir / f"{fid}.txt")
            R_cv, t_cv = invert_se3_to_opencv(pose)

            cams_mask = make_p3d_from_opencv(R_cv, t_cv, fx_m, fy_m, cx_m, cy_m, Hm, Wm, device)
            pix_to_face, _ = rasterize_visibility(meshes, cams_mask, rasterizer_mask)
            p2f_np = pix_to_face.cpu().numpy()

            if save_instance_masks:
                inst = pix_to_instance_mask(p2f_np, face_obj_ids_np, void_val=VOID_ID)
                save_png16(inst_dir / f"{fid}.png", inst)

            if save_semantic_masks:
                sem = pix_to_semantic_mask(p2f_np, face_obj_ids_np, obj_to_sem_id, void_val=VOID_ID)
                save_png16(sem_dir / f"{fid}.png", sem)

    # Auto-clean if enabled
    if auto_clean:
        print("\n[INFO] Auto-clean enabled. Deleting intermediate files...")
        if sens_file.exists():
            sens_file.unlink()
            print(f"Deleted: {sens_file}")
        for folder in [color_dir, depth_dir, pose_dir, label_dir]:
            if folder.exists():
                shutil.rmtree(folder)
                print(f"Deleted folder: {folder}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ScanNet++ NBV + masks + debug viz")
    parser.add_argument("scene_id", type=str, help="Scene ID (e.g., scene0000_00)")
    parser.add_argument("--config", type=str, default="config/default.yaml", help="Path to config YAML")
    parser.add_argument("--device", type=str, default=None, help="torch device (e.g., cuda:0 or cpu)")
    parser.add_argument("--debug", action="store_true", help="Debug mode: Print extra info and visualize some preprocessing steps")
    parser.add_argument("--viz_first", type=int, default=3, help="Visualize this many initial frames")
    parser.add_argument("--viz_selected", type=int, default=0, help="Visualize this many top selected frames")
    parser.add_argument("--auto_clean", action="store_true", help="Auto-clean intermediate files (color/depth/pose/label dirs and .sens file)")
    parser.add_argument("--save_semantic_masks", action="store_true", help="Save semantic masks (16-bit PNGs)")
    parser.add_argument("--save_instance_masks", action="store_true", help="Save instance masks (16-bit PNGs)")
    args = parser.parse_args()

    main(
        args.scene_id,
        args.config,
        device_str=args.device,
        debug=args.debug,
        viz_first=args.viz_first,
        viz_selected=args.viz_selected,
        auto_clean=args.auto_clean,
        save_semantic_masks=args.save_semantic_masks,
        save_instance_masks=args.save_instance_masks,
    )
