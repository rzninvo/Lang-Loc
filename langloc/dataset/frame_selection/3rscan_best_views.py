#!/usr/bin/env python3
"""
3RScan-style NBV selection + mask export (instance & semantic) + debug viz.

Pipeline:
1) Load 3RScan scene: mesh (+ vertex colors from OBJ+MTL+PNG), segmentation,
   semseg.json groups, color frames, poses, and intrinsics.
2) Filter frames with IQA (Qualiclip) & subsample (with fallback thresholds).
3) Rasterize visibility for candidate frames; compute per-object pixel counts.
4) Greedy NBV selection + adaptive K-means clustering for diversity.
5) Save representative frames + poses + masks (instance/semantic).
6) (Optional) Auto-clean raw frame files.

Author: Roham Zendehdel Nobari (rzendehdel@ethz.ch)
"""

import hydra
from omegaconf import DictConfig
import json
import shutil
from pathlib import Path
import numpy as np
import open3d as o3d
import torch
from PIL import Image
from tqdm import tqdm
import plyfile
from sklearn.neighbors import NearestNeighbors

from pytorch3d.structures import Meshes
from pytorch3d.renderer.mesh import TexturesUV
from pytorch3d.renderer import (
    MeshRenderer, MeshRasterizer, HardPhongShader,
    PointLights, RasterizationSettings
)
from pytorch3d.utils import cameras_from_opencv_projection

# Project imports - modular frame selection components
from langloc.dataset.frame_selection.iqa import filter_quality_images
from langloc.dataset.frame_selection.visibility import (
    VOID_ID,
    make_p3d_camera_from_opencv,
    make_rasterizer,
    per_face_object_ids,
    precompute_object_geometry,
    rasterize_visibility,
    compute_image_visibility,
    compute_visible_objects,
    compute_spatial_relations,
    depth_consistency_mask,
    compute_depth_consistent_counts,
    save_depth_debug_panel,
)
from langloc.dataset.frame_selection.dpp import (
    compute_face_normals,
    compute_clip_embeddings,
    dpp_select_views,
)
from langloc.dataset.frame_selection.legacy import greedy_next_best_views, cluster_camera_poses
from langloc.dataset.frame_selection.masks import pix_to_instance_mask, pix_to_semantic_mask, save_png16
from langloc.dataset.scene_graph_builder import build_scene_graph, add_embeddings_to_scene_graph, save_scene_graph
from langloc.utils.camera_utils import (
    invert_se3_to_opencv,
    load_cam2world,
    load_intrinsics_info,
)
from langloc.utils.nbv_config import NBVConfig, extract_nbv_config
import matplotlib.pyplot as plt


# ----------------------------- Loaders ---------------------------------


def load_scan_id_set(path: Path | None) -> set[str]:
    """
    Load newline-separated scan IDs from a text file.

    Returns an empty set if the path is unset, missing, or not a file.
    """
    if path is None:
        return set()
    if not path.exists():
        return set()
    if not path.is_file():
        print(f"[WARN] Expected partial scans file, got non-file path: {path}")
        return set()

    ids = set()
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # Support inline comments.
        if "#" in line:
            line = line.split("#", 1)[0].strip()
        if line:
            ids.add(line)
    return ids

def load_mesh_with_textures(scene_path, device):
    """
    Load the refined 3RScan mesh together with its texture map and return a
    PyTorch3D `Meshes` instance ready for rendering.

    The OBJ uses shared vertices for UV seams; to keep texture coordinates
    consistent with PyTorch3D we duplicate vertices per triangle and return the
    index mapping so auxiliary per-vertex data (e.g., segmentation) can be
    expanded in the same way.

    Args:
        scene_path (Path | str): Directory containing the 3RScan assets.
        device (torch.device): Device on which the mesh tensors should live.

    Returns:
        tuple[Meshes, np.ndarray, np.ndarray]:
            - meshes:  Single-item `Meshes` with duplicated vertices and UVs.
            - orig_vertex_idx: 1D array mapping each expanded vertex back to the
              original OBJ vertex index (length = 3 * num_faces).
            - verts: Original vertex positions from the OBJ (N, 3), kept so the
              caller can build per-vertex annotations without duplication.
    """

    obj_path = scene_path / "mesh.refined.v2.obj"
    tex_path = scene_path / "mesh.refined_0.png"
    mesh = o3d.io.read_triangle_mesh(str(obj_path), True)

    uvs = np.asarray(mesh.triangle_uvs)

    verts = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    verts_uvs = uvs
    faces_uvs = np.arange(len(verts_uvs)).reshape(-1, 3)

    tex_img = np.array(Image.open(tex_path).convert("RGB")).astype(np.float32) / 255.0
    tex_img = tex_img ** 2.2

    # Expand verts to match UVs (the key step)
    orig_vertex_idx = faces.reshape(-1)
    expanded_verts = verts[orig_vertex_idx]
    expanded_faces = np.arange(len(expanded_verts)).reshape(-1, 3)

    meshes = Meshes(
        verts=[torch.tensor(expanded_verts, dtype=torch.float32, device=device)],
        faces=[torch.tensor(expanded_faces, dtype=torch.int64, device=device)],
        textures=TexturesUV(
            maps=torch.tensor(tex_img, dtype=torch.float32, device=device).unsqueeze(0),
            faces_uvs=[torch.tensor(faces_uvs, dtype=torch.int64, device=device)],
            verts_uvs=[torch.tensor(verts_uvs, dtype=torch.float32, device=device)],
        ),
    )
    return meshes, orig_vertex_idx, verts


def load_segments_and_instances(scene_path: Path, base_vertices: np.ndarray):
    """
    Transfer per-vertex instance IDs and semantic labels onto the mesh.

    The official `mesh.refined.0.010000.segs.v2.json` is aligned with a decimated
    mesh, so we instead use the annotated point cloud
    `labels.instances.annotated.v2.ply` and find the nearest annotated point for
    every vertex of the textured mesh.

    Args:
        scene_path (Path): 3RScan scene directory.
        base_vertices (np.ndarray): (N, 3) array of the original (non-duplicated)
            OBJ vertices.

    Returns:
        tuple[np.ndarray, dict[int, int], dict[int, str]]:
            - vert_obj: per-vertex instance IDs aligned with `base_vertices`.
            - seg_to_obj: identity mapping retained for API compatibility
              (`segment` -> `objectId`).
            - obj_to_label: objectId -> semantic label (lowercased).
    """
    semseg_json = scene_path / "semseg.v2.json"
    ply_path = scene_path / "labels.instances.annotated.v2.ply"
    if not semseg_json.exists() or not ply_path.exists():
        raise FileNotFoundError("Missing semantic annotations or annotated point cloud.")

    groups = json.loads(semseg_json.read_text())["segGroups"]
    obj_to_label = {int(g["objectId"]): g.get("label", "").strip().lower() for g in groups}

    ply = plyfile.PlyData.read(ply_path)
    pts = np.vstack([ply["vertex"][axis] for axis in ("x", "y", "z")]).T.astype(np.float32)
    obj_ids = np.asarray(ply["vertex"]["objectId"], dtype=np.int32)

    nn = NearestNeighbors(n_neighbors=1, algorithm="kd_tree")
    nn.fit(pts)
    _, idx = nn.kneighbors(base_vertices.astype(np.float32), return_distance=True)
    vert_obj = obj_ids[idx[:, 0]]

    seg_to_obj = {int(oid): int(oid) for oid in np.unique(vert_obj) if oid >= 0}
    return vert_obj.astype(np.int32), seg_to_obj, obj_to_label


def load_object_geometry_from_semseg(scene_path: Path):
    """
    Load per-object geometry from 3RScan `semseg.v2.json` OBB fields.

    The file provides, per object:
      - centroid
      - axis lengths
      - normalized axes (3x3)

    We convert this into the geometry structure expected by
    `compute_visible_objects(...)`:
      centroid_world, bbox_min/max, obb_center/axes/extents.
    """
    semseg_json = scene_path / "semseg.v2.json"
    if not semseg_json.exists():
        return {}

    groups = json.loads(semseg_json.read_text()).get("segGroups", [])
    geometry = {}

    # 8 corners of a box in local OBB coordinates.
    signs = np.array(
        [[sx, sy, sz] for sx in (-1.0, 1.0) for sy in (-1.0, 1.0) for sz in (-1.0, 1.0)],
        dtype=np.float64,
    )

    for g in groups:
        oid = int(g.get("objectId", -1))
        if oid < 0:
            continue

        obb = g.get("obb") or {}
        try:
            center = np.asarray(obb["centroid"], dtype=np.float64).reshape(3)
            axes_lengths = np.asarray(obb["axesLengths"], dtype=np.float64).reshape(3)
            axes = np.asarray(obb["normalizedAxes"], dtype=np.float64).reshape(3, 3)
        except Exception:
            continue

        # Normalize row-wise to ensure unit axis vectors.
        norms = np.linalg.norm(axes, axis=1, keepdims=True)
        axes = axes / np.maximum(norms, 1e-12)
        extents = np.maximum(axes_lengths / 2.0, 1e-8)

        # rows in `axes` are interpreted as world-space axis vectors.
        local_corners = signs * extents[None, :]
        corners_world = center[None, :] + local_corners @ axes
        bbox_min = corners_world.min(axis=0)
        bbox_max = corners_world.max(axis=0)

        geometry[oid] = {
            "centroid_world": center,
            "bbox_min": bbox_min,
            "bbox_max": bbox_max,
            "obb_center": center,
            "obb_axes": axes,
            "obb_extents": extents,
        }

    return geometry


def load_object_attributes(scene_id: str, dataset_path: Path):
    """
    Load per-object metadata for a specific 3RScan scene from `objects.json`.

    Returns:
        Dict[int, Dict[str, object]] mapping `objectId -> merged metadata`.
        Includes nested semantic attributes plus selected top-level fields:
        `ply_color`, `nyu40`, `eigen13`, `rio27`, `global_id`, `id`,
        `label`, and `affordances`.
    """
    objects_json = dataset_path / "objects.json"
    if not objects_json.exists():
        print(f"[WARN] 3RScan object metadata not found: {objects_json}")
        return {}

    try:
        payload = json.loads(objects_json.read_text())
    except Exception as exc:
        print(f"[WARN] Failed to parse {objects_json}: {exc}")
        return {}

    scans = payload.get("scans", [])
    scan_entry = next((s for s in scans if s.get("scan") == scene_id), None)
    if scan_entry is None:
        print(f"[WARN] Scene {scene_id} not found in {objects_json.name}.")
        return {}

    attributes_by_oid = {}
    for obj in scan_entry.get("objects", []):
        raw_id = obj.get("id")
        if raw_id is None:
            continue
        try:
            oid = int(raw_id)
        except (TypeError, ValueError):
            continue
        attrs_raw = obj.get("attributes", {})
        merged = dict(attrs_raw) if isinstance(attrs_raw, dict) else {}

        for key in ("ply_color", "nyu40", "eigen13", "rio27", "global_id", "id", "label"):
            value = obj.get(key)
            if value is None:
                continue
            if isinstance(value, str):
                value = value.strip()
                if not value:
                    continue
            merged[key] = value

        affordances = obj.get("affordances")
        if isinstance(affordances, list):
            merged["affordances"] = affordances

        attributes_by_oid[oid] = merged

    return attributes_by_oid


def build_semantic_id_mapping(
    obj_to_label: dict,
    object_attributes: dict,
    semantic_id_key: str = "global_id",
    void_id: int = 0,
    object_ids: list[int] | None = None,
) -> dict:
    """
    Build a canonical ``objectId -> semanticId`` mapping for 3RScan.

    Instead of scene-local enumeration (1, 2, 3, …), this maps each object
    to its cross-dataset semantic class id using a field from ``objects.json``
    (typically ``global_id``).

    Args:
        obj_to_label: Mapping ``objectId -> label`` (from segmentation).
        object_attributes: Per-object metadata from :func:`load_object_attributes`.
        semantic_id_key: Which attribute to use as the semantic id
            (``"global_id"``, ``"nyu40"``, ``"rio27"``, ``"eigen13"``).
        void_id: Value for objects without a valid semantic mapping.
        object_ids: Optional explicit object id list to map (e.g., all mesh
            object ids). If omitted, keys from ``obj_to_label`` are used.

    Returns:
        Dict[int, int] mapping ``objectId -> semanticId``.
    """
    obj_to_sem: dict = {}
    n_mapped = 0
    n_missing = 0
    missing_oids = []

    all_oids: set[int] = set(int(oid) for oid in obj_to_label.keys())
    if object_ids is not None:
        all_oids.update(int(oid) for oid in object_ids)

    for oid in sorted(all_oids):
        attrs = object_attributes.get(oid, {})
        raw_val = attrs.get(semantic_id_key)
        if raw_val is not None:
            try:
                sem_id = int(raw_val)
                obj_to_sem[oid] = sem_id
                n_mapped += 1
                continue
            except (TypeError, ValueError):
                pass
        # Fallback: no valid mapping
        obj_to_sem[oid] = void_id
        n_missing += 1
        missing_oids.append(oid)

    if n_missing > 0:
        labels = []
        for oid in missing_oids[:10]:
            label = obj_to_label.get(oid)
            if not label:
                label = object_attributes.get(oid, {}).get("label", "?")
            labels.append(label)
        print(
            f"[WARN] Semantic mapping ({semantic_id_key}): "
            f"{n_mapped} mapped, {n_missing} missing → VOID. "
            f"Missing sample: {dict(zip(missing_oids[:10], labels))}"
        )
    else:
        print(
            f"[INFO] Semantic mapping ({semantic_id_key}): "
            f"all {n_mapped} objects mapped successfully."
        )

    return obj_to_sem


# ----------------------- GT Relationship Helpers ------------------------------

def load_gt_relationships(scene_id: str, dataset_path: Path) -> list[list] | None:
    """
    Load ground-truth relationships for a 3RScan scene.

    Reads ``relationships.json`` (3DSSG format) and returns the raw
    relationship triples for the requested scene.

    Args:
        scene_id: UUID of the 3RScan scene.
        dataset_path: Root of the 3RScan dataset (e.g. ``data/3RScan/``).

    Returns:
        List of ``[subject_id, object_id, predicate_id, predicate_str]``
        entries, or ``None`` if the scene is not found in the file.
    """
    rel_path = dataset_path / "relationships.json"
    if not rel_path.exists():
        print(f"[WARN] GT relationships file not found: {rel_path}")
        return None

    # Use a module-level cache to avoid re-reading for batch runs.
    if not hasattr(load_gt_relationships, "_cache"):
        load_gt_relationships._cache = {}

    if rel_path not in load_gt_relationships._cache:
        try:
            payload = json.loads(rel_path.read_text())
        except Exception as exc:
            print(f"[WARN] Failed to parse {rel_path}: {exc}")
            return None
        idx = {s["scan"]: s for s in payload.get("scans", [])}
        load_gt_relationships._cache[rel_path] = idx

    idx = load_gt_relationships._cache[rel_path]
    entry = idx.get(scene_id)
    if entry is None:
        return None

    return entry.get("relationships", [])


def build_fcl_collision_objects(
    verts_np: np.ndarray,
    faces_np: np.ndarray,
    face_obj_ids: np.ndarray,
) -> dict:
    """
    Build per-object FCL ``CollisionObject`` instances with BVH geometry.

    Each object's triangles are extracted from the global mesh and compiled
    into a BVH for fast distance and collision queries.  Call this once per
    scene and reuse across all frames.

    Args:
        verts_np: (V, 3) global mesh vertices.
        faces_np: (F, 3) global mesh face indices.
        face_obj_ids: (F,) object id per face (-1 = unlabeled).

    Returns:
        Dict mapping ``objectId -> fcl.CollisionObject``.
    """
    import fcl

    face_obj_ids = np.asarray(face_obj_ids)
    v = np.ascontiguousarray(verts_np, dtype=np.float64)
    objects: dict = {}

    for oid in np.unique(face_obj_ids):
        if oid < 0:
            continue
        obj_faces = np.ascontiguousarray(
            faces_np[face_obj_ids == oid], dtype=np.int32,
        )
        if obj_faces.shape[0] == 0:
            continue

        model = fcl.BVHModel()
        model.beginModel(v.shape[0], obj_faces.shape[0])
        model.addSubModel(v, obj_faces)
        model.endModel()
        objects[int(oid)] = fcl.CollisionObject(model)

    return objects


def compute_surface_distance(
    fcl_objects: dict,
    oid_a: int,
    oid_b: int,
) -> float:
    """
    Exact minimum surface-to-surface distance between two object meshes.

    Uses FCL's BVH-accelerated distance query with an upfront collision
    check (intersecting meshes return 0.0).

    Args:
        fcl_objects: Dict from :func:`build_fcl_collision_objects`.
        oid_a: Subject object id.
        oid_b: Object object id.

    Returns:
        Minimum surface distance in meters, or -1.0 if either object
        has no FCL geometry.
    """
    import fcl

    obj_a = fcl_objects.get(oid_a)
    obj_b = fcl_objects.get(oid_b)
    if obj_a is None or obj_b is None:
        return -1.0

    # Collision check — intersecting meshes have distance 0.
    creq = fcl.CollisionRequest(num_max_contacts=1, enable_contact=True)
    cres = fcl.CollisionResult()
    if fcl.collide(obj_a, obj_b, creq, cres) > 0:
        return 0.0

    dreq = fcl.DistanceRequest(enable_nearest_points=False)
    dres = fcl.DistanceResult()
    dist = fcl.distance(obj_a, obj_b, dreq, dres)
    return max(float(dist), 0.0)


def build_gt_spatial_relations(
    gt_relationships: list[list],
    visible_objects: dict[int, dict],
    obj_to_label: dict[int, str],
    object_geometry: dict[int, dict],
    fcl_objects: dict | None = None,
    max_surface_distance: float | None = None,
) -> list[dict]:
    """
    Build per-frame spatial relations from scene-level GT relationships.

    Only edges where **both** subject and object are visible in the current
    frame are kept.  Each edge is enriched with centroid distance and
    exact mesh surface-to-surface distance (via FCL BVH).

    View-dependent directional predicates (left/right/front/behind, IDs 2-5)
    are re-projected from scene/world coordinates into the camera frame using
    ``centroid_cam`` from ``visible_objects``.  The OpenCV convention is used:
    X = right, Y = down, Z = forward.

    Args:
        gt_relationships: Raw GT triples
            ``[[sub_id, obj_id, pred_id, pred_str], ...]``.
        visible_objects: Per-frame visible objects (keyed by int objectId).
            Must include ``centroid_cam`` for view-dependent correction.
        obj_to_label: Global objectId -> label mapping.
        object_geometry: Per-object geometry with ``centroid_world``.
        fcl_objects: Per-object FCL CollisionObjects from
            :func:`build_fcl_collision_objects`.  If ``None``, surface
            distance is reported as -1.
        max_surface_distance: Prune relations whose ``surface_min_distance``
            exceeds this threshold (meters).  ``None`` disables pruning.

    Returns:
        List of relation dicts sorted by ``surface_min_distance`` ascending.
    """
    # Predicate IDs for view-dependent directional relations (3DSSG convention)
    _VIEW_DEP_PRED_IDS = {2, 3, 4, 5}  # left, right, front, behind
    _DIR_EPS = 0.1  # min displacement (m) to call a directional relation

    visible_ids = set(int(oid) for oid in visible_objects.keys())
    relations: list[dict] = []
    # Remove exact duplicates only — keyed by ordered (src, dst, pred_id).
    # Both directions (A→B and B→A) are kept since downstream GNN uses
    # directed edges and needs messages in both directions.
    _seen_edges: set[tuple] = set()

    for triple in gt_relationships:
        if len(triple) < 4:
            continue
        sub_id, obj_id, pred_id, pred_str = (
            int(triple[0]), int(triple[1]), int(triple[2]), str(triple[3]),
        )
        if sub_id not in visible_ids or obj_id not in visible_ids:
            continue
        if sub_id == obj_id:
            continue

        sub_label = obj_to_label.get(sub_id, f"id_{sub_id}")
        obj_label = obj_to_label.get(obj_id, f"id_{obj_id}")

        # Re-project view-dependent predicates into camera frame
        view_corrected = False
        if pred_id in _VIEW_DEP_PRED_IDS:
            sub_meta = visible_objects.get(sub_id) or visible_objects.get(str(sub_id))
            obj_meta = visible_objects.get(obj_id) or visible_objects.get(str(obj_id))
            if sub_meta is not None and obj_meta is not None:
                ca = np.asarray(sub_meta["centroid_cam"], dtype=np.float64)
                cb = np.asarray(obj_meta["centroid_cam"], dtype=np.float64)
                delta = ca - cb  # vector from object → subject

                # Dominant horizontal axis: X (left/right) vs Z (front/behind)
                abs_dx, abs_dz = abs(delta[0]), abs(delta[2])
                if abs_dx >= abs_dz:
                    # X-dominant: left / right
                    if abs_dx < _DIR_EPS:
                        continue  # too close — drop relation
                    pred_str = "right" if delta[0] > 0 else "left"
                else:
                    # Z-dominant: front / behind
                    if abs_dz < _DIR_EPS:
                        continue
                    pred_str = "behind" if delta[2] > 0 else "front"
                view_corrected = True

        # Exact-duplicate check (ordered: A→B ≠ B→A).
        # For view-corrected predicates, key by corrected pred_str so that
        # different original pred_ids (e.g. 2="left" and 3="right") that
        # resolve to the same camera-frame direction are deduplicated.
        edge_key = (sub_id, obj_id, pred_str) if view_corrected else (sub_id, obj_id, pred_id)
        if edge_key in _seen_edges:
            continue
        _seen_edges.add(edge_key)

        # Centroid distance
        geom_s = object_geometry.get(sub_id)
        geom_o = object_geometry.get(obj_id)
        if geom_s is not None and geom_o is not None:
            cs = np.asarray(geom_s["centroid_world"], dtype=np.float64)
            co = np.asarray(geom_o["centroid_world"], dtype=np.float64)
            centroid_dist = float(np.linalg.norm(cs - co))
        else:
            centroid_dist = -1.0

        # Exact mesh surface distance (FCL BVH)
        if fcl_objects is not None:
            surface_dist = compute_surface_distance(fcl_objects, sub_id, obj_id)
        else:
            surface_dist = -1.0

        rel_dict = {
            "subject": sub_label,
            "object": obj_label,
            "relation": pred_str,
            "centroid_distance": centroid_dist,
            "surface_min_distance": surface_dist,
            "subject_id": sub_id,
            "object_id": obj_id,
            "predicate_id": pred_id,
            "source": "gt",
        }
        if view_corrected:
            rel_dict["view_corrected"] = True
        relations.append(rel_dict)

    # Prune distant relations by surface distance.
    if max_surface_distance is not None:
        relations = [
            r for r in relations
            if 0 <= r["surface_min_distance"] <= max_surface_distance
        ]

    # Closest relations first (downstream takes first N).
    relations.sort(key=lambda r: r["surface_min_distance"] if r["surface_min_distance"] >= 0 else 1e9)
    return relations


# -------------------------- Debug Visualization Helpers -----------------------

@torch.no_grad()
def _render_textured_rgb(meshes, cameras, H, W, device):
    """
    Render a textured 3RScan mesh using PyTorch3D at full resolution.

    Lighting is kept ambient-only to match the baked texture as closely as
    possible. The function is primarily used for debug side-by-side comparisons.

    Args:
        meshes (Meshes): PyTorch3D mesh batch (single mesh expected).
        cameras (PerspectiveCameras): Camera aligned with the RGB frame.
        H, W (int): Output resolution.
        device (torch.device): Device to run the renderer on.

    Returns:
        np.ndarray: (H, W, 3) float image in [0, 1].
    """
    raster_settings = RasterizationSettings(
        image_size=(H, W),
        faces_per_pixel=1,
        blur_radius=0.0,
        perspective_correct=True,
        cull_backfaces=True,
    )

    rasterizer = MeshRasterizer(raster_settings=raster_settings)
    lights = PointLights(
        device=device,
        ambient_color=((1.0, 1.0, 1.0),),
        diffuse_color=((0.0, 0.0, 0.0),),
        specular_color=((0.0, 0.0, 0.0),),
        location=[[0.0, 0.0, 0.0]],
    )
    shader = HardPhongShader(device=device, cameras=cameras, lights=lights)
    renderer = MeshRenderer(rasterizer=rasterizer, shader=shader)

    img = renderer(meshes, cameras=cameras)[0, ..., :3].clamp(0, 1).cpu().numpy()
    return img

def _save_side_by_side(rgb_path: Path, rendered_rgb: np.ndarray, title_left: str, title_right: str, out_path: Path):
    """
    Save a diagnostic figure comparing dataset RGB and PyTorch3D render.

    Args:
        rgb_path (Path): Path to the original RGB frame.
        rendered_rgb (np.ndarray): Rendered view in [0, 1].
        title_left (str): Title for the dataset image.
        title_right (str): Title for the rendered image.
        out_path (Path): Destination PNG path.
    """
    rgb = np.array(Image.open(rgb_path))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1); plt.imshow(rgb); plt.title(title_left); plt.axis("off")
    plt.subplot(1, 2, 2); plt.imshow(rendered_rgb); plt.title(title_right); plt.axis("off")
    plt.tight_layout(); plt.savefig(out_path, dpi=100); plt.close()

def _save_overlay(rgb_path: Path, pix_to_face: torch.Tensor, out_path: Path):
    """
    Overlay the rasterized silhouette on top of the RGB frame for alignment checks.

    Args:
        rgb_path (Path): Path to the RGB frame to use as the background.
        pix_to_face (torch.Tensor): Rasterizer output with face indices.
        out_path (Path): Destination PNG path for the overlay.
    """
    rgb = np.array(Image.open(rgb_path))
    vis = pix_to_face.detach().cpu().numpy()

    # ---- normalize dimensions ----
    if vis.ndim == 4:          # (N,H,W,K)
        vis = vis[0, ..., 0]
    elif vis.ndim == 3:        # (N,H,W)
        vis = vis[0]
    elif vis.ndim == 2 and vis.shape[0] == 1:
        vis = vis.reshape(rgb.shape[0], rgb.shape[1])
    elif vis.ndim == 1:
        vis = vis.reshape(rgb.shape[0], rgb.shape[1])

    # ---- orientation correction ----
    if vis.shape != rgb.shape[:2]:
        if vis.shape[::-1] == rgb.shape[:2]:
            vis = vis.T
        else:
            from cv2 import resize, INTER_NEAREST
            vis = resize(vis, (rgb.shape[1], rgb.shape[0]), interpolation=INTER_NEAREST)

    vis = (vis >= 0).astype(np.uint8)

    overlay = rgb.copy()
    overlay[vis > 0] = (0.3 * overlay[vis > 0] + 0.7 * np.array([0, 255, 0])).astype(np.uint8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(overlay).save(out_path)

# ------------------------------- Main ----------------------------------

def main(scene_id: str, cfg: DictConfig, device_str=None,
         debug=False, auto_clean=False,
         save_semantic_masks=False, save_instance_masks=False,
         allow_partial: bool = False):
    """
    Entry point for selecting representative 3RScan frames and exporting assets.

    Steps performed:
        1. Load mesh, annotations, and camera intrinsics/poses.
        2. Filter frames by BRISQUE quality and optional subsampling.
        3. Rasterize downsampled visibility to compute per-object coverage.
        4. Run greedy next-best-view selection and K-means clustering.
        5. Copy RGB/Depth/Pose files for the selected frames.
        6. Optionally export semantic/instance masks and render debug visualizations.

    Args:
        scene_id (str): UUID of the 3RScan scene.
        cfg: Hydra DictConfig with ``dataset`` and ``paths`` groups.
        device_str (str | None): Optional override for the compute device
            (e.g., "cuda:0" or "cpu"). Defaults to CUDA if available.
        debug (bool): If True, generate PyTorch3D vs RGB comparisons.
        auto_clean (bool): If True, removes raw frames once outputs are saved.
        save_semantic_masks (bool): Export semantic 16-bit PNG masks if True.
        save_instance_masks (bool): Export instance 16-bit PNG masks if True.
        allow_partial (bool): If True, process scenes listed in the partial
            scan blocklist.
    """

    nbv_cfg = extract_nbv_config(cfg.dataset, dataset="3rscan")
    partial_path_raw = cfg.paths.get("rscan_partial_scans")
    partial_file = Path(partial_path_raw) if partial_path_raw else None
    partial_ids = load_scan_id_set(partial_file)
    if partial_file is not None and scene_id in partial_ids and not allow_partial:
        print(
            f"[WARN] Scene {scene_id} is listed in partial scans ({partial_file}). "
            "Skipping 3RScan processing for this scene."
        )
        return

    dataset_path = Path(cfg.paths.rscan_root)
    scan_path = dataset_path / scene_id

    output_dir = scan_path / nbv_cfg.output_folder
    cache_dir = output_dir / "cache"
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    object_attributes_cache = load_object_attributes(scene_id, dataset_path)
    if object_attributes_cache:
        print(f"[INFO] Loaded attributes for {len(object_attributes_cache)} objects from objects.json.")

    # -------------------------- Device Setup ----------------------------
    device = torch.device(device_str or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    print(f"[INFO] Using device: {device}")

    # ---------------------- Load Mesh + Labels --------------------------
    meshes, orig_vertex_idx, original_verts = load_mesh_with_textures(scan_path, device)

    faces = meshes.faces_packed().cpu().numpy()
    verts = meshes.verts_packed().cpu().numpy()

    vert_seg_raw, seg_to_obj, obj_to_label = load_segments_and_instances(scan_path, original_verts)
    max_required_idx = int(orig_vertex_idx.max())
    if max_required_idx >= len(vert_seg_raw):
        pad = max_required_idx + 1 - len(vert_seg_raw)
        print(f"[WARN] Padding vert_seg with {pad} void entries to cover expanded vertices.")
        vert_seg_raw = np.pad(vert_seg_raw, (0, pad), constant_values=-1)
    vert_seg = vert_seg_raw[orig_vertex_idx]

    faces_np = meshes.faces_packed().cpu().numpy()
    verts_np = meshes.verts_packed().cpu().numpy()

    max_idx = vert_seg.shape[0]
    if (faces_np >= max_idx).any():
        raise ValueError("Vertex segmentation array misaligned with expanded mesh.")

    faces = faces_np
    face_obj_ids = per_face_object_ids(faces, vert_seg, seg_to_obj)

    mesh_oids = sorted(int(oid) for oid in np.unique(face_obj_ids[face_obj_ids >= 0]))
    obj_to_sem_id = build_semantic_id_mapping(
        obj_to_label,
        object_attributes_cache,
        semantic_id_key=nbv_cfg.semantic_id_key,
        void_id=VOID_ID,
        object_ids=mesh_oids,
    )
    missing_sem_oids = [oid for oid in mesh_oids if int(obj_to_sem_id.get(oid, VOID_ID)) == VOID_ID]
    print(
        f"[INFO] 3RScan semantic mapping coverage ({nbv_cfg.semantic_id_key}): "
        f"{len(mesh_oids) - len(missing_sem_oids)}/{len(mesh_oids)} mesh objects mapped to non-VOID ids."
    )
    if missing_sem_oids:
        sample = {
            oid: (obj_to_label.get(oid) or object_attributes_cache.get(oid, {}).get("label", ""))
            for oid in missing_sem_oids[:10]
        }
        print(f"[WARN] Objects mapped to VOID in semantic masks (sample): {sample}")

    fx, fy, cx, cy = load_intrinsics_info(scan_path / "_info.txt")

    # ------------------------- Collect Frames ---------------------------
    frame_ids = [p.name.replace(".color.jpg", "") for p in sorted(scan_path.glob("frame-*.color.jpg"))]
    if not frame_ids:
        raise RuntimeError("No frames found")
    sample_img = np.array(Image.open(scan_path / f"{frame_ids[0]}.color.jpg"))
    H0, W0 = map(int, sample_img.shape[:2])

    # IQA filtering with 3RScan file pattern (*.color.jpg)
    iqa_metric = nbv_cfg.iqa_metric
    iqa_threshold = nbv_cfg.iqa_threshold
    iqa_device = nbv_cfg.iqa_device

    frame_ids = filter_quality_images(
        scan_path,
        metric_name=iqa_metric,
        threshold=iqa_threshold,
        file_pattern="*.color.jpg",
        device=iqa_device,
    )

    if not frame_ids:
        raise RuntimeError("No quality images found.")

    frame_ids = [fid.replace(".color", "") for fid in frame_ids]
    if nbv_cfg.subsample_factor > 1:
        frame_ids = frame_ids[::nbv_cfg.subsample_factor]
    if nbv_cfg.limit_images is not None:
        frame_ids = frame_ids[:int(nbv_cfg.limit_images)]

    print(f"[INFO] Using {len(frame_ids)} candidate frames after filtering and subsampling.")

    # --------------------------- Debug Visualization ---------------------------
    if debug:
        debug_dir = output_dir / "debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        viz_n = min(5, len(frame_ids))  # visualize first few frames
        print(f"[DEBUG] Rendering {viz_n} debug comparisons to {debug_dir}")

        for fid in frame_ids[:viz_n]:
            rgb_path = scan_path / f"{fid}.color.jpg"
            pose = load_cam2world(scan_path / f"{fid}.pose.txt")
            R_cv, t_cv = invert_se3_to_opencv(pose)

            cams_full = make_p3d_camera_from_opencv(R_cv, t_cv, fx, fy, cx, cy, H0, W0, device)

            rendered = _render_textured_rgb(meshes, cams_full, H0, W0, device)

            sxs_path = debug_dir / f"{fid}_side_by_side.png"
            _save_side_by_side(rgb_path, rendered, f"RGB {fid}", "PyTorch3D render", sxs_path)

            # optional silhouette overlay to check pixel alignment
            rasterizer_full = make_rasterizer(H0, W0, faces_per_pixel=1)
            pix_to_face_full, _ = rasterize_visibility(meshes, cams_full, rasterizer_full)
            ov_path = debug_dir / f"{fid}_overlay.png"
            _save_overlay(rgb_path, pix_to_face_full, ov_path)

        print(f"[DEBUG] Saved debug renders to {debug_dir}")


    # ---------------------- Visibility Pass (Cached) ---------------------
    downsample = nbv_cfg.image_downsample_factor
    H_vis = max(1, H0 // max(1, downsample))
    W_vis = max(1, W0 // max(1, downsample))
    fx_vis, fy_vis, cx_vis, cy_vis = fx / downsample, fy / downsample, cx / downsample, cy / downsample

    cache_json = cache_dir / f"{scene_id}.json"
    mask_cache_npz = cache_dir / f"{scene_id}_masks.npz"
    if cache_json.exists():
        print(f"[INFO] Loading cached visibility stats: {cache_json}")
        image_stats = json.loads(cache_json.read_text())
        # Check for DPP-required fields
        if nbv_cfg.dpp_enabled and image_stats and "visible_face_ids" not in image_stats[0]:
            print("[WARN] Cache missing visible_face_ids (needed for DPP). Recomputing...")
            cache_json.unlink()
            image_stats = None
        elif image_stats:
            sample_obj_meta = None
            for entry in image_stats[:10]:
                vo = entry.get("visible_objects", {})
                if vo:
                    sample_obj_meta = next(iter(vo.values()))
                    break
            if sample_obj_meta is not None and "obb_world" not in sample_obj_meta:
                print("[WARN] Cache missing obb_world/full-object geometry. Recomputing...")
                cache_json.unlink()
                image_stats = None
            elif object_attributes_cache:
                missing_attributes = False
                for entry in image_stats[:20]:
                    visible_objects = entry.get("visible_objects", {})
                    for oid_raw, obj_meta in visible_objects.items():
                        try:
                            oid = int(oid_raw)
                        except (TypeError, ValueError):
                            continue
                        if oid in object_attributes_cache:
                            attrs = obj_meta.get("attributes")
                            if not isinstance(attrs, dict):
                                missing_attributes = True
                                break
                            # Older caches may have only partial attrs; force refresh.
                            if "ply_color" not in attrs:
                                missing_attributes = True
                                break
                    if missing_attributes:
                        break
                if missing_attributes:
                    print("[WARN] Cache missing/partial object attributes from objects.json. Recomputing...")
                    cache_json.unlink()
                    image_stats = None
        # Check for depth consistency fields
        if image_stats is not None and nbv_cfg.depth_visibility_enabled:
            sample_obj_meta_dc = None
            for entry in image_stats[:10]:
                vo = entry.get("visible_objects", {})
                if vo:
                    sample_obj_meta_dc = next(iter(vo.values()))
                    break
            if sample_obj_meta_dc is not None and "depth_consistent_pixels" not in sample_obj_meta_dc:
                print("[WARN] Cache missing depth_consistent_pixels. Recomputing...")
                cache_json.unlink()
                image_stats = None
        # Check for GT spatial relations in cache
        if image_stats is not None:
            sample_rels = None
            for entry in image_stats[:10]:
                sr = entry.get("spatial_relations", [])
                if sr:
                    sample_rels = sr[0]
                    break
            if sample_rels is not None and sample_rels.get("source") != "gt":
                print("[WARN] Cache contains heuristic spatial relations. Recomputing with GT...")
                cache_json.unlink()
                image_stats = None
            elif sample_rels is not None and "surface_min_distance" not in sample_rels:
                print("[WARN] Cache has GT relations but missing surface_min_distance. Recomputing...")
                cache_json.unlink()
                image_stats = None
        # Check for view-corrected directional predicates
        if image_stats is not None:
            for entry in image_stats[:20]:
                for rel in entry.get("spatial_relations", []):
                    if rel.get("source") == "gt" and rel.get("predicate_id") in (2, 3, 4, 5):
                        if "view_corrected" not in rel:
                            print("[WARN] Cache has GT directional relations without view correction. Recomputing...")
                            cache_json.unlink()
                            image_stats = None
                        break
                if image_stats is None:
                    break
        # Check for DPP mask cache
        if image_stats is not None and nbv_cfg.dpp_enabled and not mask_cache_npz.exists():
            print("[WARN] Cache missing visibility masks (needed for DPP IoU). Recomputing...")
            cache_json.unlink()
            image_stats = None
    else:
        image_stats = None

    visibility_masks = []

    if image_stats is None:
        print(f"[INFO] Computing visibility for {len(frame_ids)} frames at {H_vis}x{W_vis}...")
        rasterizer_vis = make_rasterizer(H_vis, W_vis,
                                         faces_per_pixel=nbv_cfg.faces_per_pixel,
                                         bin_size=nbv_cfg.bin_size,
                                         max_faces_per_bin=nbv_cfg.max_faces_per_bin,
                                         blur_radius=nbv_cfg.blur_radius,
                                         )
        image_stats = []
        object_geometry_cache = load_object_geometry_from_semseg(scan_path)
        if not object_geometry_cache:
            print("[WARN] semseg.v2.json OBB geometry missing/unusable. Falling back to mesh-derived geometry.")
            object_geometry_cache = precompute_object_geometry(verts, faces, face_obj_ids)
        else:
            mesh_object_ids = set(int(oid) for oid in np.unique(face_obj_ids[face_obj_ids >= 0]))
            missing_geom = sorted(mesh_object_ids - set(object_geometry_cache.keys()))
            if missing_geom:
                print(
                    f"[WARN] semseg geometry missing {len(missing_geom)} object(s). "
                    "Filling missing entries from mesh geometry."
                )
                fallback_geom = precompute_object_geometry(verts, faces, face_obj_ids)
                for oid in missing_geom:
                    if oid in fallback_geom:
                        object_geometry_cache[oid] = fallback_geom[oid]

        gt_relationships = load_gt_relationships(scene_id, dataset_path)
        if gt_relationships is not None:
            print(f"[INFO] Loaded {len(gt_relationships)} GT relationships for scene {scene_id}.")
            print("[INFO] Building FCL BVH collision objects for surface distance...")
            fcl_objects = build_fcl_collision_objects(verts_np, faces, face_obj_ids)
            print(f"[INFO] Built FCL objects for {len(fcl_objects)} mesh objects.")
        else:
            print("[WARN] No GT relationships found. Using heuristic spatial relations.")
            fcl_objects = None

        for _idx, fid in enumerate(tqdm(frame_ids, desc="Visibility", dynamic_ncols=True)):
            pose = load_cam2world(scan_path / f"{fid}.pose.txt")
            R_cv, t_cv = invert_se3_to_opencv(pose)

            cams = make_p3d_camera_from_opencv(
                R_cv, t_cv, fx_vis, fy_vis, cx_vis, cy_vis, H_vis, W_vis, device
            )
            pix_to_face, zbuf = rasterize_visibility(meshes, cams, rasterizer_vis)
            obj_px, total_px = compute_image_visibility(pix_to_face, face_obj_ids)

            # Depth-aware visibility (optional)
            dc_counts = None
            if nbv_cfg.depth_visibility_enabled:
                depth_path = scan_path / f"{fid}.depth.pgm"
                if depth_path.exists():
                    sensor_img = Image.open(depth_path)
                    if (sensor_img.height, sensor_img.width) != (H_vis, W_vis):
                        sensor_img = sensor_img.resize((W_vis, H_vis), Image.NEAREST)
                    sensor_depth_m = np.array(sensor_img).astype(np.float32) / 1000.0
                    d_mask = depth_consistency_mask(
                        zbuf, sensor_depth_m,
                        vis_thres=nbv_cfg.depth_vis_threshold,
                    )
                    dc_counts = compute_depth_consistent_counts(
                        pix_to_face, face_obj_ids, d_mask,
                    )

            filter_reasons: dict[int, str] = {}
            visible_objects = compute_visible_objects(
                verts, faces, vert_seg, seg_to_obj, obj_to_label,
                obj_px, face_obj_ids, pix_to_face,
                pose, R_cv, t_cv,
                fx_vis, fy_vis, cx_vis, cy_vis,
                W_vis, H_vis,
                fov_depth_clip=nbv_cfg.fov_depth_clip,
                coverage_threshold=nbv_cfg.coverage_threshold,
                min_pixel_count=nbv_cfg.min_pixel_count,
                full_object_geometry=object_geometry_cache,
                object_attributes=object_attributes_cache,
                depth_consistent_counts=dc_counts,
                depth_consistent_ratio_threshold=nbv_cfg.depth_consistent_ratio_threshold,
                min_depth_consistent_pixels=nbv_cfg.min_depth_consistent_pixels,
                filtered_reasons=filter_reasons,
            )
            if gt_relationships is not None:
                spatial_relations = build_gt_spatial_relations(
                    gt_relationships,
                    visible_objects,
                    obj_to_label,
                    object_geometry_cache,
                    fcl_objects=fcl_objects,
                    max_surface_distance=nbv_cfg.spatial_max_surface_distance,
                )
            else:
                spatial_relations = compute_spatial_relations(
                    visible_objects,
                    max_distance=nbv_cfg.spatial_max_distance,
                    size_ratio_threshold=nbv_cfg.spatial_size_ratio_threshold,
                    eps=nbv_cfg.spatial_eps,
                )

            # Extract unique visible face indices for DPP normal/novelty computation
            p2f_np = pix_to_face.cpu().numpy()
            visible_face_ids = np.unique(p2f_np[p2f_np >= 0]).tolist()

            # Build labeled-object binary mask for pixel IoU (DPP Stage 2)
            mask = np.zeros(p2f_np.shape, dtype=np.uint8)
            valid = p2f_np >= 0
            mask[valid] = (face_obj_ids[p2f_np[valid]] >= 0).astype(np.uint8)
            visibility_masks.append(mask)

            image_entry = {
                "fid": fid,
                "obj_pixels": obj_px,
                "total_labeled_px": int(total_px),
                "visible_objects": visible_objects,
                "spatial_relations": spatial_relations,
                "visible_face_ids": visible_face_ids,
            }
            image_stats.append(image_entry)

            if _idx < 5 and debug and dc_counts is not None:
                debug_dir = output_dir / "debug"
                debug_dir.mkdir(parents=True, exist_ok=True)
                save_depth_debug_panel(
                    rgb_path=scan_path / f"{fid}.color.jpg",
                    sensor_depth=sensor_depth_m,
                    zbuf=zbuf,
                    depth_mask=d_mask,
                    pix_to_face=pix_to_face,
                    face_obj_ids=face_obj_ids,
                    visible_objects=visible_objects,
                    out_path=debug_dir / f"{fid}_depth_debug.png",
                    fid=fid,
                    vis_thres=nbv_cfg.depth_vis_threshold,
                    obj_px=obj_px,
                    obj_to_label=obj_to_label,
                    filtered_reasons=filter_reasons,
                )
                print(f"[DEBUG] Saved depth panel: {debug_dir / f'{fid}_depth_debug.png'}")

        cache_json.write_text(json.dumps(image_stats, indent=2))
        print(f"[INFO] Saved visibility cache: {cache_json}")

        # Cache masks for DPP
        np.savez_compressed(mask_cache_npz, masks=np.stack(visibility_masks, axis=0))
        print(f"[INFO] Saved mask cache: {mask_cache_npz}")

    elif nbv_cfg.dpp_enabled:
        # Load cached masks
        masks_data = np.load(mask_cache_npz)
        masks_array = masks_data["masks"]
        visibility_masks = [masks_array[i] for i in range(masks_array.shape[0])]
        print(f"[INFO] Loaded {len(visibility_masks)} cached visibility masks.")

    # ---------------------- VIEW SELECTION --------------------------------
    if nbv_cfg.dpp_enabled:
        # ======================== DPP VIEW SELECTION ========================
        print("[INFO] Using 2-stage DPP view selection...")

        # 1. Precompute face normals (once per scene)
        face_normals = compute_face_normals(verts_np, faces)

        # 2. Compute CLIP embeddings for all candidate frames
        clip_image_paths = [scan_path / f"{stat['fid']}.color.jpg" for stat in image_stats]
        print(f"[INFO] Computing CLIP embeddings for {len(clip_image_paths)} frames...")
        clip_embeddings = compute_clip_embeddings(
            clip_image_paths,
            model_name=nbv_cfg.dpp_clip_model,
            device=nbv_cfg.dpp_clip_device,
        )

        # 3. Load camera poses for Stage 2 spatial constraints
        print("[INFO] Loading camera poses for spatial DPP stage...")
        camera_poses_dict = {}
        for stat in image_stats:
            fid = stat["fid"]
            pose_path = scan_path / f"{fid}.pose.txt"
            if pose_path.exists():
                camera_poses_dict[fid] = load_cam2world(pose_path)
        print(f"[INFO] Loaded {len(camera_poses_dict)} camera poses.")

        # 4. Run 2-stage DPP selection
        cluster_reps = dpp_select_views(
            image_stats,
            verts_np, faces, face_normals, face_obj_ids, clip_embeddings,
            visibility_masks,
            total_views=nbv_cfg.dpp_total_views,
            seed_size=nbv_cfg.dpp_seed_size,
            camera_poses=camera_poses_dict,
            stage1_total_views=nbv_cfg.dpp_stage1_total_views,
            stage2_sigma_position=nbv_cfg.dpp_stage2_sigma_position,
            stage2_sigma_angle=nbv_cfg.dpp_stage2_sigma_angle,
            stage2_iou_gamma=nbv_cfg.dpp_stage2_iou_gamma,
        )
        print(f"[INFO] DPP selected {len(cluster_reps)} views.")

        if not cluster_reps:
            raise RuntimeError("No views selected by DPP — check config or input data.")

    else:
        # ======================== LEGACY: GREEDY + K-MEANS ========================
        # Load camera poses for spatial filtering (if enabled)
        camera_poses_dict = None
        if nbv_cfg.nbv_enable_pose_filtering:
            print("[INFO] Loading camera poses for spatial diversity filtering...")
            camera_poses_dict = {}
            for stat in image_stats:
                fid = stat["fid"]
                pose_path = scan_path / f"{fid}.pose.txt"
                if pose_path.exists():
                    camera_poses_dict[fid] = load_cam2world(pose_path)
            print(f"[INFO] Loaded {len(camera_poses_dict)} camera poses.")

        order_cache = cache_dir / f"{scene_id}.pth"
        if order_cache.exists():
            print(f"[INFO] Loading cached NBV order: {order_cache}")
            best_views = torch.load(order_cache)
        else:
            print("[INFO] Computing greedy next-best views...")
            best_views = greedy_next_best_views(
                image_stats,
                max_images=nbv_cfg.max_best,
                min_gain_pixels=nbv_cfg.min_gain_pixels,
                alpha=nbv_cfg.nbv_alpha,
                min_obj_pixels_for_presence=nbv_cfg.min_obj_pixels_for_presence,
                camera_poses=camera_poses_dict,
                min_position_distance=nbv_cfg.nbv_min_position_distance,
                min_angle_distance=nbv_cfg.nbv_min_angle_distance,
                enable_pose_filtering=nbv_cfg.nbv_enable_pose_filtering,
            )
            torch.save(best_views, order_cache)
            print(f"[INFO] Saved NBV order: {order_cache}")

        if not best_views:
            raise RuntimeError("No best views selected — check BRISQUE threshold or config.")

        # Adaptive K-means Clustering
        print("[INFO] Applying adaptive K-means clustering to selected camera poses...")
        camera_poses = [load_cam2world(scan_path / f"{fid}.pose.txt") for fid in best_views]
        n_candidates = len(best_views)

        if isinstance(nbv_cfg.kmeans_n_clusters, int) and nbv_cfg.kmeans_n_clusters > 0:
            n_clusters = min(n_candidates, nbv_cfg.kmeans_n_clusters)
        else:
            n_clusters = max(6, min(40, int(round(n_candidates / 12)))) if n_candidates >= 6 else n_candidates

        cluster_reps, cluster_labels = cluster_camera_poses(
            camera_poses, best_views, n_clusters
        )

        print(f"[INFO] Selected {len(cluster_reps)} cluster representatives.")

    # -------------------------- Save Outputs ----------------------------
    color_out = output_dir / "color"
    depth_out = output_dir / "depth"
    pose_out = output_dir / "pose"
    for d in (color_out, depth_out, pose_out): d.mkdir(parents=True, exist_ok=True)

    cam_json = {}
    for fid in cluster_reps:
        cam_json[fid] = load_cam2world(scan_path / f"{fid}.pose.txt").tolist()
        shutil.copy(scan_path / f"{fid}.color.jpg", color_out / f"{fid}.jpg")
        shutil.copy(scan_path / f"{fid}.depth.pgm", depth_out / f"{fid}.pgm")
        shutil.copy(scan_path / f"{fid}.pose.txt", pose_out / f"{fid}.txt")
        print(f"[INFO] Saved selected frame {fid}.")

    with open(output_dir / "camera_pose.json", "w") as f:
        json.dump(cam_json, f, indent=2)

    # ----------------------- Scene-level Graph Generation -----------------------
    if nbv_cfg.build_scene_graph:
        print("[INFO] Building scene-level graph...")
        # Extract only list-valued semantic attributes (color, shape, etc.)
        # from the full metadata cache (which also has scalars like nyu40, ply_color).
        sg_attrs = None
        if object_attributes_cache:
            sg_attrs = {}
            for oid, attrs in object_attributes_cache.items():
                sg_attrs[oid] = {
                    k: v for k, v in attrs.items()
                    if isinstance(v, list) and all(isinstance(x, str) for x in v)
                }
        scene_graph = build_scene_graph(
            object_geometry_cache, obj_to_label, verts,
            gravity_axis=nbv_cfg.gravity_axis,
            max_distance=nbv_cfg.scene_graph_max_distance,
            object_attributes=sg_attrs,
        )
        if nbv_cfg.scene_graph_add_embeddings:
            print(f"[INFO] Adding {nbv_cfg.scene_graph_embedding_type} embeddings...")
            add_embeddings_to_scene_graph(scene_graph, nbv_cfg.scene_graph_embedding_type)
        sg_path = output_dir / f"{scene_id}_scene_graph.json"
        save_scene_graph(scene_graph, sg_path)
        print(f"[INFO] Scene graph saved: {len(scene_graph['objects'])} objects, "
              f"{len(scene_graph['edge_lists']['relation'])} edges → {sg_path}")

    # ------------------------- Mask Rendering ----------------------------
    if save_instance_masks or save_semantic_masks:
        print("[INFO] Rendering masks for selected frames...")
        inst_dir = output_dir / "instance"
        sem_dir = output_dir / "semantic"
        if save_instance_masks: inst_dir.mkdir(parents=True, exist_ok=True)
        if save_semantic_masks: sem_dir.mkdir(parents=True, exist_ok=True)
        if save_semantic_masks:
            unique_sem = sorted({int(v) for v in obj_to_sem_id.values() if int(v) != VOID_ID})
            print(
                f"[INFO] Semantic mask export uses canonical '{nbv_cfg.semantic_id_key}' ids "
                f"with {len(unique_sem)} non-VOID classes in this scene."
            )
            expected_sem_ids = {int(v) for v in obj_to_sem_id.values()}
            expected_sem_ids.add(int(VOID_ID))

        mask_ds = nbv_cfg.mask_downsample_factor
        Hm = max(1, H0 // max(1, mask_ds))
        Wm = max(1, W0 // max(1, mask_ds))
        fx_m, fy_m, cx_m, cy_m = fx / mask_ds, fy / mask_ds, cx / mask_ds, cy / mask_ds
        rasterizer_mask = make_rasterizer(Hm, Wm)
        face_obj_ids_np = np.asarray(face_obj_ids, dtype=np.int32)

        for fid in tqdm(cluster_reps, desc="Masks", dynamic_ncols=True):
            pose = load_cam2world(scan_path / f"{fid}.pose.txt")
            R_cv, t_cv = invert_se3_to_opencv(pose)
            cams = make_p3d_camera_from_opencv(
                R_cv, t_cv, fx_m, fy_m, cx_m, cy_m, Hm, Wm, device
            )
            pix_to_face, _ = rasterize_visibility(meshes, cams, rasterizer_mask)
            p2f_np = pix_to_face.cpu().numpy()

            if save_instance_masks:
                inst = pix_to_instance_mask(p2f_np, face_obj_ids_np, VOID_ID)
                save_png16(inst_dir / f"{fid}.png", inst)
            if save_semantic_masks:
                sem = pix_to_semantic_mask(p2f_np, face_obj_ids_np, obj_to_sem_id, VOID_ID)
                unexpected = set(int(x) for x in np.unique(sem)) - expected_sem_ids
                if unexpected:
                    print(f"[WARN] Semantic mask {fid} contains unexpected ids: {sorted(unexpected)}")
                save_png16(sem_dir / f"{fid}.png", sem)

    # --------------------------- Auto Clean ------------------------------
    if auto_clean:
        print("[INFO] Auto-clean enabled, deleting raw frames...")
        for ext in ("*.color.jpg", "*.depth.pgm", "*.pose.txt"):
            for f in scan_path.glob(ext):
                f.unlink()


@hydra.main(version_base=None, config_path="../../../configs", config_name="config")
def cli(cfg: DictConfig) -> None:
    main(
        scene_id=cfg.scan_id,
        cfg=cfg,
        device_str=cfg.device if cfg.device != "auto" else None,
        debug=cfg.dataset.debug,
        auto_clean=cfg.dataset.auto_clean,
        save_semantic_masks=cfg.dataset.save_semantic_masks,
        save_instance_masks=cfg.dataset.save_instance_masks,
        allow_partial=cfg.dataset.allow_partial,
    )


if __name__ == "__main__":
    cli()
