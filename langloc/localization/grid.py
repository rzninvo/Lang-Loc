"""Grid sampling, raycasting, and visibility utilities for localization.

Provides functions for loading 3RScan and ScanNet scene meshes, sampling
dense camera grids over the mesh footprint, performing single-ray visibility
tests, and computing per-camera visible object directions.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import open3d as o3d
import open3d.core as o3c

DatasetName = Literal["3rscan", "scannet"]


def _load_scene_3rscan(scan_dir: Path) -> Tuple[o3d.geometry.TriangleMesh, np.ndarray, Dict[int, np.ndarray]]:
    """Load a 3RScan scene mesh and build per-triangle / per-object maps."""
    ply = scan_dir / "labels.instances.annotated.v2.ply"
    if not ply.exists():
        raise FileNotFoundError(ply)
    mesh = o3d.io.read_triangle_mesh(str(ply))
    mesh.compute_vertex_normals()

    vc   = (np.asarray(mesh.vertex_colors) * 255 + 0.5).astype(np.uint32)
    vhex = (vc[:, 0] << 16) | (vc[:, 1] << 8) | vc[:, 2]

    with open(scan_dir.parent / "objects.json") as f:
        meta = {s["scan"]: s for s in json.load(f)["scans"]}[scan_dir.name]
    color2oid = {int(o["ply_color"].lstrip("#"), 16): int(o["id"])
                 for o in meta["objects"]}

    v_oid = np.array([color2oid.get(int(h), 0) for h in vhex], dtype=np.int32)
    tris  = np.asarray(mesh.triangles, dtype=np.int32)
    tri2obj = np.array([np.bincount(v_oid[t]).argmax() for t in tris],
                       dtype=np.int32)

    obj2faces: Dict[int, list] = {}
    for fid, oid in enumerate(tri2obj):
        if oid != 0:
            obj2faces.setdefault(int(oid), []).append(fid)
    obj2faces_arr = {k: np.asarray(v, dtype=np.int32) for k, v in obj2faces.items()}
    return mesh, tri2obj, obj2faces_arr


def _load_scene_scannet(scan_dir: Path) -> Tuple[o3d.geometry.TriangleMesh, np.ndarray, Dict[int, np.ndarray]]:
    """Load a ScanNet scene mesh and build per-triangle / per-object maps.

    Reads the ``*_vh_clean_2.ply`` mesh, segmentation JSON, and aggregation
    JSON to produce the same ``(mesh, tri2obj, obj2faces)`` tuple as the
    3RScan loader.
    """
    scan_id = scan_dir.name
    ply = scan_dir / f"{scan_id}_vh_clean_2.ply"
    if not ply.exists():
        raise FileNotFoundError(f"ScanNet mesh not found: {ply}")
    mesh = o3d.io.read_triangle_mesh(str(ply))
    mesh.compute_vertex_normals()

    segs_json = scan_dir / f"{scan_id}_vh_clean_2.0.010000.segs.json"
    if not segs_json.exists():
        segs_json = scan_dir / f"{scan_id}_vh_clean_2.segs.json"
    if not segs_json.exists():
        raise FileNotFoundError(f"Segs JSON not found for {scan_id}")

    agg_json = scan_dir / f"{scan_id}.aggregation.json"
    if not agg_json.exists():
        raise FileNotFoundError(f"Aggregation JSON not found for {scan_id}")

    segs = json.loads(segs_json.read_text())
    agg = json.loads(agg_json.read_text())

    vert_seg = np.array(segs["segIndices"], dtype=np.int32)

    seg_to_obj: Dict[int, int] = {}
    for g in agg["segGroups"]:
        oid = int(g["objectId"])
        for s in g["segments"]:
            seg_to_obj[int(s)] = oid

    v_oid = np.array([seg_to_obj.get(int(s), 0) for s in vert_seg], dtype=np.int32)
    tris = np.asarray(mesh.triangles, dtype=np.int32)
    tri2obj = np.array([np.bincount(v_oid[t]).argmax() for t in tris],
                       dtype=np.int32)

    obj2faces: Dict[int, list] = {}
    for fid, oid in enumerate(tri2obj):
        if oid != 0:
            obj2faces.setdefault(int(oid), []).append(fid)
    obj2faces_arr = {k: np.asarray(v, dtype=np.int32) for k, v in obj2faces.items()}
    return mesh, tri2obj, obj2faces_arr


def load_scene(scan_dir: Path,
               dataset: DatasetName = "3rscan",
               ) -> Tuple[o3d.geometry.TriangleMesh, np.ndarray, Dict[int, np.ndarray]]:
    """Load a scene mesh and build per-triangle / per-object maps.

    Dispatches to the 3RScan or ScanNet loader based on *dataset*.

    Args:
        scan_dir: Path to the individual scan directory.
        dataset: ``"3rscan"`` or ``"scannet"``.

    Returns:
        A 3-tuple ``(mesh, tri2obj, obj2faces)`` where

        - **mesh** is the loaded ``o3d.geometry.TriangleMesh`` with vertex
          normals computed.
        - **tri2obj** is an ``int32`` array of shape ``(T,)`` mapping each
          triangle index to an object ID (0 for background).
        - **obj2faces** is a ``dict[int, np.ndarray]`` mapping non-zero
          object IDs to their triangle-index arrays.

    Raises:
        FileNotFoundError: If the expected mesh file does not exist.
        ValueError: If *dataset* is not recognised.
    """
    if dataset == "3rscan":
        return _load_scene_3rscan(scan_dir)
    if dataset == "scannet":
        return _load_scene_scannet(scan_dir)
    raise ValueError(f"Unknown dataset: {dataset!r}")


def load_object_labels(scan_dir: Path,
                       dataset: DatasetName = "3rscan") -> Dict[int, str]:
    """Load per-object labels from scene metadata.

    Args:
        scan_dir: Path to the individual scan directory.
        dataset: ``"3rscan"`` or ``"scannet"``.

    Returns:
        Mapping from object ID to label string.
    """
    if dataset == "3rscan":
        semseg_path = scan_dir / "semseg.v2.json"
        if not semseg_path.exists():
            return {}
        groups = json.loads(semseg_path.read_text())["segGroups"]
        return {int(g["objectId"]): g.get("label", "").strip()
                for g in groups}
    if dataset == "scannet":
        scan_id = scan_dir.name
        agg_json = scan_dir / f"{scan_id}.aggregation.json"
        if not agg_json.exists():
            return {}
        agg = json.loads(agg_json.read_text())
        return {int(g.get("objectId", g.get("id", -1))): g.get("label", "").strip()
                for g in agg["segGroups"]}
    raise ValueError(f"Unknown dataset: {dataset!r}")


def extract_floor_bbox(scan_dir: Path,
                       verts: np.ndarray,
                       tris: np.ndarray,
                       obj2faces: Dict[int, np.ndarray],
                       dataset: DatasetName = "3rscan",
                       ) -> Optional[Dict[str, float]]:
    """Return the axis-aligned bounding box of all floor-labelled objects.

    Args:
        scan_dir: Path to the individual scan directory.
        verts: Mesh vertex positions, shape ``(V, 3)``.
        tris: Mesh triangle indices, shape ``(F, 3)``.
        obj2faces: Mapping from object ID to face index array.
        dataset: ``"3rscan"`` or ``"scannet"``.

    Returns:
        Dict with keys ``x_min, x_max, y_min, y_max, z_min, z_max``,
        or ``None`` if the metadata is missing or contains no floor.
    """
    labels = load_object_labels(scan_dir, dataset)
    floor_ids = {oid for oid, lbl in labels.items()
                 if "floor" in lbl.lower().split()}
    if not floor_ids:
        return None

    face_lists = [obj2faces[oid] for oid in floor_ids if oid in obj2faces]
    if not face_lists:
        return None

    floor_faces = np.concatenate(face_lists)
    floor_vert_idx = np.unique(tris[floor_faces].ravel())
    floor_verts = verts[floor_vert_idx]

    return {
        "x_min": float(floor_verts[:, 0].min()),
        "x_max": float(floor_verts[:, 0].max()),
        "y_min": float(floor_verts[:, 1].min()),
        "y_max": float(floor_verts[:, 1].max()),
        "z_min": float(floor_verts[:, 2].min()),
        "z_max": float(floor_verts[:, 2].max()),
    }


def sample_grid(verts: np.ndarray, step: float, z_eye: float = 1.6,
                mesh: Optional[o3d.geometry.TriangleMesh] = None,
                return_indices: bool = False):
    """Sample a regular XY grid of candidate camera positions over the mesh.

    The grid covers the axis-aligned bounding box of the mesh vertices in
    X and Y, with all cameras placed at ``z_min + z_eye``.

    When *mesh* is provided, grid points that fall outside the scene geometry
    are pruned: a downward ray is cast from each candidate and only points
    whose ray hits the mesh floor are kept.

    Args:
        verts: Vertex positions array of shape ``(V, 3)``.
        step: Grid spacing in metres.
        z_eye: Height above the mesh floor (minimum Z) for camera placement.
        mesh: Optional mesh used for inside-scene filtering.
        return_indices: If ``True``, return a 4-tuple
            ``(cams, linear_indices, Nx, Ny)`` where *linear_indices* are
            the 0-based positions in the full ``Nx × Ny`` grid that each
            camera in *cams* occupies.  Needed to correctly map camera-array
            indices back to grid coordinates (e.g. for arrow-field rendering).

    Returns:
        ``(N, 3)`` float array of camera positions, or a 4-tuple
        ``(cams, linear_indices, Nx, Ny)`` when *return_indices* is ``True``.
    """
    xs, ys, zs = verts[:, 0], verts[:, 1], verts[:, 2]
    gx = np.arange(xs.min(), xs.max() + 1e-4, step)
    gy = np.arange(ys.min(), ys.max() + 1e-4, step)
    Nx, Ny = len(gx), len(gy)
    xv, yv = np.meshgrid(gx, gy, indexing="xy")
    n = xv.size
    cams = np.stack([xv.ravel(), yv.ravel(), np.full(n, zs.min() + z_eye)],
                    axis=1)

    if mesh is not None:
        rc = o3d.t.geometry.RaycastingScene()
        rc.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
        # Cast rays downward (-Z) from each camera
        dirs = np.zeros((n, 3), dtype=np.float32)
        dirs[:, 2] = -1.0
        rays = np.hstack([cams.astype(np.float32), dirs])
        hits = rc.cast_rays(o3c.Tensor(rays))
        t_hit = hits["t_hit"].numpy()
        inside_mask = np.isfinite(t_hit) & (t_hit > 0) & (t_hit < z_eye + 1.0)
        linear_indices = np.where(inside_mask)[0]
        cams = cams[linear_indices]
    else:
        linear_indices = np.arange(n)

    if return_indices:
        return cams, linear_indices, Nx, Ny
    return cams


def first_hit_is_object(cam: np.ndarray, centre: np.ndarray, target_oid: int,
                        rc: o3d.t.geometry.RaycastingScene,
                        tri2obj: np.ndarray) -> bool:
    """Test whether a ray from *cam* towards *centre* first hits the target object.

    Casts a single ray using Open3D's raycasting scene and checks whether
    the triangle struck belongs to ``target_oid``.

    Args:
        cam: Camera position, shape ``(3,)``.
        centre: Target object centroid, shape ``(3,)``.
        target_oid: Object ID to test against.
        rc: Pre-built Open3D raycasting scene.
        tri2obj: Per-triangle object-ID array from :func:`load_scene`.

    Returns:
        ``True`` if the first hit triangle belongs to ``target_oid``.
    """
    d = centre - cam
    l = np.linalg.norm(d)
    if l < 1e-6:
        return False
    ray = np.concatenate([cam, d / l])[None, :]
    ans = rc.cast_rays(o3c.Tensor(ray, dtype=o3c.Dtype.Float32))
    tri = int(ans["primitive_ids"].cpu().numpy()[0])
    if tri < 0 or tri >= len(tri2obj):
        return False
    return int(tri2obj[tri]) == int(target_oid)


def grid_from_bounds(bounds: Tuple[float, float, float, float],
                     step: float,
                     z_base: float,
                     z_eye: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create an XY camera grid within explicit bounding-box limits.

    Unlike :func:`sample_grid`, which derives bounds from mesh vertices,
    this function accepts an explicit ``(x_min, x_max, y_min, y_max)``
    rectangle.  Used by the coarse-to-fine search to spawn local grids
    around peak regions.

    Args:
        bounds: ``(x_min, x_max, y_min, y_max)`` in metres.
        step: Grid spacing in metres (clamped to >= 1e-6).
        z_base: Floor height (minimum Z of the mesh).
        z_eye: Eye-height offset added to *z_base*.

    Returns:
        A 3-tuple ``(cams, gx, gy)`` where *cams* is an ``(N, 3)`` array
        of camera positions and *gx*, *gy* are the 1-D tick arrays.
    """
    x_min, x_max, y_min, y_max = bounds
    step = max(float(step), 1e-6)
    gx = np.arange(x_min, x_max + 1e-4, step)
    gy = np.arange(y_min, y_max + 1e-4, step)
    if gx.size == 0 or gy.size == 0:
        return np.empty((0, 3), dtype=np.float64), gx, gy
    xv, yv = np.meshgrid(gx, gy, indexing="xy")
    n = xv.size
    cams = np.stack([xv.ravel(), yv.ravel(), np.full(n, z_base + z_eye)],
                    axis=1)
    return cams, gx, gy


def compute_visible_dirs(cams: np.ndarray,
                         centroids: Dict[int, np.ndarray],
                         rc: o3d.t.geometry.RaycastingScene,
                         tri2obj: np.ndarray) -> List[List[np.ndarray]]:
    """Compute per-camera unit direction vectors towards visible matched objects.

    For every camera position and every matched-object centroid, a
    visibility ray is cast.  If the first hit is the target object, the
    unit direction from camera to centroid is recorded.

    Args:
        cams: Camera positions, shape ``(N, 3)``.
        centroids: Mapping from object ID to centroid position ``(3,)``.
        rc: Pre-built Open3D raycasting scene.
        tri2obj: Per-triangle object-ID array from :func:`load_scene`.

    Returns:
        A list of length ``N``, where each element is a list of unit
        direction vectors (``np.ndarray`` of shape ``(3,)``) towards
        objects visible from that camera.
    """
    visible_dirs: List[List[np.ndarray]] = [[] for _ in range(len(cams))]
    for idx, cam in enumerate(cams):
        for oid, centre in centroids.items():
            if first_hit_is_object(cam, centre, oid, rc, tri2obj):
                d = centre - cam
                l = np.linalg.norm(d)
                if l > 1e-6:
                    visible_dirs[idx].append(d / l)
    return visible_dirs


# Labels that are visible from almost anywhere and carry little
# localization signal.
_GENERIC_LABELS = frozenset({
    "wall", "floor", "ceiling", "object", "doorframe", "door",
})


def compute_visible_dirs_weighted(
    cams: np.ndarray,
    centroids: Dict[int, np.ndarray],
    rc: o3d.t.geometry.RaycastingScene,
    tri2obj: np.ndarray,
    oid_weights: Optional[Dict[int, float]] = None,
) -> tuple[List[List[np.ndarray]], List[List[float]]]:
    """Like :func:`compute_visible_dirs` but also returns per-object weights.

    Args:
        cams: Camera positions, shape ``(N, 3)``.
        centroids: Mapping from object ID to centroid position ``(3,)``.
        rc: Pre-built Open3D raycasting scene.
        tri2obj: Per-triangle object-ID array.
        oid_weights: Per-object weight.  Defaults to 1.0 for every object.

    Returns:
        ``(visible_dirs, visible_weights)`` — parallel lists of length
        ``N``, each element being a list of direction vectors / floats.
    """
    if oid_weights is None:
        oid_weights = {oid: 1.0 for oid in centroids}
    visible_dirs: List[List[np.ndarray]] = [[] for _ in range(len(cams))]
    visible_weights: List[List[float]] = [[] for _ in range(len(cams))]
    for idx, cam in enumerate(cams):
        for oid, centre in centroids.items():
            if first_hit_is_object(cam, centre, oid, rc, tri2obj):
                d = centre - cam
                l = np.linalg.norm(d)
                if l > 1e-6:
                    visible_dirs[idx].append(d / l)
                    visible_weights[idx].append(oid_weights.get(oid, 1.0))
    return visible_dirs, visible_weights
