"""Build scene-level graphs in 3DSSG-compatible format from mesh segmentation.

Generates a complete scene graph (all objects + all pairwise spatial relations)
that can be consumed by ``SceneGraph(graph_type='3dssg')``.  Relations are
computed in world coordinates using a gravity-aligned canonical reference frame
derived from PCA on the room's floor-plane projection.
"""

import json
import logging
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Canonical reference frame
# ---------------------------------------------------------------------------

def compute_canonical_frame(
    vertices: np.ndarray,
    gravity_axis: str = "y_up",
) -> Dict[str, np.ndarray]:
    """Compute a canonical room reference frame from mesh vertices.

    Projects all vertices onto the horizontal plane (perpendicular to gravity),
    runs PCA to find the room's principal horizontal axes, and returns a
    right-handed orthonormal frame.

    Args:
        vertices: ``(N, 3)`` mesh vertex positions.
        gravity_axis: ``"y_up"`` (ScanNet) or ``"z_up"`` (3RScan).

    Returns:
        Dictionary with ``front_axis``, ``right_axis``, ``up_axis`` (each 3-d
        unit vectors) and ``gravity_axis`` string.
    """
    if gravity_axis == "y_up":
        up = np.array([0.0, 1.0, 0.0])
        h_dims = [0, 2]  # X, Z
    elif gravity_axis == "z_up":
        up = np.array([0.0, 0.0, 1.0])
        h_dims = [0, 1]  # X, Y
    else:
        raise ValueError(f"Unknown gravity_axis: {gravity_axis}")

    # Project to floor plane and run PCA
    v2d = vertices[:, h_dims]
    v2d_centered = v2d - v2d.mean(axis=0)
    cov = np.cov(v2d_centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    # Principal component = largest eigenvalue = longest room extent = "front"
    order = np.argsort(eigvals)[::-1]
    pc1 = eigvecs[:, order[0]]  # 2-d unit vector

    # Embed back into 3D
    front = np.zeros(3)
    front[h_dims[0]] = pc1[0]
    front[h_dims[1]] = pc1[1]
    front = front / (np.linalg.norm(front) + 1e-12)

    # Right = front x up  (right-handed)
    right = np.cross(front, up)
    right = right / (np.linalg.norm(right) + 1e-12)

    return {
        "front_axis": front,
        "right_axis": right,
        "up_axis": up,
        "gravity_axis": gravity_axis,
    }


# ---------------------------------------------------------------------------
# OBB format conversion
# ---------------------------------------------------------------------------

def _obb_to_3dssg_format(
    obb_center: np.ndarray,
    obb_axes: np.ndarray,
    obb_extents: np.ndarray,
) -> Dict:
    """Convert internal OBB to 3DSSG format.

    Our OBB stores axes as rows and half-extents; 3DSSG uses flat
    ``normalizedAxes`` (9 floats, column-major) and full ``axesLengths``.

    Args:
        obb_center: ``(3,)`` center position.
        obb_axes: ``(3, 3)`` orthonormal axes (rows are axis vectors).
        obb_extents: ``(3,)`` half-extents along each axis.

    Returns:
        Dictionary with ``centroid``, ``normalizedAxes``, ``axesLengths``.
    """
    # 3DSSG stores axes column-major: axes_cols = axes.T, then flatten
    axes_cols = obb_axes.T  # (3, 3) with columns as axis vectors
    return {
        "centroid": obb_center.tolist(),
        "normalizedAxes": axes_cols.flatten().tolist(),
        "axesLengths": (obb_extents * 2.0).tolist(),  # full lengths
    }


# ---------------------------------------------------------------------------
# Scene-level spatial relations
# ---------------------------------------------------------------------------

# Semantic rules: restrict which relations certain labels can be SUBJECT of.
_SEMANTIC_RULES = {
    "ceiling": {"higher than"},
    "floor": {"lower than", "supported by"},
    "wall": {"behind", "front", "close by", "left", "right"},
    "carpet": {"lower than", "supported by"},
    "rug": {"lower than", "supported by"},
}


def compute_scene_relations(
    object_geometry: Dict[int, Dict[str, np.ndarray]],
    obj_to_label: Dict[int, str],
    canonical_frame: Dict[str, np.ndarray],
    gravity_axis: str = "y_up",
    max_distance: float = 2.0,
    size_ratio_threshold: float = 5.0,
    eps: float = 0.1,
) -> List[Dict]:
    """Compute spatial relations between ALL object pairs in the scene.

    Uses world coordinates with a canonical reference frame for directional
    predicates.  Matches the 3DSSG relation vocabulary.

    Args:
        object_geometry: Object ID -> geometry dict from
            ``precompute_object_geometry()``.  Must include ``centroid_world``,
            ``bbox_min``, ``bbox_max``.
        obj_to_label: Object ID -> label string.
        canonical_frame: Output of ``compute_canonical_frame()``.
        gravity_axis: ``"y_up"`` or ``"z_up"``.
        max_distance: Max centroid distance (m) to consider a pair.
        size_ratio_threshold: Max bbox diagonal ratio to consider a pair.
        eps: Min displacement (m) for directional relations.

    Returns:
        List of relation dicts with ``subject_id``, ``object_id``,
        ``subject_label``, ``object_label``, ``relation``, ``distance``.
    """
    front = canonical_frame["front_axis"]
    right = canonical_frame["right_axis"]
    up = canonical_frame["up_axis"]

    g_idx = 1 if gravity_axis == "y_up" else 2

    relations: List[Dict] = []
    oids = sorted(object_geometry.keys())

    # Pre-compute sizes and volumes
    sizes = {}
    volumes = {}
    for oid in oids:
        geo = object_geometry[oid]
        dims = np.maximum(geo["bbox_max"] - geo["bbox_min"], 1e-8)
        sizes[oid] = float(np.linalg.norm(dims))
        volumes[oid] = float(dims[0] * dims[1] * dims[2])

    def _add(subj_id, obj_id, relation, dist):
        subj_label = obj_to_label.get(subj_id, f"id_{subj_id}").lower()
        obj_label = obj_to_label.get(obj_id, f"id_{obj_id}").lower()
        allowed = _SEMANTIC_RULES.get(subj_label)
        if allowed is not None and relation not in allowed:
            return
        relations.append({
            "subject_id": subj_id,
            "object_id": obj_id,
            "subject_label": subj_label,
            "object_label": obj_label,
            "relation": relation,
            "distance": dist,
        })

    for id_a, id_b in combinations(oids, 2):
        geo_a = object_geometry[id_a]
        geo_b = object_geometry[id_b]

        cw_a = geo_a["centroid_world"]
        cw_b = geo_b["centroid_world"]
        dist_world = float(np.linalg.norm(cw_a - cw_b))

        if dist_world > max_distance:
            continue

        # Size ratio filter
        sa, sb = sizes[id_a], sizes[id_b]
        if sa > 0 and sb > 0:
            if max(sa, sb) / min(sa, sb) > size_ratio_threshold:
                continue

        delta = cw_b - cw_a  # vector from A to B in world frame

        # ── 1. DIRECTIONAL (canonical frame, dominant horizontal axis) ──
        proj_front = float(np.dot(delta, front))
        proj_right = float(np.dot(delta, right))
        proj_up = float(np.dot(delta, up))

        # Horizontal directional: pick dominant horizontal axis
        if abs(proj_front) > abs(proj_right) and abs(proj_front) > eps:
            if proj_front > 0:
                _add(id_b, id_a, "front", dist_world)
                _add(id_a, id_b, "behind", dist_world)
            else:
                _add(id_b, id_a, "behind", dist_world)
                _add(id_a, id_b, "front", dist_world)
        elif abs(proj_right) > eps:
            if proj_right > 0:
                _add(id_b, id_a, "right", dist_world)
                _add(id_a, id_b, "left", dist_world)
            else:
                _add(id_b, id_a, "left", dist_world)
                _add(id_a, id_b, "right", dist_world)

        # Vertical directional
        if abs(proj_up) > eps:
            if proj_up > 0:
                _add(id_b, id_a, "higher than", dist_world)
                _add(id_a, id_b, "lower than", dist_world)
            else:
                _add(id_b, id_a, "lower than", dist_world)
                _add(id_a, id_b, "higher than", dist_world)

        # ── 2. COMPARATIVE: bigger than / smaller than ──
        vol_a, vol_b = volumes[id_a], volumes[id_b]
        if vol_a > 0 and vol_b > 0:
            vol_ratio = vol_a / vol_b
            if vol_ratio > 1.5:
                _add(id_a, id_b, "bigger than", dist_world)
                _add(id_b, id_a, "smaller than", dist_world)
            elif vol_ratio < 1.0 / 1.5:
                _add(id_a, id_b, "smaller than", dist_world)
                _add(id_b, id_a, "bigger than", dist_world)

        # ── 3. PROXIMITY: close by ──
        if dist_world < 0.5:
            _add(id_a, id_b, "close by", dist_world)
            _add(id_b, id_a, "close by", dist_world)

        # ── 4. SUPPORT: standing on / supported by ──
        if abs(proj_up) > eps and dist_world < 1.0:
            horiz_dist = float(np.sqrt(
                dist_world**2 - proj_up**2
            ))
            if horiz_dist < max(sa, sb) * 0.7:
                if proj_up > eps:
                    # B is higher → B standing on A
                    _add(id_b, id_a, "standing on", dist_world)
                    _add(id_a, id_b, "supported by", dist_world)
                else:
                    _add(id_a, id_b, "standing on", dist_world)
                    _add(id_b, id_a, "supported by", dist_world)

    logger.info("Computed %d scene-level relations across %d objects",
                len(relations), len(oids))
    return relations


# ---------------------------------------------------------------------------
# Build the full scene graph dict
# ---------------------------------------------------------------------------

def build_scene_graph(
    object_geometry: Dict[int, Dict[str, np.ndarray]],
    obj_to_label: Dict[int, str],
    vertices: np.ndarray,
    gravity_axis: str = "y_up",
    max_distance: float = 2.0,
    size_ratio_threshold: float = 5.0,
    object_attributes: Optional[Dict[int, Dict]] = None,
) -> Dict:
    """Build a 3DSSG-compatible scene graph from mesh segmentation data.

    Args:
        object_geometry: Per-object geometry from ``precompute_object_geometry()``.
        obj_to_label: Object ID -> semantic label.
        vertices: ``(N, 3)`` mesh vertices (for canonical frame PCA).
        gravity_axis: ``"y_up"`` (ScanNet) or ``"z_up"`` (3RScan).
        max_distance: Max centroid distance for relation computation.
        size_ratio_threshold: Max bbox diagonal ratio for relation computation.
        object_attributes: Optional per-object attributes dict,
            e.g. ``{42: {"color": ["brown"], "shape": ["rectangular"]}}``.

    Returns:
        Dictionary with ``objects``, ``edge_lists``, and ``canonical_frame``
        keys, ready for ``SceneGraph(graph_type='3dssg')``.
    """
    canonical_frame = compute_canonical_frame(vertices, gravity_axis)

    # Build relations
    relations = compute_scene_relations(
        object_geometry, obj_to_label, canonical_frame,
        gravity_axis=gravity_axis,
        max_distance=max_distance,
        size_ratio_threshold=size_ratio_threshold,
    )

    # Build objects dict
    objects = {}
    for oid, geo in object_geometry.items():
        label = obj_to_label.get(oid, f"id_{oid}")
        obb = _obb_to_3dssg_format(geo["obb_center"], geo["obb_axes"], geo["obb_extents"])
        attrs = {}
        if object_attributes and oid in object_attributes:
            attrs = object_attributes[oid]

        objects[str(oid)] = {
            "id": str(oid),
            "label": label,
            "obb": obb,
            "attributes": attrs,
        }

    # Build edge lists (parallel arrays)
    edge_from = []
    edge_to = []
    edge_relation = []
    edge_distance = []
    for rel in relations:
        edge_from.append(rel["subject_id"])
        edge_to.append(rel["object_id"])
        edge_relation.append(rel["relation"])
        edge_distance.append(rel["distance"])

    graph = {
        "objects": objects,
        "edge_lists": {
            "from": edge_from,
            "to": edge_to,
            "relation": edge_relation,
            "distance": edge_distance,
        },
        "canonical_frame": {
            "front_axis": canonical_frame["front_axis"].tolist(),
            "right_axis": canonical_frame["right_axis"].tolist(),
            "up_axis": canonical_frame["up_axis"].tolist(),
            "gravity_axis": gravity_axis,
        },
    }

    logger.info("Built scene graph: %d objects, %d edges",
                len(objects), len(edge_from))
    return graph


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

def add_embeddings_to_scene_graph(
    graph: Dict,
    embedding_type: str = "word2vec",
) -> None:
    """Add label/attribute/relation embeddings to a scene graph (in-place).

    Reuses ``get_word2vec()`` from ``langloc.graphs.graph_loader_utils``.

    Args:
        graph: Scene graph dict (modified in place).
        embedding_type: ``"word2vec"`` or ``"clip"``.
    """
    from langloc.graphs.graph_loader_utils import get_word2vec, get_clip

    if embedding_type == "word2vec":
        _get_emb = get_word2vec
    elif embedding_type == "clip":
        _get_emb = get_clip
    else:
        raise ValueError(f"Unknown embedding_type: {embedding_type}")

    cache: dict = {}

    # Node embeddings
    for obj in graph["objects"].values():
        emb, cache = _get_emb(obj["label"], cache)
        emb_list = emb.tolist() if hasattr(emb, "tolist") else list(emb)
        obj[f"label_{embedding_type}"] = emb_list

        attrs_emb = {}
        for attr_key, attr_vals in obj.get("attributes", {}).items():
            attrs_emb[attr_key] = []
            for val in attr_vals:
                a_emb, cache = _get_emb(val, cache)
                a_list = a_emb.tolist() if hasattr(a_emb, "tolist") else list(a_emb)
                attrs_emb[attr_key].append(a_list)
        obj[f"attributes_{embedding_type}"] = attrs_emb

    # Edge embeddings
    rel_embeddings = []
    for rel_str in graph["edge_lists"]["relation"]:
        emb, cache = _get_emb(rel_str, cache)
        emb_list = emb.tolist() if hasattr(emb, "tolist") else list(emb)
        rel_embeddings.append(emb_list)
    graph["edge_lists"][f"relation_{embedding_type}"] = rel_embeddings

    logger.info("Added %s embeddings to %d nodes and %d edges",
                embedding_type, len(graph["objects"]),
                len(graph["edge_lists"]["relation"]))


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def save_scene_graph(graph: Dict, output_path: "Path | str") -> None:
    """Save a scene graph to JSON.

    Args:
        graph: Scene graph dictionary.
        output_path: Destination file path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(graph, f, indent=2)
    logger.info("Saved scene graph to %s", output_path)
