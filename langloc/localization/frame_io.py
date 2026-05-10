"""Frame JSON loading, caption graph building, and scene graph I/O.

This module provides the data-ingestion layer for localization evaluation.
It handles loading per-frame JSON annotations from 3RScan, constructing
caption scene graphs from visible objects and spatial relations, and
loading pre-processed 3D-SSG scene graphs.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch

from langloc.graphs.scene_graph import SceneGraph
from langloc.utils.embedding import _embed_word2vec


@dataclass
class FrameSelection:
    """A single frame JSON paired with the file it was loaded from.

    Attributes:
        frame: Parsed JSON dictionary for one frame.
        path: Filesystem path to the source JSON (or a virtual path when
            the JSON file contains a list of frames).
    """
    frame: dict
    path: Path


# ---------------------------------------------------------------------------
#  Frame loading and selection
# ---------------------------------------------------------------------------

def load_frame_jsons(desc_dir: Path) -> List[FrameSelection]:
    """Load all frame annotation JSONs from a descriptions directory.

    Prefers explicit ``frame-*.json`` files (excluding ``*_parsed.json``
    parser byproducts).  Falls back to ``all_descriptions.json`` and then
    to a broad ``*.json`` glob.  Results are de-duplicated by
    ``image_index`` to avoid double-counting when both individual and
    aggregate JSONs exist.

    Each JSON file may contain a single dict (one frame) or a list of
    dicts (multiple frames); both formats are supported.

    Args:
        desc_dir: Directory containing ``frame-*.json`` files.

    Returns:
        A list of :class:`FrameSelection` objects sorted by file name.
        Empty if *desc_dir* does not exist or contains no valid JSONs.
    """
    frames: List[FrameSelection] = []
    if not desc_dir.exists():
        return frames

    frame_jsons = sorted(
        p for p in desc_dir.glob("frame-*.json")
        if not p.stem.endswith("_parsed")
    )
    if frame_jsons:
        candidate_paths = frame_jsons
    else:
        all_desc = desc_dir / "all_descriptions.json"
        if all_desc.exists():
            candidate_paths = [all_desc]
        else:
            candidate_paths = sorted(
                p for p in desc_dir.glob("*.json")
                if not p.stem.endswith("_parsed")
            )

    for path in candidate_paths:
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue

        if isinstance(data, dict):
            frames.append(FrameSelection(frame=data, path=path))
        elif isinstance(data, list):
            for idx, item in enumerate(data):
                if not isinstance(item, dict):
                    continue
                virtual_name = path.with_name(f"{path.stem}_{idx:03d}{path.suffix}")
                frames.append(FrameSelection(frame=item, path=virtual_name))

    # De-duplicate by image_index so aggregate JSONs do not create duplicate
    # candidates for the same frame.
    deduped: List[FrameSelection] = []
    seen_image_indices: set = set()
    for fs in frames:
        image_index = str(fs.frame.get("image_index", "")).strip()
        if image_index:
            if image_index in seen_image_indices:
                continue
            seen_image_indices.add(image_index)
        deduped.append(fs)
    return deduped


def _total_pixels(fs: FrameSelection) -> int:
    """Sum ``pixel_count`` across visible objects in a frame."""
    objs = fs.frame.get("visible_objects", {}) or {}
    total = 0
    for obj in objs.values():
        try:
            total += int((obj or {}).get("pixel_count", 0))
        except (TypeError, ValueError):
            continue
    return total


def select_frame(frames: List[FrameSelection],
                 policy: str,
                 frame_index: int,
                 rng: np.random.Generator) -> Optional[FrameSelection]:
    """Pick one frame from a list according to the requested policy.

    Args:
        frames: Candidate frames to choose from.
        policy: Selection strategy.  One of ``"first"``, ``"index"``,
            ``"random"``, ``"max_visible"``, or ``"max_pixels"``.
        frame_index: Index used when *policy* is ``"index"``.
        rng: NumPy random generator for the ``"random"`` policy.

    Returns:
        The selected :class:`FrameSelection`, or ``None`` if *frames* is
        empty.

    Raises:
        ValueError: If *policy* is not a recognised strategy name.
    """
    if not frames:
        return None

    if policy == "first":
        return frames[0]
    if policy == "index":
        return frames[frame_index % len(frames)]
    if policy == "random":
        return frames[int(rng.integers(0, len(frames)))]
    if policy == "max_visible":
        # Deterministic tie-break:
        # 1) highest visible-object count
        # 2) highest total pixel_count
        # 3) stable filename order
        return min(
            frames,
            key=lambda fs: (
                -len(fs.frame.get("visible_objects", {}) or {}),
                -_total_pixels(fs),
                fs.path.name,
            ),
        )
    if policy == "max_pixels":
        return max(frames, key=_total_pixels)

    raise ValueError(f"Unknown frame selection policy '{policy}'")


# ---------------------------------------------------------------------------
#  Caption graph construction
# ---------------------------------------------------------------------------

def frame_to_scenegraph(frame: dict,
                        embedding_type: str = "word2vec",
                        use_attributes: bool = True,
                        query_embedding_mode: str = "doc") -> Tuple[SceneGraph, Dict[int, dict]]:
    """Build a caption SceneGraph from a frame's visible objects and spatial relations.

    Each visible object becomes a graph node (with word2vec label
    embedding), and each spatial relation becomes a directed edge.
    Objects are sorted by descending pixel count so that the most
    prominent instance of a duplicated label wins edge connections.

    Args:
        frame: Parsed frame JSON dict, expected to contain
            ``"visible_objects"`` and optionally ``"spatial_relations"``.
        embedding_type: Embedding backend.  Currently only ``"word2vec"``
            is supported.
        use_attributes: Whether to include attribute embeddings in the
            constructed SceneGraph.
        query_embedding_mode: Embedding mode passed to
            :func:`~langloc.utils.embedding._embed_word2vec`.
            ``"doc"`` for spaCy doc vectors, ``"token"`` for word2vec
            token embeddings.

    Returns:
        A 2-tuple ``(sg, meta)`` where

        - **sg** is the constructed :class:`SceneGraph`.
        - **meta** maps each internal node ID to a dict with keys
          ``"source_object_id"``, ``"label"``, and ``"centroid_world"``.

    Raises:
        ValueError: If *embedding_type* is not ``"word2vec"``.
    """
    if embedding_type != "word2vec":
        raise ValueError("Only word2vec embedding supported for evaluation graphs.")

    visible_objects = frame.get("visible_objects", {}) or {}
    sorted_items = sorted(
        visible_objects.items(),
        key=lambda kv: int(kv[1].get("pixel_count", 0)),
        reverse=True,
    )

    nodes: List[dict] = []
    label_lookup: Dict[str, List[int]] = {}
    meta: Dict[int, dict] = {}

    for new_id, (raw_id, obj) in enumerate(sorted_items):
        label = obj.get("label", f"object_{raw_id}")
        label_key = label.strip().lower()
        nodes.append({
            "id": new_id,
            "label": label,
            "attributes": [],
            "label_word2vec": _embed_word2vec(label, mode=query_embedding_mode),
            "attributes_word2vec": {"all": []},
        })
        label_lookup.setdefault(label_key, []).append(new_id)
        meta[new_id] = {
            "source_object_id": raw_id,
            "label": label,
            "centroid_world": np.asarray(obj.get("centroid_world", [0, 0, 0]),
                                         dtype=np.float32),
        }

    edges: List[dict] = []
    for rel in frame.get("spatial_relations", []) or []:
        subj = str(rel.get("subject", "")).strip().lower()
        obj = str(rel.get("object", "")).strip().lower()
        rel_type = rel.get("relation", "").strip()
        if not subj or not obj or not rel_type:
            continue
        subj_ids = label_lookup.get(subj)
        obj_ids = label_lookup.get(obj)
        if not subj_ids or not obj_ids:
            continue
        edges.append({
            "source": subj_ids[0],
            "target": obj_ids[0],
            "relationship": rel_type,
            "relation_word2vec": _embed_word2vec(rel_type, mode=query_embedding_mode),
        })

    graph_dict = {"nodes": nodes, "edges": edges}
    sg = SceneGraph(scene_id=frame.get("scene_index", "unknown_scene"),
                    txt_id=frame.get("image_index"),
                    graph_type="scanscribe",
                    graph=graph_dict,
                    embedding_type=embedding_type,
                    use_attributes=use_attributes)
    return sg, meta


# ---------------------------------------------------------------------------
#  Parsed-graph centroid recovery
# ---------------------------------------------------------------------------

def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors, returning 0 for degenerate inputs."""
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def recover_centroids(parsed_graph: dict,
                      parsed_path: Path,
                      embedding_mode: str = "doc",
                      similarity_threshold: float = 0.7) -> Dict[int, dict]:
    """Match parsed-graph node labels to visible objects to recover centroids.

    When a scene description is parsed by GPT into a structured graph, the
    resulting nodes have labels but no 3D geometry.  This function matches
    each parsed node label to the closest visible object (from the original
    frame JSON) via word2vec cosine similarity and copies the centroid.

    Args:
        parsed_graph: Dict with ``"nodes"`` list, each having ``"id"`` and
            ``"label"`` (and optionally ``"label_word2vec"``).
        parsed_path: Path to the ``*_parsed.json`` file.  The corresponding
            raw frame JSON is found by stripping the ``_parsed`` suffix.
        embedding_mode: Word2vec mode (``"doc"`` or ``"token"``).
        similarity_threshold: Minimum cosine similarity to accept a match.

    Returns:
        Dict mapping parsed node ID to a metadata dict with keys
        ``"source_object_id"``, ``"label"``, ``"centroid_world"``, and
        ``"match_similarity"``.
    """
    # Resolve raw frame path from parsed path
    stem = parsed_path.stem
    if stem.endswith("_parsed"):
        raw_name = stem[: -len("_parsed")] + parsed_path.suffix
    else:
        raw_name = parsed_path.name
    raw_path = parsed_path.with_name(raw_name)
    if not raw_path.exists():
        return {}

    raw_frame = json.loads(raw_path.read_text())
    visible_objects = raw_frame.get("visible_objects", {}) or {}
    if not visible_objects:
        return {}

    # Build visible-object embeddings
    vo_ids: List[str] = []
    vo_labels: List[str] = []
    vo_centroids: List[np.ndarray] = []
    vo_embeddings: List[np.ndarray] = []
    for raw_id, obj in visible_objects.items():
        label = obj.get("label", "")
        if not label:
            continue
        vo_ids.append(raw_id)
        vo_labels.append(label)
        vo_centroids.append(
            np.asarray(obj.get("centroid_world", [0, 0, 0]), dtype=np.float32)
        )
        vo_embeddings.append(_embed_word2vec(label, mode=embedding_mode))

    if not vo_embeddings:
        return {}
    vo_emb_arr = np.array(vo_embeddings, dtype=np.float32)

    # Match each parsed node to the best visible object
    meta: Dict[int, dict] = {}
    for node in parsed_graph.get("nodes", []):
        nid = int(node["id"])
        label = node.get("label", "")
        if not label:
            continue

        # Use cached embedding if available, otherwise compute
        node_emb = node.get("label_word2vec")
        if node_emb is not None:
            node_emb = np.asarray(node_emb, dtype=np.float32)
        else:
            node_emb = _embed_word2vec(label, mode=embedding_mode)

        best_idx = -1
        best_sim = -1.0
        for j, vo_emb in enumerate(vo_emb_arr):
            sim = _cosine_sim(node_emb, vo_emb)
            if sim > best_sim:
                best_sim = sim
                best_idx = j

        if best_sim >= similarity_threshold and best_idx >= 0:
            meta[nid] = {
                "source_object_id": vo_ids[best_idx],
                "label": label,
                "centroid_world": vo_centroids[best_idx],
                "match_similarity": best_sim,
            }

    return meta


# ---------------------------------------------------------------------------
#  Parsed-JSON caption path (paper §3.3 / Supp §4.3)
# ---------------------------------------------------------------------------

def load_parsed_frame_jsons(desc_dir: Path) -> List[FrameSelection]:
    """Load all ``*_parsed.json`` files from a descriptions directory.

    Each parsed JSON carries a ``parsed_graph`` field (nodes + edges
    extracted by GPT-4o-mini from the natural-language ``description``)
    plus the original frame metadata.  Used by the paper-protocol
    fine-localization path (``caption_source=parsed``).

    Naming conventions handled:

    - 3RScan: ``frame-NNNNNN_parsed.json``
    - ScanNet: ``NNNNNN_parsed.json``

    Args:
        desc_dir: Directory containing ``*_parsed.json`` files.

    Returns:
        List of :class:`FrameSelection` objects sorted by file name.
        Empty if the directory is missing or contains no parsed files.
    """
    frames: List[FrameSelection] = []
    if not desc_dir.exists():
        return frames

    for path in sorted(desc_dir.glob("*_parsed.json")):
        if not path.stem.endswith("_parsed"):
            continue  # Defensive — glob already filters.
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            frames.append(FrameSelection(frame=data, path=path))
    return frames


def parsed_frame_to_scenegraph(parsed_data: dict,
                               parsed_path: Path,
                               query_embedding_mode: str = "doc",
                               centroid_similarity_threshold: float = 0.7,
                               ) -> Tuple[Optional["SceneGraph"], Dict[int, dict]]:
    """Build a SceneGraph from a parsed-JSON caption (paper protocol).

    Implements the parsed-JSON caption path used by the published
    Table 4 numbers:

    1. Read the GPT-parsed nodes / edges from ``parsed_data["parsed_graph"]``.
    2. Ensure each node carries a Word2Vec label embedding (computed
       on demand if missing).
    3. Match each parsed node label back to the original frame's
       ``visible_objects`` via :func:`recover_centroids`, accepting
       only matches whose Word2Vec cosine ≥ γ
       (``centroid_similarity_threshold``).  The matched VO's
       ``centroid_world`` becomes the parsed node's centroid.
    4. Drop parsed nodes that fail to ground (paper "Frames without
       sufficient grounding are discarded" — Supp §4.3).
    5. Drop edges whose endpoints did not survive grounding.
    6. Build a SceneGraph with the surviving nodes and edges, in the
       same 518-D feature schema used by ``frame_to_scenegraph`` (zero-
       padded color / radius for parsed nodes).

    Args:
        parsed_data: Top-level dict loaded from a ``*_parsed.json``
            file.  Must contain ``parsed_graph`` with ``nodes`` and
            ``edges`` lists; falls back to an empty graph otherwise.
        parsed_path: Path to the parsed JSON.  Used to resolve the raw
            frame JSON for centroid recovery.
        query_embedding_mode: Word2vec aggregation mode for any nodes
            whose precomputed ``label_word2vec`` is missing.  Paper
            default: ``"doc"`` (Supp Table 7 / Master Table A.4).
        centroid_similarity_threshold: Grounding threshold γ
            (paper / supp / master report value: 0.7).  Parsed nodes
            whose best Word2Vec match falls below γ are discarded.

    Returns:
        A 2-tuple ``(scene_graph, meta)``.  ``scene_graph`` is ``None``
        if no parsed nodes survived grounding.  ``meta`` maps surviving
        parsed-node IDs to grounding metadata
        (``source_object_id``, ``matched_vo_label``, ``match_similarity``,
        ``centroid_world``).

    Line-by-line equivalent of mk5's ``parsed_frame_to_scenegraph``
    (visualize_eval_loc_mk5.py:276-331), specialised to our existing
    518-D feature schema and ``SceneGraph`` constructor.
    """
    parsed_graph = parsed_data.get("parsed_graph", {}) or {}
    parsed_nodes = parsed_graph.get("nodes", []) or []
    parsed_edges = parsed_graph.get("edges", []) or []
    if not parsed_nodes:
        return None, {}

    meta = recover_centroids(
        parsed_graph,
        parsed_path,
        embedding_mode=query_embedding_mode,
        similarity_threshold=centroid_similarity_threshold,
    )
    if not meta:
        return None, {}

    # Re-emit nodes / edges in the schema accepted by the SceneGraph
    # constructor (same shape as `frame_to_scenegraph`).  Each surviving
    # parsed node carries its Word2Vec embedding; centroid information
    # is recorded in the `meta` return — visibility ray-casting uses
    # scene-graph centroids (not caption-graph ones), so the caption
    # graph itself does not need geometry.
    surviving_node_ids: set = set()
    out_nodes: List[dict] = []
    for node in parsed_nodes:
        nid = int(node["id"])
        if nid not in meta:
            continue  # Grounding γ=0.7 dropped this parsed node.
        label = node.get("label", "") or f"object_{nid}"
        node_emb = node.get("label_word2vec")
        if node_emb is None or (hasattr(node_emb, "__len__") and len(node_emb) == 0):
            node_emb = _embed_word2vec(label, mode=query_embedding_mode)
        attr_w2v = node.get("attributes_word2vec") or {"all": []}
        out_nodes.append({
            "id": nid,
            "label": label,
            "attributes": list(node.get("attributes", []) or []),
            "label_word2vec": node_emb,
            "attributes_word2vec": attr_w2v,
        })
        surviving_node_ids.add(nid)

    if not out_nodes:
        return None, {}

    out_edges: List[dict] = []
    for edge in parsed_edges:
        try:
            src = int(edge.get("source"))
            tgt = int(edge.get("target"))
        except (TypeError, ValueError):
            continue
        if src not in surviving_node_ids or tgt not in surviving_node_ids:
            continue
        rel_type = edge.get("relationship") or edge.get("relation", "near")
        rel_emb = edge.get("relation_word2vec")
        if rel_emb is None or (hasattr(rel_emb, "__len__") and len(rel_emb) == 0):
            rel_emb = _embed_word2vec(rel_type, mode=query_embedding_mode)
        out_edges.append({
            "source": src,
            "target": tgt,
            "relationship": rel_type,
            "relation_word2vec": rel_emb,
        })

    graph_dict = {"nodes": out_nodes, "edges": out_edges}
    sg = SceneGraph(
        scene_id=parsed_data.get("scene_index", "unknown_scene"),
        txt_id=parsed_data.get("source_frame", parsed_path.stem),
        graph_type="scanscribe",
        graph=graph_dict,
        embedding_type="word2vec",
        use_attributes=True,
    )
    return sg, meta


# ---------------------------------------------------------------------------
#  Camera pose helper
# ---------------------------------------------------------------------------

def camera_center_from_pose(pose: Iterable[Iterable[float]]) -> np.ndarray:
    """Extract the camera centre (translation) from a 4x4 pose matrix.

    Args:
        pose: A 4x4 camera-to-world transformation matrix (nested lists
            or array-like).

    Returns:
        Camera centre as a float32 array of shape ``(3,)``.

    Raises:
        ValueError: If the pose does not have shape ``(4, 4)``.
    """
    mat = np.asarray(pose, dtype=np.float64)
    if mat.shape != (4, 4):
        raise ValueError(f"Expected 4x4 scene_pose, got shape {mat.shape}")
    t = mat[:3, 3]
    return t.astype(np.float32)


# ---------------------------------------------------------------------------
#  Scene graph loading
# ---------------------------------------------------------------------------

def load_scene_graphs(graphs_dir: Path,
                      max_dist: float = 1.0,
                      embedding_type: str = "word2vec",
                      use_attributes: bool = True,
                      dataset: str = "3rscan") -> Dict[str, SceneGraph]:
    """Load pre-processed scene graphs into a dict keyed by scene ID.

    Args:
        graphs_dir: Root of the processed-data directory.
        max_dist: Maximum distance for graph construction.
        embedding_type: Embedding backend name.
        use_attributes: Whether to include attribute embeddings.
        dataset: ``"3rscan"`` or ``"scannet"``.  Controls which ``.pt`` file
            is loaded (``3dssg/...`` vs ``scannet_scene_graphs.pt``).

    Returns:
        A dict mapping scene ID strings to :class:`SceneGraph` instances.

    Raises:
        FileNotFoundError: If the expected ``.pt`` file does not exist.
    """
    if dataset == "scannet":
        g3d_path = graphs_dir / "scannet_scene_graphs.pt"
    else:
        g3d_path = graphs_dir / "3dssg" / "3dssg_graphs_processed_edgelists_relationembed.pt"
    if not g3d_path.exists():
        raise FileNotFoundError(g3d_path)
    g3d = torch.load(g3d_path, map_location="cpu", weights_only=False)
    scenes: Dict[str, SceneGraph] = {}
    for sid, graph in g3d.items():
        scenes[sid] = SceneGraph(sid,
                                 graph_type="3dssg",
                                 graph=graph,
                                 max_dist=max_dist,
                                 embedding_type=embedding_type,
                                 use_attributes=use_attributes)
    return scenes


def ensure_query_root(query_root: Optional[Path], root: Path) -> Path:
    """Return *query_root* if set, otherwise fall back to *root*.

    Args:
        query_root: Explicitly specified query root, or ``None``.
        root: Default root directory to use as fallback.

    Returns:
        The resolved query root path.
    """
    if query_root is not None:
        return Path(query_root)
    return Path(root)


def format_args_section(args: object) -> str:
    """Render a configuration object as a human-readable parameter listing.

    Supports both ``argparse.Namespace`` objects and dict-like Hydra
    configs (``OmegaConf`` ``DictConfig``).

    Args:
        args: Configuration namespace or dict-like object.

    Returns:
        A multi-line string suitable for log files.
    """

    def _stringify(value: object) -> str:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, (list, tuple)):
            return "[" + ", ".join(_stringify(v) for v in value) + "]"
        if isinstance(value, dict):
            items = ", ".join(f"{k}: {_stringify(v)}" for k, v in value.items())
            return "{" + items + "}"
        return str(value)

    if isinstance(args, argparse.Namespace):
        items = vars(args)
    elif hasattr(args, "items"):
        items = dict(args.items()) if callable(getattr(args, "items", None)) else dict(args)
    else:
        items = vars(args)

    lines = ["Parameters used", "---------------"]
    for key in sorted(items):
        if key.startswith("_"):
            continue
        value = items[key]
        lines.append(f"{key}: {_stringify(value)}")
    return "\n".join(lines)
