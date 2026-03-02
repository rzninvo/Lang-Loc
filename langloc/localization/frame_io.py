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

from langloc.data_processing.scene_graph import SceneGraph
from langloc.data_processing.create_text_embeddings import create_embedding_nlp


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
#  Word2vec embedding cache
# ---------------------------------------------------------------------------

_EMBED_CACHE: Dict[str, np.ndarray] = {}


def _embed_word2vec(text: str) -> List[float]:
    """Return a word2vec embedding for *text*, using a module-level cache.

    Args:
        text: Free-form text string to embed.

    Returns:
        Embedding vector as a list of floats.
    """
    key = text.strip().lower()
    cached = _EMBED_CACHE.get(key)
    if cached is None:
        vec = np.asarray(create_embedding_nlp(text), dtype=np.float32)
        cached = vec
        _EMBED_CACHE[key] = cached
    return cached.tolist()


# ---------------------------------------------------------------------------
#  Frame loading and selection
# ---------------------------------------------------------------------------

def load_frame_jsons(desc_dir: Path) -> List[FrameSelection]:
    """Load all frame annotation JSONs from a descriptions directory.

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
    for path in sorted(desc_dir.glob("*.json")):
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
    return frames


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
        return max(frames,
                   key=lambda fs: len(fs.frame.get("visible_objects", {})))
    if policy == "max_pixels":
        def total_pixels(fs: FrameSelection) -> int:
            objs = fs.frame.get("visible_objects", {})
            return sum(int(obj.get("pixel_count", 0)) for obj in objs.values())

        return max(frames, key=total_pixels)

    raise ValueError(f"Unknown frame selection policy '{policy}'")


# ---------------------------------------------------------------------------
#  Caption graph construction
# ---------------------------------------------------------------------------

def frame_to_scenegraph(frame: dict,
                        embedding_type: str = "word2vec",
                        use_attributes: bool = True) -> Tuple[SceneGraph, Dict[int, dict]]:
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
            "label_word2vec": _embed_word2vec(label),
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
            "relation_word2vec": _embed_word2vec(rel_type),
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
                      use_attributes: bool = True) -> Dict[str, SceneGraph]:
    """Load pre-processed 3D-SSG graphs into a dict keyed by scene ID.

    Args:
        graphs_dir: Root of the processed-data directory, expected to
            contain ``3dssg/3dssg_graphs_processed_edgelists_relationembed.pt``.
        max_dist: Maximum distance for graph construction.
        embedding_type: Embedding backend name.
        use_attributes: Whether to include attribute embeddings.

    Returns:
        A dict mapping scene ID strings to :class:`SceneGraph` instances.

    Raises:
        FileNotFoundError: If the expected ``.pt`` file does not exist.
    """
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
        return query_root
    return root


def format_args_section(args) -> str:
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
