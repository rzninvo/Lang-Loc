"""Scene data structures, label canonicalization, and scene loading.

Provides the core data types (``FrameInfo``, ``SceneData``) for representing
parsed scene frames with their poses, visible labels, and spatial relations.
Also provides label/relation canonicalization helpers and the I/O routine
that loads ``all_descriptions*.json`` into a ``SceneData`` instance.

Key exports:
    FrameInfo, SceneData, load_scene_data, load_relaxed_json,
    DEFAULT_ALIASES, parse_aliases, canon_label, canon_relation,
    relation_to_phrase, pose_to_pos_dir, normalize_frame_id.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# JSON loader (tolerant of trailing commas)
# ---------------------------------------------------------------------------

def load_relaxed_json(path: Path) -> dict:
    """Load a JSON file, tolerating trailing commas.

    Args:
        path: Path to the JSON file.

    Returns:
        Parsed JSON object as a dictionary.
    """
    s = path.read_text(errors="ignore")
    s = re.sub(r",\s*(\]|\})", r"\1", s)  # tolerate trailing commas
    return json.loads(s)


# ---------------------------------------------------------------------------
# Canonicalization
# ---------------------------------------------------------------------------

DEFAULT_ALIASES: Dict[str, str] = {
    "commode": "toilet",
    "wc": "toilet",
    "lavatory": "toilet",
    "basin": "sink",
    "washbasin": "sink",
    "cupboard": "cabinet",
    "closet": "cabinet",
    "counter": "countertop",
}


def parse_aliases(alias_str: str) -> Dict[str, str]:
    """Parse an alias string into a ``{source: target}`` dictionary.

    Supports two formats: JSON object (``'{"a":"b"}'``) or comma-separated
    key=value pairs (``'a=b,c=d'``).

    Args:
        alias_str: Raw alias specification string.

    Returns:
        Lower-cased alias mapping.
    """
    s = (alias_str or "").strip()
    if not s:
        return {}
    if s.startswith("{"):
        try:
            obj = json.loads(s)
            return {str(k).strip().lower(): str(v).strip().lower() for k, v in obj.items()}
        except Exception:
            return {}
    out: Dict[str, str] = {}
    for part in s.split(","):
        part = part.strip()
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        out[k.strip().lower()] = v.strip().lower()
    return out


def canon_label(label: str, aliases: Dict[str, str]) -> str:
    """Canonicalize a label string using alias mappings.

    Strips, lowercases, collapses whitespace, and applies *aliases*.

    Args:
        label: Raw label string.
        aliases: Alias mapping from ``parse_aliases`` or ``DEFAULT_ALIASES``.

    Returns:
        Canonical label string.
    """
    l = (label or "").strip().lower().replace("_", " ")
    l = re.sub(r"\s+", " ", l)
    return aliases.get(l, l)


def canon_relation(rel: str) -> str:
    """Canonicalize a relation string to underscore-separated form.

    Args:
        rel: Raw relation string.

    Returns:
        Canonical relation string (e.g. ``"in_front_of"``).
    """
    r = (rel or "").strip().lower().replace(" ", "_")
    r = re.sub(r"_+", "_", r)
    return r


def relation_to_phrase(rel: str) -> str:
    """Convert a relation code to a human-readable phrase.

    Examples: ``"close_by"`` → ``"close to"``, ``"in_front_of"`` →
    ``"in front of"``.

    Args:
        rel: Canonical relation code.

    Returns:
        Human-readable phrase.
    """
    r = canon_relation(rel)
    if r == "close_by":
        return "close to"
    if r == "in_front_of":
        return "in front of"
    return r.replace("_", " ")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class FrameInfo:
    """A single parsed scene frame with pose and semantics.

    Attributes:
        frame_id: Unique frame identifier string.
        position: 3D camera position (from cam-to-world matrix).
        direction: Unit forward direction (3rd column of rotation matrix).
        visible_labels: Set of canonical label strings visible in this frame.
        rel_triples: Set of ``(subject, relation, object)`` canonical tuples.
    """

    frame_id: str
    position: np.ndarray
    direction: np.ndarray
    visible_labels: Set[str]
    rel_triples: Set[Tuple[str, str, str]]


@dataclass
class SceneData:
    """Loaded scene data: frames with stacked pose arrays and index.

    Attributes:
        frames: List of all parsed ``FrameInfo`` objects.
        frame_pos: Stacked frame positions, shape ``(F, 3)``.
        frame_dir: Stacked frame directions, shape ``(F, 3)``.
        frame_id_to_idx: Mapping from frame ID string to list index.
    """

    frames: List[FrameInfo]
    frame_pos: np.ndarray
    frame_dir: np.ndarray
    frame_id_to_idx: Dict[str, int]


# ---------------------------------------------------------------------------
# Pose / frame-ID helpers
# ---------------------------------------------------------------------------

def pose_to_pos_dir(
    scene_pose: Sequence[Sequence[float]],
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract camera position and forward direction from a 4x4 cam-to-world.

    Args:
        scene_pose: 4x4 camera-to-world matrix (nested sequence).

    Returns:
        Tuple of ``(position, direction)`` as float32 arrays of shape ``(3,)``.
    """
    M = np.array(scene_pose, dtype=np.float32)
    pos = M[:3, 3].astype(np.float32)
    direction = M[:3, 2].astype(np.float32)
    n = float(np.linalg.norm(direction))
    if n > 1e-6:
        direction = direction / n
    return pos, direction


def normalize_frame_id(e: dict) -> str:
    """Derive a canonical frame ID string from a description entry.

    Checks ``image_index`` (int or numeric string → ``frame-NNNNNN``),
    then ``frame_id``, falling back to the raw ``image_index`` value.

    Args:
        e: Single entry dictionary from ``all_descriptions*.json``.

    Returns:
        Normalised frame ID string.
    """
    ii = e.get("image_index", "")
    if isinstance(ii, int):
        return f"frame-{ii:06d}"
    if isinstance(ii, str) and ii.isdigit():
        return f"frame-{int(ii):06d}"
    if "frame_id" in e and e["frame_id"] is not None and str(e["frame_id"]).strip():
        return str(e["frame_id"])
    return str(ii)


# ---------------------------------------------------------------------------
# Scene loading
# ---------------------------------------------------------------------------

def load_scene_data(
    dataset_root: Path,
    scene_id: str,
    aliases: Dict[str, str],
) -> SceneData:
    """Load and parse scene description data from disk.

    Reads ``all_descriptions*.json`` from the scene's output directory,
    extracts poses, visible labels, and spatial relations for each frame,
    and returns them as a ``SceneData`` instance.

    Args:
        dataset_root: Root of the dataset (e.g. ``/path/to/3RScan``).
        scene_id: Scene identifier sub-directory.
        aliases: Label alias mapping for canonicalization.

    Returns:
        Parsed ``SceneData`` with all usable frames.

    Raises:
        FileNotFoundError: If no description JSON is found.
        ValueError: If the JSON is not a list.
        RuntimeError: If no usable frames could be parsed.
    """
    scene_dir = dataset_root / scene_id
    desc_dir = scene_dir / "output" / "descriptions"
    desc_path = desc_dir / "all_descriptions.json"
    if not desc_path.exists():
        alts = list(desc_dir.glob("all_descriptions*.json"))
        if not alts:
            raise FileNotFoundError(f"Missing all_descriptions*.json in {desc_dir}")
        desc_path = alts[0]

    entries = json.loads(desc_path.read_text(errors="ignore"))
    if not isinstance(entries, list):
        raise ValueError(f"Expected list JSON in {desc_path}")

    frames: List[FrameInfo] = []
    for e in entries:
        if not isinstance(e, dict) or "scene_pose" not in e:
            continue

        frame_id = normalize_frame_id(e)
        try:
            pos, direction = pose_to_pos_dir(e["scene_pose"])
        except Exception:
            continue

        visible_labels: Set[str] = set()
        vo = e.get("visible_objects", {})
        if isinstance(vo, dict):
            for _oid_str, meta in vo.items():
                if isinstance(meta, dict) and "label" in meta:
                    visible_labels.add(canon_label(str(meta["label"]), aliases))

        rel_triples: Set[Tuple[str, str, str]] = set()
        sr = e.get("spatial_relations", [])
        if isinstance(sr, list):
            for r in sr:
                if not isinstance(r, dict):
                    continue
                subj = canon_label(str(r.get("subject", "")), aliases)
                obj = canon_label(str(r.get("object", "")), aliases)
                rel = canon_relation(str(r.get("relation", "")))
                if subj and obj and rel:
                    rel_triples.add((subj, rel, obj))

        frames.append(FrameInfo(frame_id, pos, direction, visible_labels, rel_triples))

    if not frames:
        raise RuntimeError(f"No usable frames parsed from {desc_path}")

    frame_pos = np.stack([f.position for f in frames], axis=0).astype(np.float32)
    frame_dir = np.stack([f.direction for f in frames], axis=0).astype(np.float32)
    frame_id_to_idx = {f.frame_id: i for i, f in enumerate(frames)}
    return SceneData(frames, frame_pos, frame_dir, frame_id_to_idx)
