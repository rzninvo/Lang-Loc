"""Frame semantic extraction for the dialogue system.

Extracts label salience dictionaries and spatial relation sets from frame
objects produced by the ``dsf`` (pose_level_dialogue_semantic_fallback)
module.  Handles multiple attribute naming conventions robustly.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Set, Tuple

import pose_level_dialogue_semantic_fallback as dsf

from src.dialogue.math_utils import _to_prob01


def frame_label_salience(fr: Any) -> Dict[str, float]:
    """Extract a ``label → salience`` mapping from a frame object.

    Tries several attribute names (``visible_labels``, ``visible_objects``,
    ``labels``, ``label_set``, ``objects``) and several value formats
    (``float``, ``dict`` with ``pixel_percent``/``score``, or plain list).

    Args:
        fr: Frame object with label attributes.

    Returns:
        Dictionary mapping lower-case label strings to salience values in
        ``[0, 1]``.  Returns ``1.0`` for labels whose salience is not
        available.
    """
    # Case 1: visible_labels is dict label->score
    if hasattr(fr, "visible_labels"):
        v = getattr(fr, "visible_labels")
        if isinstance(v, dict):
            out: Dict[str, float] = {}
            for k, val in v.items():
                lab = str(k).strip().lower()
                if isinstance(val, (int, float)):
                    out[lab] = _to_prob01(val)
                elif isinstance(val, dict):
                    # try pixel_percent or score
                    if "pixel_percent" in val:
                        out[lab] = _to_prob01(val["pixel_percent"])
                    elif "score" in val:
                        out[lab] = _to_prob01(val["score"])
                    else:
                        out[lab] = 1.0
                else:
                    out[lab] = 1.0
            return out
        if isinstance(v, (list, tuple, set)):
            return {str(x).strip().lower(): 1.0 for x in v}

    # Case 2: visible_objects might be dict id->info{label,pixel_percent}
    if hasattr(fr, "visible_objects"):
        v = getattr(fr, "visible_objects")
        if isinstance(v, dict):
            out = {}
            for _, info in v.items():
                if isinstance(info, dict) and "label" in info:
                    lab = str(info["label"]).strip().lower()
                    px = info.get("pixel_percent", info.get("score", 1.0))
                    out[lab] = _to_prob01(px)
            if out:
                return out

    # Fallback: labels list
    for name in ("labels", "label_set", "objects"):
        if hasattr(fr, name):
            v = getattr(fr, name)
            if isinstance(v, (list, tuple, set)):
                return {str(x).strip().lower(): 1.0 for x in v}
            if isinstance(v, dict):
                out = {}
                for _, info in v.items():
                    if isinstance(info, dict) and "label" in info:
                        out[str(info["label"]).strip().lower()] = 1.0
                if out:
                    return out

    return {}


def _rel_to_tuple(rel_item: Any) -> Optional[Tuple[str, str, str]]:
    """Parse a relation item into a ``(subject, relation, object)`` tuple.

    Handles attribute-based objects, dictionaries with various key names,
    and tuple/list representations.

    Args:
        rel_item: Relation item in any supported format.

    Returns:
        Normalised lower-case tuple, or ``None`` if parsing fails.
    """
    if rel_item is None:
        return None
    if hasattr(rel_item, "subj") and hasattr(rel_item, "rel") and hasattr(rel_item, "obj"):
        return (str(rel_item.subj).strip().lower(), str(rel_item.rel).strip().lower(), str(rel_item.obj).strip().lower())
    if isinstance(rel_item, dict):
        s = rel_item.get("subj") or rel_item.get("subject")
        r = rel_item.get("rel") or rel_item.get("relation") or rel_item.get("predicate")
        o = rel_item.get("obj") or rel_item.get("object")
        if s and r and o:
            return (str(s).strip().lower(), str(r).strip().lower(), str(o).strip().lower())
        return None
    if isinstance(rel_item, (tuple, list)) and len(rel_item) >= 3:
        if isinstance(rel_item[1], str):
            return (str(rel_item[0]).strip().lower(), str(rel_item[1]).strip().lower(), str(rel_item[2]).strip().lower())
        if isinstance(rel_item[2], str):
            return (str(rel_item[0]).strip().lower(), str(rel_item[2]).strip().lower(), str(rel_item[1]).strip().lower())
    return None


def rel_item_to_tuple(rel_item: Any) -> Tuple[str, str, str]:
    """Parse a relation item, raising on failure.

    Args:
        rel_item: Relation item in any supported format.

    Returns:
        ``(subject, relation, object)`` tuple with lower-case strings.

    Raises:
        ValueError: If *rel_item* cannot be parsed.
    """
    t = _rel_to_tuple(rel_item)
    if t is None:
        raise ValueError(f"Could not parse relation item: {rel_item}")
    return t


def frame_relations(fr: Any) -> Set[Tuple[str, str, str]]:
    """Extract the set of spatial relations from a frame object.

    Tries attribute names ``rels``, ``relations``, and
    ``spatial_relations``.

    Args:
        fr: Frame object with relation attributes.

    Returns:
        Set of ``(subject, relation, object)`` tuples.
    """
    for name in ("rels", "relations", "spatial_relations"):
        if hasattr(fr, name):
            v = getattr(fr, name)
            if isinstance(v, dict):
                out: Set[Tuple[str, str, str]] = set()
                for _, it in v.items():
                    t = _rel_to_tuple(it)
                    if t:
                        out.add(t)
                return out
            if isinstance(v, (list, tuple, set)):
                out = set()
                for it in v:
                    t = _rel_to_tuple(it)
                    if t:
                        out.add(t)
                return out
    return set()


def relation_phrase(rel: str) -> str:
    """Convert a relation code to a human-readable phrase.

    Delegates to ``dsf.relation_to_phrase`` when available, falling back to
    returning the raw code.

    Args:
        rel: Relation code string.

    Returns:
        Human-readable phrase for the relation.
    """
    if hasattr(dsf, "relation_to_phrase"):
        try:
            return dsf.relation_to_phrase(rel)
        except Exception:
            pass
    return rel
