#!/usr/bin/env python3
"""Re-parse natural-language frame descriptions into structured graphs.

The fine-localization paper protocol (paper §3.3, Supp §4.3, Master
§5.5.1) does not consume ``visible_objects`` directly.  Instead, each
frame's natural-language ``description`` (produced by
``generate_descriptions.py``) is re-parsed by GPT-4o-mini back into a
SceneGraph; the parsed nodes are then grounded to ``visible_objects``
via Word2Vec at γ=0.7 to recover centroids.  This mirrors a real-user
input: the system only sees the free-form description, not the
structured ground truth.

This script is the precompute step that produces the
``frame-XXXXXX_parsed.json`` files consumed by
``langloc.localization`` when ``caption_source: parsed`` is set in the
config.

Output schema (one file per frame, written next to the source frame
JSON, with the ``_parsed`` suffix added before ``.json``):

    {
        "scene_index":   <copied from source>,
        "image_index":   <copied from source>,
        "source_frame":  "frame-XXXXXX.json",
        "scene_pose":    <copied from source>,
        "description":   <copied from source>,
        "parsed_graph": {
            "nodes": [
                {"id": int, "label": str, "attributes": [str, ...],
                 "label_word2vec": [float, ...]},
                ...
            ],
            "edges": [
                {"source": int, "target": int, "relationship": str,
                 "relation_word2vec": [float, ...]},
                ...
            ]
        }
    }

Usage::

    python -m langloc.dataset.annotation.parse_descriptions \\
        --data_root data/3RScan \\
        --max_frames 50 --dry_run        # quick smoke test
    python -m langloc.dataset.annotation.parse_descriptions \\
        --data_root data/3RScan          # full precompute (all frames)

A line-by-line port of
``whereami-text2sgm/playground/graph_models/models/
preprocess_descriptions.py`` (mk5's precompute), specialised to our
data layout (``<scene>/output/descriptions/frame-*.json``) and our
shared embedding cache (``langloc.utils.embedding._embed_word2vec``).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm

from langloc.utils.embedding import _embed_word2vec
from langloc.utils.seed import set_seed, CANONICAL_SEED


# ---------------------------------------------------------------------------
# OpenAI client (lazily initialised so `python -m ... --help` works without
# a key)
# ---------------------------------------------------------------------------

_OPENAI_CLIENT = None


def _get_openai_client():
    global _OPENAI_CLIENT
    if _OPENAI_CLIENT is None:
        try:
            from dotenv import load_dotenv
            project_root = Path(__file__).resolve().parents[3]
            dotenv_path = project_root / ".env"
            if dotenv_path.exists():
                load_dotenv(dotenv_path)
        except ImportError:
            pass  # python-dotenv not required when OPENAI_API_KEY is in env
        from openai import OpenAI
        _OPENAI_CLIENT = OpenAI()
    return _OPENAI_CLIENT


# ---------------------------------------------------------------------------
# GPT parser (line-by-line port of mk5 single_inference.parse_text_to_json)
# ---------------------------------------------------------------------------

_PARSE_PROMPT_TEMPLATE = """You are a parser that converts natural language scene descriptions into a JSON graph.
Extract:
- objects (with id, label, attributes if any)
- relationships (edges: source, target, relationship)

Rules:
- Assign each object an integer id starting at 0.
- Each node: {{"id": int, "label": str, "attributes": [str,...]}}
- Each edge: {{"source": int, "target": int, "relationship": str}}
- If no attributes -> "attributes": []
- If no edges -> "edges": []

Example:
Input: "There is a wooden chair next to a table."
Output:
{{
"nodes": [
    {{"id": 0, "label": "chair", "attributes": ["wooden"]}},
    {{"id": 1, "label": "table", "attributes": []}}
],
"edges": [
    {{"source": 0, "target": 1, "relationship": "next to"}}
]
}}

Now process:
"{query_text}"
Only output valid JSON, nothing else."""


def parse_text_to_json(query_text: str,
                       model: str = "gpt-4o-mini",
                       debug: bool = False) -> Dict[str, list]:
    """Parse a natural-language scene description into a node/edge dict.

    Calls GPT-4o-mini (paper default) with a strict JSON-output prompt
    and returns the parsed dict.  Falls back to ``{"nodes": [], "edges":
    []}`` on JSON-decode failure rather than raising — the caller can
    decide whether to skip or retry the frame.

    Args:
        query_text: The frame's natural-language ``description`` field.
        model: OpenAI model name.  Defaults to ``"gpt-4o-mini"`` to
            match the paper's preprocess (``Supp §1.4``).
        debug: When True, prints the raw model output before JSON
            parsing.

    Returns:
        A dict with ``"nodes"`` and ``"edges"`` keys (mk5 schema).
    """
    client = _get_openai_client()
    prompt = _PARSE_PROMPT_TEMPLATE.format(query_text=query_text)
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system",
                 "content": "You are a JSON scene graph extractor."},
                {"role": "user", "content": prompt},
            ],
            # gpt-5.x rejects custom temperature; older gpt-4o-mini
            # accepts temperature=0.  We avoid setting it so the call
            # is portable across model versions.
        )
        raw = response.choices[0].message.content.strip()
    except Exception as exc:
        print(f"[WARN] parse_text_to_json: GPT request failed "
              f"(query={query_text[:60]!r}): {exc}", flush=True)
        return {"nodes": [], "edges": []}

    if debug:
        print(f"  raw: {raw}", flush=True)

    # Strip ``` fences if the model added them despite the prompt.
    cleaned = raw
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        print(f"[WARN] parse_text_to_json: JSON decode failed "
              f"(query={query_text[:60]!r}): {exc}", flush=True)
        return {"nodes": [], "edges": []}
    if not isinstance(parsed, dict):
        return {"nodes": [], "edges": []}
    parsed.setdefault("nodes", [])
    parsed.setdefault("edges", [])
    return parsed


# ---------------------------------------------------------------------------
# Per-frame work
# ---------------------------------------------------------------------------

def _embed_node(node: dict, mode: str) -> dict:
    """Add ``label_word2vec`` (and per-attribute embeddings) in place."""
    label = node.get("label", "") or ""
    node["label_word2vec"] = _embed_word2vec(label, mode=mode)
    attrs = node.get("attributes", []) or []
    node["attributes_word2vec"] = {
        "all": [_embed_word2vec(a, mode=mode) for a in attrs]
    }
    return node


def _embed_edge(edge: dict, mode: str) -> dict:
    rel = edge.get("relationship", "") or ""
    edge["relation_word2vec"] = _embed_word2vec(rel, mode=mode)
    return edge


def process_frame(frame_path: Path,
                  query_embedding_mode: str = "doc",
                  model: str = "gpt-4o-mini",
                  overwrite: bool = False,
                  dry_run: bool = False) -> Tuple[Path, str]:
    """Re-parse a single frame JSON and write its ``*_parsed.json`` sibling.

    Returns a ``(parsed_path, status)`` tuple where status is one of
    ``"written"``, ``"skipped_existing"``, ``"skipped_no_description"``,
    ``"dry_run"``, or ``"error"``.
    """
    parsed_path = frame_path.with_name(frame_path.stem + "_parsed" + frame_path.suffix)
    if parsed_path.exists() and not overwrite:
        return parsed_path, "skipped_existing"

    try:
        frame = json.loads(frame_path.read_text())
    except json.JSONDecodeError:
        return parsed_path, "error"

    description = frame.get("description", "") or ""
    if not description.strip():
        return parsed_path, "skipped_no_description"

    if dry_run:
        return parsed_path, "dry_run"

    parsed_graph = parse_text_to_json(description, model=model)
    for node in parsed_graph.get("nodes", []):
        _embed_node(node, query_embedding_mode)
    for edge in parsed_graph.get("edges", []):
        _embed_edge(edge, query_embedding_mode)

    out = {
        "scene_index": frame.get("scene_index"),
        "image_index": frame.get("image_index"),
        "source_frame": frame_path.name,
        "scene_pose": frame.get("scene_pose"),
        "description": description,
        "parsed_graph": parsed_graph,
    }
    parsed_path.write_text(json.dumps(out, indent=2))
    return parsed_path, "written"


# ---------------------------------------------------------------------------
# Frame discovery
# ---------------------------------------------------------------------------

def discover_frame_jsons(data_root: Path,
                         scene_ids: Optional[List[str]] = None) -> List[Path]:
    """Return all source frame JSON paths under ``data_root``.

    Looks under ``<data_root>/<scene_id>/output/descriptions/frame-*.json``
    and excludes ``*_parsed.json`` byproducts.  When *scene_ids* is
    given, restricts to those scenes; otherwise visits every scene
    directory.
    """
    if scene_ids:
        dirs = [data_root / sid / "output" / "descriptions" for sid in scene_ids]
    else:
        dirs = sorted(data_root.glob("*/output/descriptions"))

    paths: List[Path] = []
    for d in dirs:
        if not d.exists():
            continue
        for p in sorted(d.glob("frame-*.json")):
            if p.stem.endswith("_parsed"):
                continue
            paths.append(p)
    return paths


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_root", required=True, type=Path,
                    help="Dataset root containing <scene_id>/output/descriptions/.")
    ap.add_argument("--scene_ids", nargs="*",
                    help="Optional list of scene IDs to restrict to.")
    ap.add_argument("--max_frames", type=int, default=None,
                    help="Cap on total frames processed (for smoke tests).")
    ap.add_argument("--query_embedding_mode", choices=["token", "doc"],
                    default="doc",
                    help="Word2Vec aggregation mode for the embedded "
                         "parsed nodes/edges.  Paper protocol: 'doc'.")
    ap.add_argument("--model", default="gpt-4o-mini",
                    help="OpenAI model name.  Paper default: gpt-4o-mini.")
    ap.add_argument("--overwrite", action="store_true",
                    help="Re-parse frames even if a *_parsed.json exists.")
    ap.add_argument("--dry_run", action="store_true",
                    help="Discover frames + check for missing descriptions; "
                         "do not call the API or write files.")
    ap.add_argument("--workers", type=int, default=4,
                    help="Number of parallel API workers.")
    ap.add_argument("--seed", type=int, default=CANONICAL_SEED,
                    help="RNG seed (canonical project seed = 42).")
    args = ap.parse_args()

    set_seed(args.seed)

    paths = discover_frame_jsons(args.data_root, args.scene_ids)
    if args.max_frames is not None:
        paths = paths[: args.max_frames]
    print(f"[parse_descriptions] discovered {len(paths)} frame JSONs", flush=True)
    if not paths:
        return 1

    counts: Dict[str, int] = {}
    t0 = time.time()
    if args.dry_run:
        for p in tqdm(paths, desc="dry run"):
            _, status = process_frame(p, args.query_embedding_mode,
                                      args.model, args.overwrite, dry_run=True)
            counts[status] = counts.get(status, 0) + 1
    else:
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
            futures = [
                ex.submit(process_frame, p, args.query_embedding_mode,
                          args.model, args.overwrite, False)
                for p in paths
            ]
            for fut in tqdm(as_completed(futures), total=len(futures),
                            desc="parse_descriptions"):
                try:
                    _, status = fut.result()
                except Exception as exc:
                    print(f"[WARN] worker failed: {exc}", flush=True)
                    status = "error"
                counts[status] = counts.get(status, 0) + 1

    dt = time.time() - t0
    print(f"[parse_descriptions] done in {dt:.1f}s | counts: {counts}",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
