#!/usr/bin/env python3
"""
preprocess_descriptions.py
==========================
Batch-parse frame description texts via GPT-4o-mini and cache the resulting
scene graphs (with word2vec embeddings) as ``frame-XXXXXX_parsed.json`` files
alongside the original ``frame-XXXXXX.json`` files.

Usage:
    python preprocess_descriptions.py \
        --data_root /path/to/3RScan_processed \
        --api_key_file /path/to/openai_key.txt \
        --max_frames 5 --dry_run

Or set OPENAI_API_KEY and omit --api_key_file:
    export OPENAI_API_KEY=sk-...
    python preprocess_descriptions.py --data_root /path/to/3RScan_processed
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Tuple

import openai
from tqdm import tqdm

from langloc.graph_matching.single_inference import parse_text_to_json
from langloc.utils.embedding import _embed_word2vec


# --------------------------------------------------------------------------- #
# Discover frame JSONs                                                        #
# --------------------------------------------------------------------------- #

def discover_frame_jsons(data_root: Path,
                         scene_ids: Optional[List[str]] = None) -> List[Path]:
    """Return all frame-*.json paths (excluding *_parsed.json) under data_root."""
    if scene_ids:
        dirs = []
        for sid in scene_ids:
            d = data_root / sid / "output" / "descriptions"
            if d.exists():
                dirs.append(d)
    else:
        dirs = sorted(data_root.glob("*/output/descriptions"))

    paths: List[Path] = []
    for d in dirs:
        for p in sorted(d.glob("frame-*.json")):
            if p.stem.endswith("_parsed"):
                continue
            paths.append(p)
    return paths


def parsed_path_for(frame_path: Path) -> Path:
    """Return the *_parsed.json path corresponding to a frame JSON."""
    return frame_path.with_name(frame_path.stem + "_parsed.json")


# --------------------------------------------------------------------------- #
# Process a single frame (split into GPT call + embedding for concurrency)    #
# --------------------------------------------------------------------------- #

def _gpt_parse_frame(frame_path: Path,
                     max_retries: int = 3) -> Optional[Tuple[Path, dict, str, str, str]]:
    """Read frame JSON and call GPT to parse the description (thread-safe).

    Returns (frame_path, parsed_graph, description, scene_index, image_index)
    or None if the frame has no description.
    """
    data = json.loads(frame_path.read_text())
    description = data.get("description", "")
    if not description.strip():
        return None

    scene_index = data.get("scene_index", "")
    image_index = data.get("image_index", frame_path.stem)

    # Call GPT with retry on rate-limit / transient errors
    last_exc = None
    for attempt in range(max_retries):
        try:
            parsed = parse_text_to_json(description)
            return (frame_path, parsed, description, scene_index, image_index)
        except openai.RateLimitError as exc:
            last_exc = exc
            wait = 2 ** attempt
            print(f"  [RATE LIMIT] {frame_path.name} – retrying in {wait}s …")
            time.sleep(wait)
        except openai.APIError as exc:
            last_exc = exc
            wait = 2 ** attempt
            print(f"  [API ERROR] {frame_path.name} – retrying in {wait}s …")
            time.sleep(wait)
    raise last_exc  # type: ignore[misc]


def _embed_parsed_graph(parsed: dict, embedding_mode: str) -> None:
    """Add word2vec embeddings to parsed graph nodes/edges (NOT thread-safe)."""
    for node in parsed.get("nodes", []):
        node["label_word2vec"] = _embed_word2vec(node["label"], mode=embedding_mode)
        node["attributes_word2vec"] = {
            "all": [_embed_word2vec(a, mode=embedding_mode)
                    for a in node.get("attributes", [])]
        }
    for edge in parsed.get("edges", []):
        edge["relation_word2vec"] = _embed_word2vec(
            edge["relationship"], mode=embedding_mode
        )


def process_frame(frame_path: Path,
                  embedding_mode: str,
                  dry_run: bool = False) -> Optional[dict]:
    """Parse description via GPT, embed with word2vec, return parsed dict."""
    if dry_run:
        data = json.loads(frame_path.read_text())
        description = data.get("description", "")
        scene_index = data.get("scene_index", "")
        if not description.strip():
            print(f"  [SKIP] No description in {frame_path.name}")
            return None
        print(f"  [DRY RUN] Would parse: {frame_path.name} "
              f"(scene={scene_index}, desc length={len(description)})")
        return None

    result_tuple = _gpt_parse_frame(frame_path)
    if result_tuple is None:
        print(f"  [SKIP] No description in {frame_path.name}")
        return None

    _, parsed, description, scene_index, image_index = result_tuple
    _embed_parsed_graph(parsed, embedding_mode)

    return {
        "source_frame": image_index,
        "scene_index": scene_index,
        "description": description,
        "parsed_graph": parsed,
    }


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Preprocess frame descriptions via GPT and cache parsed scene graphs."
    )
    p.add_argument("--data_root", required=True, type=Path,
                   help="Root of 3RScan_processed (contains <scene_id>/output/descriptions/).")
    p.add_argument("--api_key_file", type=Path,
                   help="Optional path to file with line 'OPENAI_API_KEY=sk-...' or just the key. "
                        "If omitted, OPENAI_API_KEY env var is used.")
    p.add_argument("--embedding_mode", choices=["token", "doc"], default="token",
                   help="word2vec embedding mode: 'token' (first-token) or 'doc' (spaCy doc.vector).")
    p.add_argument("--scene_ids", nargs="+",
                   help="Optional list of scene IDs to restrict processing.")
    p.add_argument("--dry_run", action="store_true",
                   help="Preview which frames would be processed without calling GPT.")
    p.add_argument("--overwrite", action="store_true",
                   help="Re-process frames even if *_parsed.json already exists.")
    p.add_argument("--max_frames", type=int,
                   help="Limit the number of frames to process (for testing).")
    p.add_argument("--workers", type=int, default=8,
                   help="Number of parallel threads for GPT API calls (default: 8). "
                        "Set to 1 for sequential processing.")
    p.add_argument("--batch_size", type=int, default=32,
                   help="Number of frames per batch (default: 32). Each batch is "
                        "GPT-parsed concurrently, then embedded and saved before "
                        "starting the next batch.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Load API key from optional file, otherwise OPENAI_API_KEY.
    key = ""
    if args.api_key_file is not None:
        with open(args.api_key_file, "r") as f:
            line = f.read().strip()
            if line.startswith("OPENAI_API_KEY="):
                key = line.split("=", 1)[1]
            else:
                key = line
    else:
        key = os.getenv("OPENAI_API_KEY", "").strip()

    if not key:
        raise ValueError(
            "No OpenAI API key found. Set OPENAI_API_KEY or pass --api_key_file."
        )

    openai.api_key = key

    # Discover frames
    all_frames = discover_frame_jsons(args.data_root, args.scene_ids)
    print(f"Found {len(all_frames)} frame JSON(s) under {args.data_root}")

    # Filter already-processed (unless --overwrite)
    if not args.overwrite:
        todo = [p for p in all_frames if not parsed_path_for(p).exists()]
        skipped = len(all_frames) - len(todo)
        if skipped:
            print(f"Skipping {skipped} already-parsed frame(s) (use --overwrite to redo).")
    else:
        todo = all_frames

    if args.max_frames is not None:
        todo = todo[: args.max_frames]

    workers = max(1, args.workers)
    print(f"Processing {len(todo)} frame(s) with {workers} worker(s)...\n")

    t_start = time.monotonic()
    success = 0
    errors = 0
    failed_scans = set()

    if args.dry_run or workers == 1:
        # --- Sequential path (dry-run or single worker) ---
        for frame_path in tqdm(todo, desc="Processing", unit="frame"):
            scene_id = frame_path.parts[-4]
            try:
                result = process_frame(frame_path, args.embedding_mode, dry_run=args.dry_run)
            except Exception as exc:
                tqdm.write(f"  [ERROR] {scene_id}/{frame_path.name}: {exc}")
                errors += 1
                failed_scans.add(scene_id)
                continue
            if result is None:
                continue
            out_path = parsed_path_for(frame_path)
            out_path.write_text(json.dumps(result, indent=2))
            success += 1
    else:
        # --- Parallel path: process in batches for crash resilience ---
        batch_size = max(1, args.batch_size)
        n_batches = (len(todo) + batch_size - 1) // batch_size
        overall_pbar = tqdm(total=len(todo), desc="Overall", unit="frame")

        for batch_idx in range(n_batches):
            batch_start = batch_idx * batch_size
            batch = todo[batch_start : batch_start + batch_size]
            tqdm.write(f"\n--- Batch {batch_idx + 1}/{n_batches} "
                       f"({len(batch)} frames) ---")

            # Phase 1: GPT-parse this batch concurrently
            future_to_path = {}
            with ThreadPoolExecutor(max_workers=workers) as executor:
                for frame_path in batch:
                    fut = executor.submit(_gpt_parse_frame, frame_path)
                    future_to_path[fut] = frame_path

                gpt_results: List[Tuple[Path, dict, str, str, str]] = []
                for fut in as_completed(future_to_path):
                    frame_path = future_to_path[fut]
                    scene_id = frame_path.parts[-4]
                    try:
                        result_tuple = fut.result()
                    except Exception as exc:
                        tqdm.write(f"  [ERROR] {scene_id}/{frame_path.name}: {exc}")
                        errors += 1
                        failed_scans.add(scene_id)
                        overall_pbar.update(1)
                        continue
                    if result_tuple is None:
                        overall_pbar.update(1)
                        continue
                    gpt_results.append(result_tuple)

            # Phase 2: embed + save this batch (main thread)
            for frame_path, parsed, description, scene_index, image_index in sorted(
                gpt_results, key=lambda r: r[0]
            ):
                _embed_parsed_graph(parsed, args.embedding_mode)
                result = {
                    "source_frame": image_index,
                    "scene_index": scene_index,
                    "description": description,
                    "parsed_graph": parsed,
                }
                out_path = parsed_path_for(frame_path)
                out_path.write_text(json.dumps(result, indent=2))
                success += 1
                overall_pbar.update(1)

            tqdm.write(f"  Batch {batch_idx + 1} saved: {len(gpt_results)} frames")

        overall_pbar.close()

    elapsed = time.monotonic() - t_start
    rate = success / elapsed if elapsed > 0 else 0
    print(f"\nDone in {elapsed:.1f}s ({rate:.1f} frames/sec). "
          f"Success: {success} | Errors: {errors} | "
          f"Skipped: {len(todo) - success - errors}")
    if failed_scans:
        print("Failed scans:")
        for scan_id in sorted(failed_scans):
            print(f"  - {scan_id}")
    else:
        print("Failed scans: none")


if __name__ == "__main__":
    main()
