#!/usr/bin/env python3
"""Resize keyframes for the annotation pool of one dataset and copy them
into ``static/keyframes/<dataset>/<scene_id>/<frame_id>.jpg``.

Source layout:
    3RScan: data/3RScan/<scene_id>/output/color/frame-XXXXXX.jpg
            (captured portrait, stored landscape; we rotate 90° CCW)
    ScanNet: data/scans/<scene_id>/output/color/<NNNNNN>.jpg
            (already correctly oriented landscape)

Selection: by default we take the keyframes already referenced in the
scene's ``output/descriptions/all_descriptions.json`` (these are the same
keyframes the paper's pipeline runs on, so the human descriptions are
head-to-head with the GPT ones).

Usage:

    python scripts/prepare_keyframes.py \\
        --dataset scannet \\
        --manifest ../../manifests/scannet_table4_first_100.txt \\
        --data-root ../../data/scans \\
        --num-scenes 100 \\
        --out static/keyframes
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

from PIL import Image, ImageOps


_ROTATE_BY_DATASET = {
    "3rscan": True,    # captured portrait, stored landscape
    "scannet": False,  # already correct
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True, choices=["3rscan", "scannet"])
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True,
                   help="root directory for static/keyframes (the dataset subfolder is created automatically)")
    p.add_argument("--num-scenes", type=int, default=100)
    p.add_argument("--ordered", action="store_true",
                   help="take scenes in manifest order (default: shuffle with seed)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--long-edge", type=int, default=1024)
    p.add_argument("--quality", type=int, default=85)
    return p.parse_args()


def _load_scene_ids(manifest: Path, num: int, ordered: bool, seed: int) -> List[str]:
    ids = [ln.strip() for ln in manifest.read_text().splitlines() if ln.strip()]
    if not ordered:
        import random
        random.Random(seed).shuffle(ids)
    return ids[: num]


def _list_keyframes(scene_dir: Path) -> List[str]:
    desc_path = scene_dir / "output" / "descriptions" / "all_descriptions.json"
    if not desc_path.exists():
        return []
    data = json.loads(desc_path.read_text())
    out = []
    for entry in data:
        if not isinstance(entry, dict):
            continue
        fid = str(entry.get("image_index") or entry.get("frame_id") or "")
        if fid:
            out.append(fid)
    return out


def _resize_to_jpeg(src: Path, dst: Path, long_edge: int, quality: int, rotate_landscape: bool) -> bool:
    if dst.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as im:
        im = ImageOps.exif_transpose(im)
        if rotate_landscape and im.width > im.height:
            im = im.transpose(Image.ROTATE_270)
        im = im.convert("RGB")
        w, h = im.size
        scale = long_edge / max(w, h)
        if scale < 1.0:
            new = (int(round(w * scale)), int(round(h * scale)))
            im = im.resize(new, Image.LANCZOS)
        im.save(dst, format="JPEG", quality=quality, optimize=True, progressive=True)
    return True


def _frame_filename(dataset: str, frame_id: str) -> str:
    """Convert the frame_id stored in all_descriptions.json into the
    on-disk filename inside ``output/color``.

    3RScan: frame_id is something like ``frame-000042`` and matches the
            filename ``frame-000042.jpg``.
    ScanNet: frame_id is ``001229`` (no prefix) and matches ``001229.jpg``.
    """
    return f"{frame_id}.jpg"


def main() -> None:
    args = parse_args()
    rotate_landscape = _ROTATE_BY_DATASET[args.dataset]
    scene_ids = _load_scene_ids(args.manifest, args.num_scenes, args.ordered, args.seed)
    print(f"[{args.dataset}] selected {len(scene_ids)} scenes from {args.manifest.name}")

    out_root: Path = args.out / args.dataset
    out_root.mkdir(parents=True, exist_ok=True)

    pool: list[dict] = []
    n_total_frames = 0
    n_written = 0
    n_missing_scenes = 0
    n_missing_frames = 0

    for idx, scene_id in enumerate(scene_ids, start=1):
        scene_dir = args.data_root / scene_id
        if not scene_dir.exists():
            print(f"[WARN] scene dir missing: expected={scene_dir}, got=missing, fallback=skip", file=sys.stderr, flush=True)
            n_missing_scenes += 1
            continue

        frames = _list_keyframes(scene_dir)
        if not frames:
            print(f"[WARN] no keyframes for {scene_id} (missing all_descriptions.json), skipping", file=sys.stderr, flush=True)
            n_missing_scenes += 1
            continue

        scene_frames = []
        for fid in frames:
            src_jpg = scene_dir / "output" / "color" / _frame_filename(args.dataset, fid)
            if not src_jpg.exists():
                print(f"[WARN] keyframe image missing: expected={src_jpg}", file=sys.stderr, flush=True)
                n_missing_frames += 1
                continue
            dst_jpg = out_root / scene_id / f"{fid}.jpg"
            if _resize_to_jpeg(src_jpg, dst_jpg, args.long_edge, args.quality, rotate_landscape):
                n_written += 1
            n_total_frames += 1
            scene_frames.append({
                "frame_id": fid,
                "image_path": f"keyframes/{args.dataset}/{scene_id}/{fid}.jpg",
            })

        if not scene_frames:
            print(f"[WARN] {scene_id} ended up with zero usable frames, skipping", file=sys.stderr, flush=True)
            continue

        pool.append({
            "scene_id": scene_id,
            "display_index": idx,
            "frames": scene_frames,
        })

    # The manifest file lives in <website-root>/data/, which is two
    # levels up from <static>/<keyframes>/<dataset>/.
    site_root = out_root.parent.parent.parent  # static/keyframes/<ds> → website root
    pool_path = site_root / "data" / f"scenes_keyframes_{args.dataset}.json"
    pool_path.parent.mkdir(parents=True, exist_ok=True)
    pool_path.write_text(json.dumps({"dataset": args.dataset, "scenes": pool}, indent=2))
    print(
        f"[{args.dataset}] wrote {n_total_frames} frames ({n_written} new) across {len(pool)} scenes; "
        f"manifest at {pool_path}"
    )
    if n_missing_scenes:
        print(f"[WARN] {n_missing_scenes} scenes missing on disk and skipped", file=sys.stderr)
    if n_missing_frames:
        print(f"[WARN] {n_missing_frames} individual keyframes missing", file=sys.stderr)


if __name__ == "__main__":
    main()
