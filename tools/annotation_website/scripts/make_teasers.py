#!/usr/bin/env python3
"""Generate teaser composite images from prepared keyframes.

For each dataset, picks 3 representative scenes (the easiest 3 by
difficulty_rank) and pulls one keyframe from each, then stitches them
into a 16:9 montage at static/img/teaser_<dataset>.jpg.

Run after prepare_keyframes.py + compute_difficulty.py.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True, choices=["3rscan", "scannet"])
    p.add_argument("--keyframes-root", type=Path, required=True)
    p.add_argument("--scenes-json", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--size", default="1600x900",
                   help="output resolution WxH; default 1600x900")
    return p.parse_args()


def _load_pool(path: Path) -> list:
    pool = json.loads(path.read_text())
    return pool["scenes"]


def _pick_three(scenes: list) -> list:
    """Three scenes spread across the difficulty rank: easiest, middle,
    near-easiest+1. We avoid the hardest scenes since they tend to be
    visually messy frames that don't read as a teaser."""
    by_rank = sorted(scenes, key=lambda s: s.get("difficulty_rank", 9999))
    n = len(by_rank)
    if n == 0:
        return []
    if n == 1:
        return [by_rank[0]] * 3
    picks = [by_rank[0], by_rank[min(n // 4, n - 1)], by_rank[min(n // 2, n - 1)]]
    return picks


def _frame_path(keyframes_root: Path, scene: dict) -> Path:
    """Return path to a single keyframe of this scene to put in the teaser.
    Pick the middle frame so we avoid the often-blurry start/end."""
    frames = scene.get("frames", [])
    if not frames:
        raise RuntimeError(f"scene {scene.get('scene_id')} has no frames")
    pick = frames[len(frames) // 2]
    rel = pick["image_path"]  # 'keyframes/<dataset>/<scene_id>/<frame_id>.jpg'
    # keyframes_root is the static/keyframes folder, which is the bit
    # AFTER 'keyframes/' in the relative path. Strip the prefix.
    parts = Path(rel).parts
    # parts ~ ('keyframes', '<dataset>', '<scene_id>', '<frame>.jpg')
    return keyframes_root.joinpath(*parts[1:])


def _crop_cover(im: Image.Image, w: int, h: int) -> Image.Image:
    """Object-fit: cover. Keeps aspect by cropping the longer side."""
    ar_t = w / h
    ar_s = im.width / im.height
    if ar_s > ar_t:
        # source wider; crop sides
        new_w = int(round(im.height * ar_t))
        x = (im.width - new_w) // 2
        im = im.crop((x, 0, x + new_w, im.height))
    else:
        new_h = int(round(im.width / ar_t))
        y = (im.height - new_h) // 2
        im = im.crop((0, y, im.width, y + new_h))
    return im.resize((w, h), Image.LANCZOS)


def main() -> None:
    args = parse_args()
    out_w, out_h = (int(x) for x in args.size.lower().split("x"))
    scenes = _load_pool(args.scenes_json)
    picks = _pick_three(scenes)
    if not picks:
        print(f"[ERROR] no scenes in {args.scenes_json}; cannot generate teaser", file=sys.stderr)
        sys.exit(1)

    # 3 panels side by side, each 1/3 of the width with a 4 px gap
    gap = 4
    panel_w = (out_w - 2 * gap) // 3
    panel_h = out_h
    canvas = Image.new("RGB", (out_w, out_h), (10, 10, 18))

    x = 0
    for i, sc in enumerate(picks):
        try:
            p = _frame_path(args.keyframes_root, sc)
        except Exception as e:
            print(f"[WARN] {e}; skipping", file=sys.stderr)
            x += panel_w + gap
            continue
        if not p.exists():
            print(f"[WARN] keyframe missing on disk: {p}", file=sys.stderr)
            x += panel_w + gap
            continue
        with Image.open(p) as im:
            im = im.convert("RGB")
            im = _crop_cover(im, panel_w, panel_h)
            canvas.paste(im, (x, 0))
        x += panel_w + gap

    # subtle vignette so the dataset name (stamped over by template) reads
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    ovd = ImageDraw.Draw(overlay)
    # bottom gradient
    for i in range(80):
        alpha = int(160 * (i / 80))
        ovd.rectangle(
            [(0, out_h - 80 + i), (out_w, out_h - 80 + i + 1)],
            fill=(0, 0, 0, alpha),
        )
    canvas = Image.alpha_composite(canvas.convert("RGBA"), overlay).convert("RGB")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.out, format="JPEG", quality=86, optimize=True, progressive=True)
    print(f"[{args.dataset}] wrote teaser to {args.out} ({out_w}x{out_h})")


if __name__ == "__main__":
    main()
