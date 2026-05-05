#!/usr/bin/env python3
"""Generate D-IQA dataset slide figure: rejected vs kept frames + QualiCLIP scores.

Scores a sample of raw RGB frames from one scene with the same QualiCLIP metric
the dataset pipeline uses, then composes a 2-row strip (3 rejected ✗ + 3 kept ✓)
with the score annotated on each thumbnail. The threshold band is drawn between
the two rows.

Outputs ``{output}/iqa_strip_{scan_id}.pdf`` (and a ``.png`` sibling).

Usage::

    python -m tools.viz.fig_supp_iqa \
        --root ./data/scans \
        --scan-id scene0002_00 \
        --threshold 0.45 \
        --output docs/figures
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image


def _set_eccv_rc():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["CMU Serif", "Computer Modern Roman", "Times New Roman",
                        "DejaVu Serif", "serif"],
        "mathtext.fontset": "cm",
        "font.size": 10,
    })


def score_frames(paths, device="cuda", batch_size: int = 8):
    """Score a list of image paths with QualiCLIP. Returns dict[stem -> score]."""
    import pyiqa
    import torch
    from torchvision import transforms

    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"[IQA] Loading qualiclip on {dev}...")
    model = pyiqa.create_metric("qualiclip", device=dev)

    out: dict[str, float] = {}
    with torch.no_grad():
        if batch_size <= 1:
            for p in paths:
                try:
                    s = float(model(str(p)).item())
                    out[Path(p).stem] = s
                except Exception as e:
                    print(f"  [warn] {Path(p).name}: {e}")
        else:
            tt = transforms.ToTensor()
            for i in range(0, len(paths), batch_size):
                chunk = paths[i:i + batch_size]
                try:
                    tensors = []
                    for p in chunk:
                        with Image.open(p) as im:
                            tensors.append(tt(im.convert("RGB")))
                    batch = torch.stack(tensors, dim=0).to(dev)
                    scores = model(batch).view(-1).cpu().tolist()
                    for p, s in zip(chunk, scores):
                        out[Path(p).stem] = float(s)
                except Exception as e:
                    print(f"  [warn] batch starting {Path(chunk[0]).name}: {e}; "
                          "falling back to sequential")
                    for p in chunk:
                        try:
                            out[Path(p).stem] = float(model(str(p)).item())
                        except Exception:
                            pass
                if (i // batch_size) % 5 == 0:
                    print(f"  scored {min(i + batch_size, len(paths))}/{len(paths)}")
    return out


def _load_thumb(path: Path, target_w: int = 480) -> np.ndarray:
    """Load + resize a frame for display."""
    with Image.open(path) as im:
        im = im.convert("RGB")
        w, h = im.size
        scale = target_w / w
        im = im.resize((target_w, int(h * scale)), Image.LANCZOS)
        return np.asarray(im)


def render_strip(rejected, kept, threshold: float, output_path: Path,
                 title_h_in: float = 0.3, dpi: int = 300) -> None:
    """rejected / kept : list of (path, score) tuples (length 3 each)."""
    _set_eccv_rc()

    n_cols = 3
    fig, axes = plt.subplots(2, n_cols,
                             figsize=(n_cols * 3.8, 2 * 2.6 + title_h_in * 2),
                             dpi=dpi,
                             gridspec_kw={"hspace": 0.18, "wspace": 0.06})

    badge_red = "#d32f2f"
    badge_green = "#2e7d32"

    def _panel(ax, path, score, kept_flag):
        img = _load_thumb(path, target_w=480)
        ax.imshow(img)
        ax.set_axis_off()
        # Border
        for spine in ax.spines.values():
            spine.set_visible(False)
        # Score annotation in top-left
        s_text = f"{score:.3f}"
        ax.text(0.04, 0.94, s_text, transform=ax.transAxes,
                fontsize=11, fontweight="bold",
                color="white",
                bbox=dict(boxstyle="round,pad=0.25",
                          fc=(0.0, 0.0, 0.0, 0.7), ec="none"),
                verticalalignment="top", horizontalalignment="left",
                zorder=10)
        # ✓ / ✗ badge in top-right (use DejaVu Sans — DejaVu Serif lacks these glyphs)
        badge = "✓" if kept_flag else "✗"
        bg = badge_green if kept_flag else badge_red
        ax.text(0.96, 0.94, badge, transform=ax.transAxes,
                fontsize=14, fontweight="bold", color="white",
                family="DejaVu Sans",
                bbox=dict(boxstyle="circle,pad=0.20",
                          fc=bg, ec="none"),
                verticalalignment="top", horizontalalignment="right",
                zorder=10)

    for i, (path, score) in enumerate(rejected):
        _panel(axes[0, i], path, score, kept_flag=False)
    for i, (path, score) in enumerate(kept):
        _panel(axes[1, i], path, score, kept_flag=True)

    # Row labels on the left edge
    fig.text(0.012, 0.74, "Rejected",
             rotation=90, fontsize=12, fontweight="bold",
             color=badge_red, va="center", ha="center")
    fig.text(0.012, 0.28, "Kept",
             rotation=90, fontsize=12, fontweight="bold",
             color=badge_green, va="center", ha="center")

    # Threshold band annotation between rows
    fig.text(0.5, 0.503,
             f"QualiCLIP threshold $\\tau \\approx {threshold:.2f}$",
             fontsize=10, style="italic", color="#555555",
             ha="center", va="center",
             bbox=dict(boxstyle="round,pad=0.4",
                       fc=(1, 1, 1, 0.85),
                       ec="#bbbbbb", lw=0.6))

    fig.subplots_adjust(left=0.035, right=0.99, top=0.98, bottom=0.02)
    fig.savefig(output_path, format="pdf", dpi=dpi,
                bbox_inches="tight", pad_inches=0.05)
    fig.savefig(output_path.with_suffix(".png"), format="png", dpi=dpi,
                bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def parse_args():
    ap = argparse.ArgumentParser(description="IQA strip figure (D-IQA slide).")
    ap.add_argument("--root", type=Path, required=True,
                    help="Dataset root (e.g. ./data/scans).")
    ap.add_argument("--scan-id", "--scan_id", dest="scan_id", required=True)
    ap.add_argument("--output", type=Path, default=Path("docs/figures"))
    ap.add_argument("--threshold", type=float, default=0.45,
                    help="QualiCLIP threshold annotation on the figure.")
    ap.add_argument("--reject-sample", "--reject_sample",
                    dest="reject_sample", type=int, default=80,
                    help="How many non-survivor frames to score for picking rejects.")
    ap.add_argument("--keep-from", choices=["survivors", "all"],
                    default="survivors",
                    help="Where to pick the 'kept' examples from. "
                         "'survivors' = the 69 cache survivors (matches pipeline). "
                         "'all' = top 3 across the whole sample.")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rescore", action="store_true",
                    help="Force re-scoring even if {output}/iqa_scores_*.json exists.")
    ap.add_argument("--max-rejects", "--max_rejects",
                    dest="max_rejects", type=int, default=3)
    ap.add_argument("--max-keeps", "--max_keeps",
                    dest="max_keeps", type=int, default=3)
    return ap.parse_args()


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    scan_dir = args.root / args.scan_id
    color_dir = scan_dir / "color"
    if not color_dir.exists():
        raise FileNotFoundError(f"Missing color dir: {color_dir}")

    all_frames = sorted(color_dir.glob("*.jpg"))
    print(f"[1/4] {len(all_frames)} raw frames in {color_dir}")

    # Load cached survivors
    cache_path = scan_dir / "output" / "cache" / f"{args.scan_id}.json"
    cache = json.loads(cache_path.read_text())
    survivor_ids = {str(f["fid"]) for f in cache}
    print(f"[2/4] {len(survivor_ids)} survivors in {cache_path.name}")

    # Sample non-survivor frames to score for the reject pool
    non_survivors = [p for p in all_frames if p.stem not in survivor_ids]
    sample_n = min(args.reject_sample, len(non_survivors))
    reject_sample = rng.sample(non_survivors, sample_n)
    survivor_paths = [p for p in all_frames if p.stem in survivor_ids]

    paths_to_score = list(reject_sample) + list(survivor_paths)
    cache_scores = args.output / f"iqa_scores_{args.scan_id}.json"
    if cache_scores.exists() and not args.rescore:
        scores = json.loads(cache_scores.read_text())
        print(f"[3/4] Loaded {len(scores)} cached IQA scores from {cache_scores}")
    else:
        print(f"[3/4] Scoring {len(paths_to_score)} frames "
              f"({sample_n} non-survivors + {len(survivor_paths)} survivors)...")
        scores = score_frames(paths_to_score, device=args.device)
        cache_scores.write_text(json.dumps(scores, indent=2))
        print(f"  Cached IQA scores: {cache_scores}")

    # Pick rejects (lowest scores among non-survivors)
    reject_scored = sorted(
        ((p, scores[p.stem]) for p in reject_sample if p.stem in scores),
        key=lambda x: x[1])
    rejected = reject_scored[:args.max_rejects]

    # Pick keeps (highest scores among survivors, by default)
    if args.keep_from == "survivors":
        keep_scored = sorted(
            ((p, scores[p.stem]) for p in survivor_paths if p.stem in scores),
            key=lambda x: -x[1])
    else:
        keep_scored = sorted(
            ((p, scores[p.stem]) for p in paths_to_score if p.stem in scores),
            key=lambda x: -x[1])
    kept = keep_scored[:args.max_keeps]

    print("  Rejected:")
    for p, s in rejected:
        print(f"    {p.stem}  s={s:.3f}")
    print("  Kept:")
    for p, s in kept:
        print(f"    {p.stem}  s={s:.3f}")

    # Render
    print(f"[4/4] Rendering IQA strip...")
    out_pdf = args.output / f"iqa_strip_{args.scan_id}.pdf"
    render_strip(rejected, kept, threshold=args.threshold, output_path=out_pdf)
    print(f"  Saved: {out_pdf}")
    print(f"  Saved: {out_pdf.with_suffix('.png')}")
    print("Done.")


if __name__ == "__main__":
    main()
