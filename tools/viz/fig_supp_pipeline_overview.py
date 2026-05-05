#!/usr/bin/env python3
"""Generate D1 dataset slide figure: 4-step LangLoc pipeline overview.

A single wide PDF with four numbered columns (Quality filter,
Visibility rasterisation, Two-stage DPP, Description generation), each
showing a representative thumbnail + one-line caption, plus a dataset stats
strip at the bottom.

Drops directly into a slide as one image — matches the numbered-header style
of the other dataset figures.

Usage::

    python -m tools.viz.fig_supp_pipeline_overview \\
        --root ./data/scans \\
        --scan-id scene0002_00 \\
        --keep-frame 003769 --reject-frame 004516 \\
        --visibility-img ./docs/figures/visibility_isometric_scene0002_00.png \\
        --dpp-img ./docs/figures/supp_dpp_selection.png \\
        --caption-frame 000838 \\
        --output docs/figures
"""
from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from PIL import Image


def _set_pres_rc():
    plt.rcParams.update({
        "font.family": ["Aptos Display", "Aptos", "Inter",
                        "Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 11,
    })


def _load_thumb(path: Path, target_w: int = 1200) -> np.ndarray:
    with Image.open(path) as im:
        im = im.convert("RGB")
        w, h = im.size
        scale = target_w / w
        im = im.resize((target_w, int(h * scale)), Image.LANCZOS)
        return np.asarray(im)


def _draw_step_label(ax, n: int, title: str, *, color="#1d2b53") -> None:
    """Numbered header above a panel: ``01 │ Title``."""
    ax.text(0.0, 1.060, f"{n:02d}", transform=ax.transAxes,
            fontsize=20, fontweight="bold", color=color,
            ha="left", va="center", zorder=10, clip_on=False)
    ax.plot([0.085, 0.085], [1.018, 1.105],
            transform=ax.transAxes,
            color="#bbbbbb", linewidth=0.8, clip_on=False, zorder=10)
    ax.text(0.105, 1.060, title, transform=ax.transAxes,
            fontsize=12.5, fontweight="bold", color="#1c1c1c",
            ha="left", va="center", zorder=10, clip_on=False)


def _make_iqa_thumb(reject_path: Path, keep_path: Path,
                    score_reject: float = 0.15,
                    score_keep: float = 0.66,
                    out_h: int = 1000) -> np.ndarray:
    """Build a small 2-row thumbnail with the rejected ✗ on top, kept ✓ below."""
    badge_red = (211, 47, 47)
    badge_green = (46, 125, 50)
    target_w = int(out_h * 0.85)  # slightly portrait-ish
    half_h = (out_h - 18) // 2

    def _open(p):
        with Image.open(p) as im:
            im = im.convert("RGB")
            w, h = im.size
            scale = target_w / w
            return im.resize((target_w, int(h * scale)), Image.LANCZOS)

    rj = _open(reject_path)
    kp = _open(keep_path)

    # Crop / resize to half_h vertically
    def _fit_v(im, height):
        w, h = im.size
        if h > height:
            crop_top = (h - height) // 2
            return im.crop((0, crop_top, w, crop_top + height))
        else:
            scale = height / h
            return im.resize((int(w * scale), height), Image.LANCZOS)

    rj_fit = _fit_v(rj, half_h)
    kp_fit = _fit_v(kp, half_h)

    canvas = Image.new("RGB", (target_w, out_h), (250, 250, 248))
    canvas.paste(rj_fit, (0, 0))
    canvas.paste(kp_fit, (0, half_h + 18))

    # Score + badge overlays via PIL
    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(canvas)

    # Score chip top-left of each
    def _score_chip(y, score):
        text = f"{score:.2f}"
        # rough chip box
        x0, y0 = 14, y + 12
        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 30)
        except OSError:
            font = ImageFont.load_default()
        bbox = draw.textbbox((x0, y0), text, font=font)
        pad = 8
        draw.rounded_rectangle(
            (bbox[0] - pad, bbox[1] - pad // 2,
             bbox[2] + pad, bbox[3] + pad // 2),
            radius=8, fill=(0, 0, 0))
        draw.text((x0, y0), text, fill=(255, 255, 255), font=font)

    _score_chip(0, score_reject)
    _score_chip(half_h + 18, score_keep)

    # ✗ / ✓ small circles top-right
    def _badge(y, color, glyph):
        cx, cy = target_w - 48, y + 36
        r = 26
        draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=color)
        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 32)
        except OSError:
            font = ImageFont.load_default()
        bbox = draw.textbbox((cx, cy), glyph, font=font, anchor="mm")
        draw.text((cx, cy), glyph, fill=(255, 255, 255), font=font, anchor="mm")

    _badge(0, badge_red, "✗")
    _badge(half_h + 18, badge_green, "✓")

    return np.asarray(canvas)


def _make_caption_thumb(rgb_path: Path, caption: str,
                        out_w: int = 1600,
                        keyframe_h: int = 1100,
                        quote_h: int = 900) -> np.ndarray:
    """Step-4 composite: keyframe on top, pink quote card below with the full
    GPT caption rendered at slide-readable size. The two stack vertically in a
    single PDF page sized for one PowerPoint placeholder.
    """
    pink_fg = (179, 65, 90)
    pink_bg = (253, 234, 239)
    pink_edge = (230, 154, 167)
    text_color = (58, 29, 36)

    gap = 30
    out_h = keyframe_h + gap + quote_h

    canvas = Image.new("RGB", (out_w, out_h), (255, 255, 255))

    # ---- Keyframe (top) ----
    with Image.open(rgb_path) as im:
        im = im.convert("RGB")
        w, h = im.size
        scale = min(out_w / w, keyframe_h / h)
        new_w, new_h = int(round(w * scale)), int(round(h * scale))
        im = im.resize((new_w, new_h), Image.LANCZOS)
        x = (out_w - new_w) // 2
        y = (keyframe_h - new_h) // 2
        canvas.paste(im, (x, y))

    # ---- Pink quote card (bottom) ----
    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(canvas)

    qy0 = keyframe_h + gap
    qy1 = out_h
    margin = 24
    draw.rounded_rectangle(
        (margin, qy0 + margin, out_w - margin, qy1 - margin),
        radius=42, fill=pink_bg, outline=pink_edge, width=4)

    # Vertical accent bar
    bar_x = margin + 50
    bar_top = qy0 + margin + 70
    bar_bot = qy1 - margin - 70
    draw.rectangle(
        (bar_x, bar_top, bar_x + 18, bar_bot),
        fill=pink_fg)

    # Decorative opening quote
    try:
        quote_font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf", 180)
    except OSError:
        quote_font = ImageFont.load_default()
    draw.text((bar_x + 40, qy0 + margin + 5), "“",
              fill=pink_fg, font=quote_font)

    # Caption text
    try:
        text_font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Italic.ttf", 56)
    except OSError:
        text_font = ImageFont.load_default()
    text_x = bar_x + 60
    text_y = qy0 + margin + 200
    wrapped = textwrap.fill(caption, width=36)
    draw.text((text_x, text_y), wrapped,
              fill=text_color, font=text_font, spacing=14)

    return np.asarray(canvas)


def _save_image_as_pdf(img: np.ndarray, path: Path, dpi: int = 300) -> None:
    """Wrap a numpy image into a tight PDF (and PNG sibling)."""
    H, W = img.shape[:2]
    fig = plt.figure(figsize=(W / dpi, H / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(img)
    ax.set_axis_off()
    fig.savefig(path, format="pdf", dpi=dpi,
                bbox_inches="tight", pad_inches=0.0)
    fig.savefig(path.with_suffix(".png"), format="png", dpi=dpi,
                bbox_inches="tight", pad_inches=0.0)
    plt.close(fig)


def render_pipeline_steps(args) -> None:
    """Produce 4 separate PDFs (one per pipeline stage), so the user can drop
    each into a slide independently. No labels, no captions inside the
    figures — text gets added in PowerPoint.
    """
    _set_pres_rc()
    out = args.output
    scan_dir = args.root / args.scan_id

    # Step 1 — IQA composite (rejected ✗ on top, kept ✓ on bottom)
    rej_path = scan_dir / "color" / f"{args.reject_frame}.jpg"
    keep_path = scan_dir / "color" / f"{args.keep_frame}.jpg"
    thumb1 = _make_iqa_thumb(rej_path, keep_path,
                             score_reject=args.reject_score,
                             score_keep=args.keep_score)
    p1 = out / "pipeline_step1_quality_filter.pdf"
    _save_image_as_pdf(thumb1, p1)
    print(f"Saved: {p1}")

    # Step 2 — visibility (re-wrap the already-rendered isometric image)
    if args.visibility_img.exists():
        img2 = np.asarray(Image.open(args.visibility_img).convert("RGB"))
        p2 = out / "pipeline_step2_visibility.pdf"
        _save_image_as_pdf(img2, p2)
        print(f"Saved: {p2}")
    else:
        print(f"  [skip] visibility image not found: {args.visibility_img}")

    # Step 3 — DPP floor plan (re-wrap)
    if args.dpp_img.exists():
        img3 = np.asarray(Image.open(args.dpp_img).convert("RGB"))
        p3 = out / "pipeline_step3_dpp.pdf"
        _save_image_as_pdf(img3, p3)
        print(f"Saved: {p3}")
    else:
        print(f"  [skip] DPP image not found: {args.dpp_img}")

    # Step 4 — keyframe + pink quote card composite
    rgb_path = scan_dir / "color" / f"{args.caption_frame}.jpg"
    desc_path = scan_dir / "output" / "descriptions" / f"{args.caption_frame}.json"
    desc = json.loads(desc_path.read_text())
    thumb4 = _make_caption_thumb(rgb_path=rgb_path,
                                 caption=desc.get("description", ""))
    p4 = out / "pipeline_step4_caption.pdf"
    _save_image_as_pdf(thumb4, p4)
    print(f"Saved: {p4}")


def parse_args():
    ap = argparse.ArgumentParser(description="D1 pipeline overview figure.")
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--scan-id", "--scan_id", dest="scan_id", required=True)
    ap.add_argument("--keep-frame", "--keep_frame", dest="keep_frame",
                    type=str, default="003769")
    ap.add_argument("--reject-frame", "--reject_frame", dest="reject_frame",
                    type=str, default="004516")
    ap.add_argument("--keep-score", "--keep_score", dest="keep_score",
                    type=float, default=0.66)
    ap.add_argument("--reject-score", "--reject_score", dest="reject_score",
                    type=float, default=0.15)
    ap.add_argument("--visibility-img", "--visibility_img",
                    dest="visibility_img", type=Path,
                    default=Path("docs/figures/visibility_isometric_scene0002_00.png"))
    ap.add_argument("--dpp-img", "--dpp_img", dest="dpp_img", type=Path,
                    default=Path("docs/figures/supp_dpp_selection.png"))
    ap.add_argument("--caption-frame", "--caption_frame",
                    dest="caption_frame", type=str, default="000838")
    ap.add_argument("--output", type=Path, default=Path("docs/figures"))
    return ap.parse_args()


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    render_pipeline_steps(args)


if __name__ == "__main__":
    main()
