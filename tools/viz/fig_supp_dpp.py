#!/usr/bin/env python3
"""Generate supp_dpp_selection.pdf: top-down floor plan with DPP stage markers.

Shows IQA survivors (small dots), Stage-1 semantic selections (circles),
and Stage-2 spatial selections (stars) projected onto the top-down view.

Reads pre-saved DPP diagnostics from the dataset pipeline
(``{cache_dir}/{scan_id}_dpp_diag.json``).

Usage::

    python -m tools.viz.fig_supp_dpp \
        --dataset scannet --root ./data/scans \
        --scan-id scene0002_00 \
        --output docs/figures
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _project_to_topdown(pts_3d, intrinsic, extrinsic):
    pts_h = np.hstack([pts_3d, np.ones((len(pts_3d), 1))]).T
    pts_cam = extrinsic @ pts_h
    pts_2d = intrinsic @ pts_cam[:3, :]
    px = pts_2d[0] / (pts_2d[2] + 1e-12)
    py = pts_2d[1] / (pts_2d[2] + 1e-12)
    return px, py


def _set_eccv_rc():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["CMU Serif", "Computer Modern Roman", "Times New Roman",
                        "DejaVu Serif", "serif"],
        "mathtext.fontset": "cm",
        "font.size": 10,
    })


# ---------------------------------------------------------------------------
# Figure rendering
# ---------------------------------------------------------------------------

def render_dpp_figure(
    topdown_img: np.ndarray,
    intrinsic: np.ndarray,
    extrinsic: np.ndarray,
    all_positions: np.ndarray,
    stage1_positions: np.ndarray,
    stage2_positions: np.ndarray,
    stage1_quality: np.ndarray,
    output_path: Path,
    *,
    stage2_fids: list[str] | None = None,
    color_dir: Path | None = None,
    n_thumbs: int = 0,
):
    """Render the DPP selection figure.

    When ``n_thumbs > 0`` and ``color_dir`` / ``stage2_fids`` are provided,
    a side column of ``n_thumbs`` keyframe thumbnails is added to the right
    of the floor plan, with numbered badges drawn on the matching stage-2
    stars so the audience can match the dot on the floor plan to the actual
    image.
    """
    from PIL import Image as PILImage
    from matplotlib.patches import FancyBboxPatch
    _set_eccv_rc()
    H, W = topdown_img.shape[:2]

    show_thumbs = n_thumbs > 0 and color_dir is not None and stage2_fids
    if show_thumbs:
        # Pick `n_thumbs` evenly-spaced indices into the stage-2 selection
        idx = np.linspace(0, len(stage2_fids) - 1, n_thumbs).astype(int).tolist()
        idx = sorted(set(idx))[:n_thumbs]
        chosen_fids = [stage2_fids[i] for i in idx]
        chosen_pos_idx = idx
    else:
        chosen_fids = []
        chosen_pos_idx = []

    if show_thumbs:
        # Wider figure: floor plan on the left, thumbnail column on the right
        thumb_col_in = 3.0          # column width in inches
        floor_w_in = W / 300
        floor_h_in = H / 300
        fig_w = floor_w_in + thumb_col_in + 0.3
        fig_h = max(floor_h_in, n_thumbs * (thumb_col_in * 0.78))
        fig = plt.figure(figsize=(fig_w, fig_h), dpi=300)
        gs = fig.add_gridspec(1, 2,
                              width_ratios=[floor_w_in, thumb_col_in],
                              wspace=0.04,
                              left=0.005, right=0.995, top=0.99, bottom=0.01)
        ax = fig.add_subplot(gs[0, 0])
        ax_thumbs = fig.add_subplot(gs[0, 1])
        ax_thumbs.set_xlim(0, 1)
        ax_thumbs.set_ylim(0, 1)
        ax_thumbs.set_axis_off()
    else:
        fig, ax = plt.subplots(1, 1, figsize=(W / 300, H / 300), dpi=300)

    ax.imshow(topdown_img)

    # Project all camera positions (IQA survivors)
    if len(all_positions) > 0:
        px_all, py_all = _project_to_topdown(all_positions, intrinsic, extrinsic)
        mask = (px_all >= 0) & (px_all < W) & (py_all >= 0) & (py_all < H)
        ax.scatter(px_all[mask], py_all[mask], s=6, c="#2e7d32", alpha=0.5,
                   edgecolors="none", zorder=2, label="IQA survivors")

    # Stage 1 selections (circles, colored by quality)
    if len(stage1_positions) > 0:
        px_s1, py_s1 = _project_to_topdown(stage1_positions, intrinsic, extrinsic)
        mask = (px_s1 >= 0) & (px_s1 < W) & (py_s1 >= 0) & (py_s1 < H)
        q_norm = stage1_quality / max(stage1_quality.max(), 1e-12)
        cmap = plt.get_cmap("YlOrRd")
        colors = [cmap(q_norm[i]) if mask[i] else (0, 0, 0, 0) for i in range(len(mask))]
        ax.scatter(px_s1[mask], py_s1[mask], s=50, c=[c for i, c in enumerate(colors) if mask[i]],
                   edgecolors="black", linewidths=0.6, zorder=3,
                   label="Stage 1 (semantic)")

    # Stage 2 selections (stars)
    s2_px = s2_py = None
    if len(stage2_positions) > 0:
        s2_px, s2_py = _project_to_topdown(stage2_positions, intrinsic, extrinsic)
        mask = (s2_px >= 0) & (s2_px < W) & (s2_py >= 0) & (s2_py < H)
        ax.scatter(s2_px[mask], s2_py[mask], s=140, c="#1565c0", marker="*",
                   edgecolors="white", linewidths=0.8, zorder=4,
                   label="Stage 2 (spatial)")

    # Numbered badges on the chosen stars
    if show_thumbs and s2_px is not None:
        for badge_n, pos_idx in enumerate(chosen_pos_idx, start=1):
            if pos_idx >= len(s2_px):
                continue
            x, y = float(s2_px[pos_idx]), float(s2_py[pos_idx])
            if not (0 <= x < W and 0 <= y < H):
                continue
            # Larger highlighted star + numeric badge to the side
            ax.scatter([x], [y], s=320, c="#0d47a1", marker="*",
                       edgecolors="white", linewidths=1.6, zorder=5)
            ax.text(x + 22, y - 22, str(badge_n),
                    fontsize=11, fontweight="bold", color="white",
                    bbox=dict(boxstyle="circle,pad=0.25",
                              fc="#0d47a1", ec="white", linewidth=1.0),
                    ha="center", va="center", zorder=6)

    # Legend
    legend = ax.legend(loc="upper right", fontsize=7, framealpha=0.9,
                       edgecolor="#888888", handletextpad=0.3, borderpad=0.4)
    legend.set_zorder(10)

    ax.set_axis_off()
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)

    # Thumbnail column
    if show_thumbs:
        slot_h = 1.0 / n_thumbs
        for i, fid in enumerate(chosen_fids):
            top = 1.0 - i * slot_h
            bot = 1.0 - (i + 1) * slot_h
            cx = 0.5

            img_path = color_dir / f"{fid}.jpg"
            if not img_path.exists():
                # Try png alternate
                alt = color_dir / f"{fid}.png"
                img_path = alt if alt.exists() else None
            if img_path is None:
                continue
            with PILImage.open(img_path) as im:
                im = im.convert("RGB")
                w0, h0 = im.size
                target_w = 800
                im = im.resize((target_w, int(h0 * target_w / w0)),
                               PILImage.LANCZOS)
                thumb = np.asarray(im)

            # Inner thumbnail axes positioned within the right column
            # Use add_axes inside the column ax to place the image with
            # the correct aspect ratio.
            cell_y0 = bot + 0.03 * slot_h
            cell_y1 = top - 0.10 * slot_h
            cell_h = cell_y1 - cell_y0
            cell_w = 0.92  # leave a small left/right margin
            cell_x0 = (1.0 - cell_w) / 2.0

            # Convert column axes-fractional to figure-fractional coords for add_axes
            bb = ax_thumbs.get_position()
            fig_x = bb.x0 + cell_x0 * bb.width
            fig_y = bb.y0 + cell_y0 * bb.height
            fig_w_box = cell_w * bb.width
            fig_h_box = cell_h * bb.height
            sub_ax = fig.add_axes([fig_x, fig_y, fig_w_box, fig_h_box])
            sub_ax.imshow(thumb)
            sub_ax.set_axis_off()
            for spine in sub_ax.spines.values():
                spine.set_visible(True)
                spine.set_color("#bbbbbb")
                spine.set_linewidth(0.6)

            # Numbered badge on top-left of thumbnail
            sub_ax.text(0.04, 0.92, str(i + 1),
                        transform=sub_ax.transAxes,
                        fontsize=12, fontweight="bold", color="white",
                        bbox=dict(boxstyle="circle,pad=0.25",
                                  fc="#0d47a1", ec="white", linewidth=1.0),
                        ha="center", va="center", zorder=10)

            # Caption (frame id) below
            ax_thumbs.text(cx, bot + 0.005 * slot_h,
                           f"frame {fid}",
                           fontsize=8, color="#444444",
                           ha="center", va="bottom",
                           transform=ax_thumbs.transAxes)

    if not show_thumbs:
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    fig.savefig(output_path, bbox_inches="tight", dpi=300, pad_inches=0.02)
    if str(output_path).endswith(".pdf"):
        png_path = output_path.with_suffix(".png")
        fig.savefig(png_path, bbox_inches="tight", dpi=300, pad_inches=0.02,
                    format="png")
        print(f"  Saved: {png_path}")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser(description="DPP keyframe selection figure.")
    ap.add_argument("--dataset", choices=["3rscan", "scannet"], required=True)
    ap.add_argument("--root", type=Path, required=True,
                    help="Dataset root (e.g. ./data/scans)")
    ap.add_argument("--scan-id", "--scan_id", dest="scan_id", required=True)
    ap.add_argument("--cache-dir", "--cache_dir", dest="cache_dir", type=Path, default=None,
                    help="Directory with {scan_id}_dpp_diag.json. "
                         "Default: {root}/{scan_id}/output/cache")
    ap.add_argument("--output", type=Path, default=Path("docs/figures"))
    ap.add_argument("--topdown-size", "--topdown_size", dest="topdown_size",
                    type=int, default=2048)
    ap.add_argument("--with-thumbnails", "--with_thumbnails",
                    dest="with_thumbnails", action="store_true",
                    help="Add a side column of stage-2 keyframe thumbnails "
                         "and number-match them to stars on the floor plan.")
    ap.add_argument("--n-thumbs", "--n_thumbs", dest="n_thumbs",
                    type=int, default=3,
                    help="How many stage-2 thumbnails to show (default 3).")
    return ap.parse_args()


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    scan_dir = args.root / args.scan_id
    cache_dir = args.cache_dir or (scan_dir / "output" / "cache")

    # 1. Load mesh
    from tools.viz.visualize_teaser import load_scene_any, detect_up_axis, render_topdown
    print(f"[1/3] Loading scene: {args.scan_id}")
    mesh, tri2obj, obj2faces = load_scene_any(scan_dir, dataset=args.dataset)
    up_axis = detect_up_axis(mesh)

    # 2. Load DPP diagnostics
    diag_path = cache_dir / f"{args.scan_id}_dpp_diag.json"
    if not diag_path.exists():
        # Check alternative locations
        alt = scan_dir / "output" / "cache" / f"{args.scan_id}_dpp_diag.json"
        if alt.exists():
            diag_path = alt
        else:
            raise FileNotFoundError(
                f"DPP diagnostics not found: {diag_path}\n"
                f"Re-run the dataset pipeline to generate it.")

    print(f"[2/3] Loading DPP diagnostics: {diag_path}")
    diag = json.loads(diag_path.read_text())

    all_fids = diag["all_fids"]
    stage1_fids = set(diag["stage1_fids"])
    stage2_fids = set(diag["stage2_fids"])
    quality_scores = np.array(diag["quality_scores"])
    poses = {fid: np.array(p) for fid, p in diag["camera_poses"].items()}

    print(f"  {len(all_fids)} IQA survivors, "
          f"{len(stage1_fids)} Stage 1, "
          f"{len(stage2_fids)} Stage 2")
    print(f"  {len(poses)} camera poses available")

    # 3. Render figure
    print(f"[3/3] Rendering figure...")
    topdown_img, intrinsic, extrinsic = render_topdown(
        mesh, up_axis, args.topdown_size)

    # Build fid→index for quality lookup
    fid_to_idx = {fid: i for i, fid in enumerate(all_fids)}

    # Gather 3D positions per tier
    all_positions = []
    for fid in all_fids:
        p = poses.get(fid)
        if p is not None:
            all_positions.append(p[:3, 3])
    all_positions = np.array(all_positions) if all_positions else np.empty((0, 3))

    s1_positions, s1_quality = [], []
    for fid in diag["stage1_fids"]:
        p = poses.get(fid)
        if p is not None:
            s1_positions.append(p[:3, 3])
            s1_quality.append(quality_scores[fid_to_idx[fid]])
    s1_positions = np.array(s1_positions) if s1_positions else np.empty((0, 3))
    s1_quality = np.array(s1_quality) if s1_quality else np.empty(0)

    s2_positions = []
    for fid in diag["stage2_fids"]:
        p = poses.get(fid)
        if p is not None:
            s2_positions.append(p[:3, 3])
    s2_positions = np.array(s2_positions) if s2_positions else np.empty((0, 3))

    out_path = args.output / (
        "supp_dpp_selection_thumbs.pdf" if args.with_thumbnails
        else "supp_dpp_selection.pdf")
    render_dpp_figure(
        topdown_img, intrinsic, extrinsic,
        all_positions, s1_positions, s2_positions,
        s1_quality,
        out_path,
        stage2_fids=diag["stage2_fids"] if args.with_thumbnails else None,
        color_dir=(scan_dir / "color") if args.with_thumbnails else None,
        n_thumbs=args.n_thumbs if args.with_thumbnails else 0,
    )
    print("Done.")


if __name__ == "__main__":
    main()
