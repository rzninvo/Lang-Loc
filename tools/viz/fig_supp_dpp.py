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
):
    """Render the DPP selection figure."""
    _set_eccv_rc()
    H, W = topdown_img.shape[:2]

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
    if len(stage2_positions) > 0:
        px_s2, py_s2 = _project_to_topdown(stage2_positions, intrinsic, extrinsic)
        mask = (px_s2 >= 0) & (px_s2 < W) & (py_s2 >= 0) & (py_s2 < H)
        ax.scatter(px_s2[mask], py_s2[mask], s=120, c="#1565c0", marker="*",
                   edgecolors="white", linewidths=0.8, zorder=4,
                   label="Stage 2 (spatial)")

    # Legend
    legend = ax.legend(loc="upper right", fontsize=7, framealpha=0.9,
                       edgecolor="#888888", handletextpad=0.3, borderpad=0.4)
    legend.set_zorder(10)

    ax.set_axis_off()
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    fig.savefig(output_path, bbox_inches="tight", dpi=300, pad_inches=0.02)
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

    render_dpp_figure(
        topdown_img, intrinsic, extrinsic,
        all_positions, s1_positions, s2_positions,
        s1_quality,
        args.output / "supp_dpp_selection.pdf",
    )
    print("Done.")


if __name__ == "__main__":
    main()
