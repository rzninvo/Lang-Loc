#!/usr/bin/env python3
"""Two-panel DPP-intuition figure showing both kernels:

* Stage 1 — semantic CLIP similarity ``S^sem`` (over the IQA-survivor pool),
  with Stage-1 selections marked.
* Stage 2 — spatial pose similarity ``S^pose`` (over the Stage-1 pool),
  with Stage-2 selections marked.

The block / cluster structure should be visible in both — that's the visual
intuition for why a DPP picks one item per cluster.

Usage::

    python -m tools.viz.fig_supp_dpp_both_kernels \\
        --root ./data/scans \\
        --scan-id scene0002_00 \\
        --output docs/figures
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


SIGMA_P = 0.75
SIGMA_A_DEG = 20.0


def _set_pres_rc():
    plt.rcParams.update({
        "font.family": ["Aptos Display", "Aptos", "Inter",
                        "Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 11,
        "mathtext.fontset": "cm",
    })


# ---------------------------------------------------------------------------
# Pose similarity (Stage 2 kernel — same as the existing kernel script)
# Implementation matches dpp.py exactly: angles in degrees, σ_a = 20°.
# ---------------------------------------------------------------------------

def pose_similarity_matrix(poses: np.ndarray) -> np.ndarray:
    pos = poses[:, :3, 3]
    fwd = poses[:, :3, 2]
    fwd = fwd / np.maximum(np.linalg.norm(fwd, axis=1, keepdims=True), 1e-9)
    d_cam = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
    cos_theta = np.clip(fwd @ fwd.T, -1.0, 1.0)
    dtheta_deg = np.degrees(np.arccos(cos_theta))
    s_pos = np.exp(-(d_cam ** 2) / (2 * SIGMA_P ** 2))
    s_ang = np.exp(-(dtheta_deg ** 2) / (2 * SIGMA_A_DEG ** 2))
    S = s_pos * s_ang
    np.fill_diagonal(S, 1.0)
    return S


# ---------------------------------------------------------------------------
# CLIP semantic similarity (Stage 1 kernel)
# ---------------------------------------------------------------------------

def compute_clip_sem_matrix(image_paths: list[Path],
                            device: str = "cuda") -> np.ndarray:
    """S^sem_ij = (cos(ψ_i, ψ_j) + 1) / 2  with L2-normalised CLIP ViT-B/32."""
    from langloc.dataset.frame_selection.dpp import compute_clip_embeddings
    emb = compute_clip_embeddings(image_paths, model_name="ViT-B/32",
                                  device=device, batch_size=32)
    cos_sim = emb @ emb.T
    S = np.clip((cos_sim + 1.0) / 2.0, 0.0, 1.0)
    np.fill_diagonal(S, 1.0)
    return S


# ---------------------------------------------------------------------------
# Heatmap helper
# ---------------------------------------------------------------------------

def _draw_heatmap(ax, S: np.ndarray, picks_mask: np.ndarray,
                  title: str, cbar_label: str,
                  cmap_name: str = "magma",
                  pick_colour: str = "#00e5ff",
                  pick_label: str = "selected",
                  vmin: float | None = None,
                  vmax: float | None = None):
    n = S.shape[0]
    # Default to auto-stretching the colormap to the actual off-diagonal
    # range — the CLIP-based kernel has very compressed values, so a fixed
    # [0, 1] range hides the block structure that's actually there.
    if vmin is None or vmax is None:
        triu_idx = np.triu_indices(n, k=1)
        off = S[triu_idx]
        if vmin is None:
            vmin = float(np.percentile(off, 2.0))
        if vmax is None:
            vmax = float(np.percentile(off, 99.5))
    im = ax.imshow(S, cmap=cmap_name, vmin=vmin, vmax=vmax,
                   interpolation="nearest", aspect="equal")

    # 5 evenly-spaced index ticks (more readable than every frame ID)
    tick_n = 5 if n >= 12 else min(n, 5)
    tick_idx = np.linspace(0, n - 1, tick_n, dtype=int)
    ax.set_xticks(tick_idx)
    ax.set_yticks(tick_idx)
    ax.set_xticklabels([f"#{i + 1}" for i in tick_idx], fontsize=9)
    ax.set_yticklabels([f"#{i + 1}" for i in tick_idx], fontsize=9)
    ax.set_xlabel(r"frame index $j$", fontsize=11, labelpad=8)
    ax.set_ylabel(r"frame index $i$", fontsize=11, labelpad=8)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)

    # Highlight picked rows/cols with translucent overlay + diagonal stars
    pick_idx = np.where(picks_mask)[0]
    for i in pick_idx:
        ax.add_patch(Rectangle((-0.5, i - 0.5), n, 1,
                               linewidth=0.0, facecolor=pick_colour,
                               alpha=0.10, zorder=4))
        ax.add_patch(Rectangle((i - 0.5, -0.5), 1, n,
                               linewidth=0.0, facecolor=pick_colour,
                               alpha=0.10, zorder=4))
    ax.scatter(pick_idx, pick_idx, marker="*",
               s=120 if n <= 30 else 70,
               facecolor=pick_colour, edgecolor="black", linewidth=0.4,
               zorder=6, label=pick_label)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.92,
              edgecolor="#888888")

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.045, pad=0.025)
    cbar.set_label(cbar_label, fontsize=11, labelpad=8)
    cbar.ax.tick_params(labelsize=9)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def render_both_kernels(diag: dict,
                        scan_dir: Path,
                        output_path: Path,
                        device: str = "cuda",
                        dpi: int = 300) -> None:
    _set_pres_rc()

    # --- Stage 1 pool: IQA survivors -------------------------------------
    all_fids = sorted(diag["all_fids"], key=lambda x: int(x))
    stage1_set = set(diag["stage1_fids"])
    stage1_mask = np.array([fid in stage1_set for fid in all_fids], dtype=bool)

    color_dir = scan_dir / "color"
    s1_paths = [color_dir / f"{fid}.jpg" for fid in all_fids]
    missing = [p for p in s1_paths if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing color frames for CLIP scoring; example: {missing[0]}")
    print(f"[1/3] Computing CLIP semantic kernel ({len(all_fids)} frames)...")
    S_sem = compute_clip_sem_matrix(s1_paths, device=device)

    # --- Stage 2 pool: Stage-1 selections -------------------------------
    s1_fids = sorted(diag["stage1_fids"], key=lambda x: int(x))
    poses_dict = {fid: np.array(p) for fid, p in diag["camera_poses"].items()}
    poses = np.stack([poses_dict[fid] for fid in s1_fids], axis=0)
    print(f"[2/3] Computing pose similarity kernel ({len(s1_fids)} frames)...")
    S_pose = pose_similarity_matrix(poses)
    stage2_set = set(diag["stage2_fids"])
    stage2_mask = np.array([fid in stage2_set for fid in s1_fids], dtype=bool)

    # --- Figure -----------------------------------------------------------
    print(f"[3/3] Rendering...")
    fig = plt.figure(figsize=(15, 6.5), dpi=dpi)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0],
                          wspace=0.32,
                          left=0.06, right=0.97,
                          top=0.90, bottom=0.13)

    ax1 = fig.add_subplot(gs[0, 0])
    _draw_heatmap(
        ax1, S_sem, stage1_mask,
        title=r"Stage 1 — semantic kernel  $S^{\mathrm{sem}}_{ij}$",
        cbar_label=r"$S^{\mathrm{sem}}_{ij}$",
        cmap_name="magma",
        pick_colour="#00e5ff",
        pick_label="Stage-1 selection",
        # Let the auto-percentile stretch take over so the dynamic range
        # of CLIP cosine similarity actually reveals block structure.
    )

    ax2 = fig.add_subplot(gs[0, 1])
    _draw_heatmap(
        ax2, S_pose, stage2_mask,
        title=r"Stage 2 — spatial kernel  $S^{\mathrm{pose}}_{ij}$",
        cbar_label=r"$S^{\mathrm{pose}}_{ij}$",
        cmap_name="viridis",
        pick_colour="#ff5252",
        pick_label="Stage-2 selection",
        vmin=0.0, vmax=1.0,  # spatial kernel naturally covers full range
    )

    fig.savefig(output_path, format="pdf", dpi=dpi,
                bbox_inches="tight", pad_inches=0.05)
    fig.savefig(output_path.with_suffix(".png"), format="png", dpi=dpi,
                bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"Saved: {output_path}")
    print(f"Saved: {output_path.with_suffix('.png')}")
    print(f"  Stage 1: {S_sem.shape[0]} frames in pool, "
          f"{int(stage1_mask.sum())} selected")
    print(f"  Stage 2: {S_pose.shape[0]} frames in pool, "
          f"{int(stage2_mask.sum())} selected")


def parse_args():
    ap = argparse.ArgumentParser(
        description="Two-panel DPP kernel figure (Stage 1 + Stage 2).")
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--scan-id", "--scan_id", dest="scan_id", required=True)
    ap.add_argument("--cache-dir", "--cache_dir", dest="cache_dir",
                    type=Path, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output", type=Path, default=Path("docs/figures"))
    return ap.parse_args()


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    scan_dir = args.root / args.scan_id
    cache_dir = args.cache_dir or (scan_dir / "output" / "cache")
    diag = json.loads((cache_dir / f"{args.scan_id}_dpp_diag.json").read_text())
    out = args.output / f"dpp_kernels_both_{args.scan_id}.pdf"
    render_both_kernels(diag, scan_dir, out, device=args.device)


if __name__ == "__main__":
    main()
