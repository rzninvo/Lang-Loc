#!/usr/bin/env python3
"""Generate a DPP-intuition figure: pairwise pose-similarity matrix of the
Stage-1 frame pool with Stage-2 picks highlighted, plus a histogram of off-
diagonal similarities so the audience can see the block / cluster structure
that the DPP repels.

Usage::

    python -m tools.viz.fig_supp_dpp_kernel \\
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


# ---------------------------------------------------------------------------
# Pairwise pose similarity (matches the spatial kernel in the report)
#   S_pose_ij = exp(-d^2 / (2 σ_p^2)) · exp(-Δθ^2 / (2 σ_a^2))
# ---------------------------------------------------------------------------

SIGMA_P = 0.75                     # metres
SIGMA_A_RAD = np.deg2rad(20.0)     # radians


def _set_pres_rc():
    plt.rcParams.update({
        "font.family": ["Aptos Display", "Aptos", "Inter",
                        "Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 11,
        "mathtext.fontset": "cm",
    })


def pose_similarity_matrix(poses: np.ndarray) -> np.ndarray:
    """``poses`` is (N, 4, 4) cam2world; returns (N, N) pose-only similarity."""
    pos = poses[:, :3, 3]                               # (N, 3)
    fwd = poses[:, :3, 2]                               # (N, 3) — OpenCV +Z
    fwd = fwd / np.maximum(np.linalg.norm(fwd, axis=1, keepdims=True), 1e-9)

    # Pairwise distance and angle
    d_cam = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
    cos_theta = np.clip(fwd @ fwd.T, -1.0, 1.0)
    dtheta = np.arccos(cos_theta)

    s_pos = np.exp(-(d_cam ** 2) / (2 * SIGMA_P ** 2))
    s_ang = np.exp(-(dtheta ** 2) / (2 * SIGMA_A_RAD ** 2))
    return s_pos * s_ang


def render_dpp_kernel_figure(diag: dict, output_path: Path,
                             dpi: int = 300) -> None:
    _set_pres_rc()

    s1_fids = list(diag["stage1_fids"])
    s2_set = set(diag["stage2_fids"])
    poses_dict = {fid: np.array(p) for fid, p in diag["camera_poses"].items()}

    # Order frames numerically by frame id (≈ temporal/sweep order, so nearby
    # viewpoints cluster as adjacent rows → block structure is visible).
    s1_fids = sorted(s1_fids, key=lambda x: int(x))
    poses = np.stack([poses_dict[fid] for fid in s1_fids], axis=0)

    S = pose_similarity_matrix(poses)
    n = S.shape[0]
    s2_mask = np.array([fid in s2_set for fid in s1_fids], dtype=bool)
    s2_idx = np.where(s2_mask)[0]

    # Off-diagonal entries for the histogram
    triu_idx = np.triu_indices(n, k=1)
    off_diag = S[triu_idx]

    # --- Figure: heatmap on the left, histogram on the right ----------------
    fig = plt.figure(figsize=(14.5, 6.2), dpi=dpi)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 0.95],
                          wspace=0.32,
                          left=0.06, right=0.97,
                          top=0.90, bottom=0.13)

    # ---- Heatmap ----
    ax = fig.add_subplot(gs[0, 0])
    im = ax.imshow(S, cmap="magma", vmin=0.0, vmax=1.0,
                   interpolation="nearest", aspect="equal")

    # Drop dense per-frame tick labels — they crowd the axis at slide size.
    # Show 5 evenly-spaced indices to give the reader a rough scale instead.
    tick_n = 5
    tick_idx = np.linspace(0, n - 1, tick_n, dtype=int)
    ax.set_xticks(tick_idx)
    ax.set_yticks(tick_idx)
    ax.set_xticklabels([f"#{i+1}" for i in tick_idx], fontsize=9)
    ax.set_yticklabels([f"#{i+1}" for i in tick_idx], fontsize=9)
    ax.set_xlabel(r"Stage-1 frame index $j$", fontsize=11, labelpad=8)
    ax.set_ylabel(r"Stage-1 frame index $i$", fontsize=11, labelpad=8)
    ax.set_title(r"Stage-1 pool: pose similarity $\;S^{\mathrm{pose}}_{ij}$",
                 fontsize=13, fontweight="bold", pad=12)

    # Outline Stage-2 picks: cyan rows and cyan columns + diagonal star
    for i in s2_idx:
        ax.add_patch(Rectangle((-0.5, i - 0.5), n, 1,
                               linewidth=0.0, facecolor="cyan",
                               alpha=0.10, zorder=4))
        ax.add_patch(Rectangle((i - 0.5, -0.5), 1, n,
                               linewidth=0.0, facecolor="cyan",
                               alpha=0.10, zorder=4))
    ax.scatter(s2_idx, s2_idx, marker="*", s=110,
               facecolor="#00e5ff", edgecolor="black", linewidth=0.4,
               zorder=6, label="Stage-2 selection")

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.025)
    cbar.set_label(r"$S^{\mathrm{pose}}_{ij}$", fontsize=11, labelpad=8)
    cbar.ax.tick_params(labelsize=9)

    ax.legend(loc="upper right", fontsize=9, framealpha=0.92,
              edgecolor="#888888")

    # ---- Histogram ----
    ax2 = fig.add_subplot(gs[0, 1])
    bins = np.linspace(0.0, 1.0, 41)
    ax2.hist(off_diag, bins=bins,
             color="#7e3a91", edgecolor="white", linewidth=0.6, alpha=0.92)
    ax2.set_xlim(0, 1)
    ax2.set_xlabel(r"$S^{\mathrm{pose}}_{ij}$  (off-diagonal pairs)",
                   fontsize=11, labelpad=8)
    ax2.set_ylabel("# frame pairs", fontsize=11, labelpad=6)
    ax2.set_title("Pairwise similarity distribution",
                  fontsize=13, fontweight="bold", pad=12)
    for spine in ("top", "right"):
        ax2.spines[spine].set_visible(False)
    ax2.tick_params(axis="both", labelsize=9)

    # Annotate the regimes (push annotations above the tallest bar so they
    # never overlap the histogram itself)
    ax2.axvspan(0.0, 0.10, alpha=0.10, color="#2e7d32")
    ax2.axvspan(0.40, 1.0, alpha=0.10, color="#d32f2f")
    ymax = ax2.get_ylim()[1]
    ax2.set_ylim(0, ymax * 1.18)
    ymax = ax2.get_ylim()[1]
    ax2.text(0.05, ymax * 0.96, "diverse\npairs",
             ha="center", va="top", fontsize=9.5, color="#2e7d32",
             style="italic", fontweight="bold")
    ax2.text(0.70, ymax * 0.96, "redundant\npairs",
             ha="center", va="top", fontsize=9.5, color="#d32f2f",
             style="italic", fontweight="bold")

    fig.savefig(output_path, format="pdf", dpi=dpi,
                bbox_inches="tight", pad_inches=0.05)
    fig.savefig(output_path.with_suffix(".png"), format="png", dpi=dpi,
                bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"Saved: {output_path}")
    print(f"Saved: {output_path.with_suffix('.png')}")

    # Print a short sanity summary
    print(f"  N stage-1 frames: {n}")
    print(f"  N stage-2 picks:  {int(s2_mask.sum())}")
    print(f"  Off-diag S^pose: median={np.median(off_diag):.3f}  "
          f"mean={off_diag.mean():.3f}  "
          f">{0.4} = {(off_diag > 0.4).sum()}  pairs")


def parse_args():
    ap = argparse.ArgumentParser(description="DPP kernel-heatmap figure.")
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--scan-id", "--scan_id", dest="scan_id", required=True)
    ap.add_argument("--cache-dir", "--cache_dir", dest="cache_dir",
                    type=Path, default=None)
    ap.add_argument("--output", type=Path, default=Path("docs/figures"))
    return ap.parse_args()


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cache_dir = args.cache_dir or (args.root / args.scan_id / "output" / "cache")
    diag_path = cache_dir / f"{args.scan_id}_dpp_diag.json"
    diag = json.loads(diag_path.read_text())
    out = args.output / f"dpp_kernel_{args.scan_id}.pdf"
    render_dpp_kernel_figure(diag, out)


if __name__ == "__main__":
    main()
