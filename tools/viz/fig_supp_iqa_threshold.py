#!/usr/bin/env python3
"""Generate a QualiCLIP threshold-analysis figure for the dataset slide.

Loads the cached scores from ``iqa_scores_{scan_id}.json`` (produced by
``fig_supp_iqa``) and renders a histogram with quartile + threshold markers,
plus a box plot — same content as ``scripts/dataset/analyze_qualiclip.py``
but in the slide's visual style and without re-running QualiCLIP.

Usage::

    python -m tools.viz.fig_supp_iqa_threshold \\
        --scores docs/figures/iqa_scores_scene0002_00.json \\
        --threshold 0.45 \\
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


def _set_pres_rc():
    plt.rcParams.update({
        "font.family": ["Aptos Display", "Aptos", "Inter",
                        "Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 11,
        "mathtext.fontset": "cm",
    })


def render_threshold_figure(scores: np.ndarray,
                            threshold: float,
                            output_path: Path,
                            scan_id: str = "",
                            dpi: int = 300) -> None:
    _set_pres_rc()

    scores = np.asarray(scores, dtype=np.float64)
    scores = scores[~np.isnan(scores)]

    median = float(np.median(scores))
    q1 = float(np.percentile(scores, 25))
    q3 = float(np.percentile(scores, 75))

    fig = plt.figure(figsize=(13, 5.4), dpi=dpi)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.45, 0.55],
                          wspace=0.30,
                          left=0.07, right=0.97,
                          top=0.88, bottom=0.16)

    accent_keep = "#2e7d32"
    accent_reject = "#d32f2f"
    accent_threshold = "#1565c0"

    # ---- Histogram ----
    ax = fig.add_subplot(gs[0, 0])
    bins = np.linspace(0.0, 1.0, 41)
    ax.hist(scores, bins=bins, color="#5b6abf",
            edgecolor="white", linewidth=0.6, alpha=0.92)
    ax.set_xlim(0, 1)

    # Shaded reject / keep bands
    ax.axvspan(0.0, threshold, alpha=0.07, color=accent_reject, zorder=0)
    ax.axvspan(threshold, 1.0, alpha=0.07, color=accent_keep, zorder=0)

    # Threshold line
    ax.axvline(threshold, color=accent_threshold, linestyle="--",
               linewidth=2.0, zorder=4,
               label=fr"Threshold $\tau = {threshold:.2f}$")

    # Quartile markers (distinct colours so Q1 / Median / Q3 are tellable apart)
    ax.axvline(q1, color="#cc7a00", linestyle=":", linewidth=1.5,
               alpha=0.9, zorder=3, label=f"Q1 = {q1:.2f}")
    ax.axvline(median, color="#7e3a91", linestyle=":", linewidth=1.8,
               alpha=0.95, zorder=3, label=f"Median = {median:.2f}")
    ax.axvline(q3, color="#0a6e3f", linestyle=":", linewidth=1.5,
               alpha=0.9, zorder=3, label=f"Q3 = {q3:.2f}")

    # Add headroom for the legend
    ymax = ax.get_ylim()[1]
    ax.set_ylim(0, ymax * 1.22)
    ymax = ax.get_ylim()[1]

    # Band labels — placed near the x-axis (bottom) so they cannot collide
    # with the legend that sits in the upper-left.
    label_y = ymax * 0.07
    ax.text(threshold / 2, label_y, "rejected",
            ha="center", va="bottom", fontsize=11, color=accent_reject,
            fontweight="bold", style="italic",
            bbox=dict(boxstyle="round,pad=0.30",
                      fc=(1, 0.93, 0.93), ec=accent_reject, lw=0.7))
    ax.text((1 + threshold) / 2, label_y, "kept",
            ha="center", va="bottom", fontsize=11, color=accent_keep,
            fontweight="bold", style="italic",
            bbox=dict(boxstyle="round,pad=0.30",
                      fc=(0.93, 0.98, 0.93), ec=accent_keep, lw=0.7))

    ax.set_xlabel("QualiCLIP score", fontsize=11.5, labelpad=8)
    ax.set_ylabel("# frames", fontsize=11.5, labelpad=6)
    title = "QualiCLIP score distribution"
    if scan_id:
        title += f" — {scan_id}"
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="both", labelsize=9.5)

    ax.legend(loc="upper left", fontsize=9, framealpha=0.92,
              edgecolor="#888888")

    # ---- Box plot ----
    ax2 = fig.add_subplot(gs[0, 1])
    bp = ax2.boxplot(scores, vert=True, patch_artist=True, widths=0.55,
                     showfliers=True,
                     medianprops=dict(color="#7e3a91", linewidth=2),
                     whiskerprops=dict(color="#444"),
                     capprops=dict(color="#444"),
                     flierprops=dict(marker="o", markersize=4,
                                     markerfacecolor="#888", alpha=0.5))
    bp["boxes"][0].set_facecolor("#cdd5f7")
    bp["boxes"][0].set_edgecolor("#444")

    ax2.axhline(threshold, color=accent_threshold, linestyle="--",
                linewidth=1.8, alpha=0.85, zorder=4)
    # Place τ label INSIDE the panel (left margin), so the bbox never spills
    # off the right edge of the figure.
    ax2.text(0.58, threshold + 0.025,
             fr"$\tau = {threshold:.2f}$",
             color=accent_threshold, fontsize=10.5, fontweight="bold",
             va="bottom", ha="left",
             bbox=dict(boxstyle="round,pad=0.30", fc="white",
                       ec=accent_threshold, lw=0.9),
             zorder=5)
    ax2.set_xlim(0.5, 1.5)
    ax2.set_ylim(0, 1)
    ax2.set_xticks([])
    ax2.set_ylabel("QualiCLIP score", fontsize=11.5, labelpad=6)
    ax2.set_title("Spread", fontsize=13, fontweight="bold", pad=12)
    for spine in ("top", "right"):
        ax2.spines[spine].set_visible(False)
    ax2.tick_params(axis="y", labelsize=9.5)

    fig.savefig(output_path, format="pdf", dpi=dpi,
                bbox_inches="tight", pad_inches=0.05)
    fig.savefig(output_path.with_suffix(".png"), format="png", dpi=dpi,
                bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"Saved: {output_path}")
    print(f"Saved: {output_path.with_suffix('.png')}")
    print(f"  N frames analysed: {len(scores)}")
    print(f"  Q1 / median / Q3 = {q1:.3f} / {median:.3f} / {q3:.3f}")
    print(f"  At threshold {threshold:.2f}:  "
          f"{int((scores >= threshold).sum())} kept  /  "
          f"{int((scores < threshold).sum())} rejected")


def parse_args():
    ap = argparse.ArgumentParser(
        description="QualiCLIP threshold-analysis figure (cached scores).")
    ap.add_argument("--scores", type=Path, required=True,
                    help="Path to iqa_scores_{scan_id}.json (cached score map).")
    ap.add_argument("--threshold", type=float, default=0.45,
                    help="QualiCLIP threshold to highlight (default 0.45).")
    ap.add_argument("--output", type=Path, default=Path("docs/figures"))
    ap.add_argument("--scan-id", "--scan_id", dest="scan_id",
                    type=str, default="",
                    help="Optional scan id for the title (purely cosmetic).")
    return ap.parse_args()


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    payload = json.loads(args.scores.read_text())
    if isinstance(payload, dict):
        scores = list(payload.values())
    elif isinstance(payload, list):
        scores = payload
    else:
        raise ValueError(f"Unexpected JSON shape in {args.scores}")
    name = args.scan_id or args.scores.stem.replace("iqa_scores_", "")
    out = args.output / f"iqa_threshold_{name}.pdf"
    render_threshold_figure(np.asarray(scores, dtype=np.float64),
                            threshold=args.threshold,
                            output_path=out,
                            scan_id=args.scan_id)


if __name__ == "__main__":
    main()
