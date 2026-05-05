#!/usr/bin/env python3
"""Generate a paper-style flowchart of the 4-stage LangLoc data pipeline.

Clean block diagram in the style of a methodology figure: rounded boxes for
each stage with a numbered chip + title + method line + parameter line, thin
arrows between boxes, optional input/output bubbles on the ends, and a
sub-arrow data-flow caption between adjacent stages.

Usage::

    python -m tools.viz.fig_supp_pipeline_flowchart \\
        --output docs/figures
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


def _set_pres_rc():
    plt.rcParams.update({
        "font.family": ["Aptos Display", "Aptos", "Inter",
                        "Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 11,
    })


# ---------------------------------------------------------------------------
# Stage definitions — edit here to change titles / params / colours
# ---------------------------------------------------------------------------

STAGES = [
    {
        "n": "01",
        "title": "Quality Filter",
        "method": "QualiCLIP",
        "params": r"score $\geq$ $\tau$ ≈ 0.45",
        "fill": "#eef4fb",
        "edge": "#3a6ea5",
        "chip": "#3a6ea5",
    },
    {
        "n": "02",
        "title": "Visibility Rasterisation",
        "method": "PyTorch3D",
        "params": "4-test cascade",
        "fill": "#fff5e6",
        "edge": "#cc7a00",
        "chip": "#cc7a00",
    },
    {
        "n": "03",
        "title": "Two-Stage DPP",
        "method": "Semantic + Spatial",
        "params": r"$K_1\!\leq\!25$ → $K_2\!\leq\!10$",
        "fill": "#f3eaf7",
        "edge": "#7e3a91",
        "chip": "#7e3a91",
    },
    {
        "n": "04",
        "title": "Description Generation",
        "method": "GPT-4o-mini",
        "params": "objects + relations → caption",
        "fill": "#fde9ee",
        "edge": "#b3415a",
        "chip": "#b3415a",
    },
]

# Optional inter-stage labels. Left empty by default for a cleaner look —
# pass --labels on the CLI to enable them.
ARROW_LABELS_DEFAULT = ["", "", ""]
ARROW_LABELS_VERBOSE = [
    "filtered frames",
    "visible objects",
    "selected keyframes",
]


def _draw_box(ax, x: float, y: float, w: float, h: float, stage: dict):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.0,rounding_size=0.6",
        facecolor=stage["fill"], edgecolor=stage["edge"], linewidth=1.4,
        zorder=2,
    )
    ax.add_patch(box)

    # Numbered chip in top-left
    chip_r = 0.85
    chip_x = x + 0.6 + chip_r
    chip_y = y + h - 0.6 - chip_r
    chip = plt.Circle((chip_x, chip_y), chip_r,
                      facecolor=stage["chip"], edgecolor="white",
                      linewidth=1.2, zorder=4)
    ax.add_patch(chip)
    ax.text(chip_x, chip_y, stage["n"],
            ha="center", va="center",
            fontsize=10.5, fontweight="bold", color="white", zorder=5)

    # Stage title (large, bold) — vertically centred a bit above middle
    cx = x + w / 2
    title_y = y + h * 0.62
    ax.text(cx, title_y, stage["title"],
            ha="center", va="center",
            fontsize=13, fontweight="bold", color="#1c1c1c", zorder=3)

    # Method line (italic, smaller)
    method_y = y + h * 0.38
    ax.text(cx, method_y, stage["method"],
            ha="center", va="center",
            fontsize=10.5, style="italic", color="#333333", zorder=3)

    # Params line (small, muted)
    params_y = y + h * 0.18
    ax.text(cx, params_y, stage["params"],
            ha="center", va="center",
            fontsize=9.5, color="#555555", zorder=3)


def _draw_arrow(ax, x_start, y, x_end, label: str | None,
                color: str = "#444444"):
    arr = FancyArrowPatch(
        (x_start, y), (x_end, y),
        arrowstyle="-|>,head_length=10,head_width=7",
        mutation_scale=1.0,
        linewidth=2.0,
        color=color,
        shrinkA=0, shrinkB=0,
        zorder=2,
    )
    ax.add_patch(arr)

    if label:
        midx = (x_start + x_end) / 2
        ax.text(midx, y + 1.1, label,
                ha="center", va="bottom",
                fontsize=8.5, style="italic", color="#666666", zorder=3)


def render_paper_flowchart(output_path: Path,
                           include_io: bool = True,
                           show_arrow_labels: bool = False,
                           dpi: int = 300) -> None:
    _set_pres_rc()

    fig_w, fig_h = 19.5, 4.8
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    ax.set_aspect("equal")
    ax.set_axis_off()

    # Coordinate system: 0..100 wide, height tracks figure aspect
    canvas_w = 100
    canvas_h = canvas_w * fig_h / fig_w   # keep equal aspect undistorted
    ax.set_xlim(0, canvas_w)
    ax.set_ylim(0, canvas_h)

    # Layout planning — give plenty of arrow room so labels never overlap boxes
    n = len(STAGES)
    margin = 1.0
    io_w = 8.0 if include_io else 0
    arrow_w = 7.5
    box_h = 14
    box_y = (canvas_h - box_h) / 2

    # Figure out box width
    available = canvas_w - 2 * margin - 2 * io_w - (n - 1) * arrow_w \
                - (2 * arrow_w if include_io else 0)
    box_w = available / n

    cur_x = margin
    box_xs = []

    # Input bubble
    if include_io:
        in_w = io_w
        in_h = 7
        in_y = box_y + (box_h - in_h) / 2
        in_box = FancyBboxPatch(
            (cur_x, in_y), in_w, in_h,
            boxstyle="round,pad=0.0,rounding_size=2.5",
            facecolor="#f5f5f4", edgecolor="#888888", linewidth=1.0,
            zorder=2,
        )
        ax.add_patch(in_box)
        ax.text(cur_x + in_w / 2, in_y + in_h * 0.66, "RGB-D frames",
                ha="center", va="center",
                fontsize=10.5, fontweight="bold", color="#222222")
        ax.text(cur_x + in_w / 2, in_y + in_h * 0.32, "+ scene mesh",
                ha="center", va="center",
                fontsize=9, color="#666666")
        cur_x += in_w
        # Arrow from input to stage 1
        _draw_arrow(ax, cur_x + 0.4, canvas_h / 2,
                    cur_x + arrow_w - 0.4, label=None)
        cur_x += arrow_w

    arrow_labels = (ARROW_LABELS_VERBOSE if show_arrow_labels
                    else ARROW_LABELS_DEFAULT)

    # Stage boxes + inter-stage arrows
    for i, stage in enumerate(STAGES):
        _draw_box(ax, cur_x, box_y, box_w, box_h, stage)
        box_xs.append((cur_x, cur_x + box_w))
        cur_x += box_w
        if i < n - 1:
            _draw_arrow(ax, cur_x + 0.4, canvas_h / 2,
                        cur_x + arrow_w - 0.4,
                        label=arrow_labels[i] or None)
            cur_x += arrow_w

    # Output bubble
    if include_io:
        # Arrow from stage 4 to output
        _draw_arrow(ax, cur_x + 0.4, canvas_h / 2,
                    cur_x + arrow_w - 0.4,
                    label=("caption + pose" if show_arrow_labels else None))
        cur_x += arrow_w
        out_w = io_w
        out_h = 7
        out_y = box_y + (box_h - out_h) / 2
        out_box = FancyBboxPatch(
            (cur_x, out_y), out_w, out_h,
            boxstyle="round,pad=0.0,rounding_size=2.5",
            facecolor="#eef7ee", edgecolor="#3a8a3a", linewidth=1.0,
            zorder=2,
        )
        ax.add_patch(out_box)
        ax.text(cur_x + out_w / 2, out_y + out_h * 0.66, "LangLoc",
                ha="center", va="center",
                fontsize=10.5, fontweight="bold", color="#1f5b1f")
        ax.text(cur_x + out_w / 2, out_y + out_h * 0.32, "13K descriptions",
                ha="center", va="center",
                fontsize=9, color="#3a8a3a")

    # Save
    fig.savefig(output_path, format="pdf", dpi=dpi,
                bbox_inches="tight", pad_inches=0.05)
    fig.savefig(output_path.with_suffix(".png"), format="png", dpi=dpi,
                bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"Saved: {output_path}")
    print(f"Saved: {output_path.with_suffix('.png')}")


def parse_args():
    ap = argparse.ArgumentParser(
        description="Paper-style flowchart of the LangLoc data pipeline.")
    ap.add_argument("--output", type=Path, default=Path("docs/figures"))
    ap.add_argument("--no-io", "--no_io", dest="no_io", action="store_true",
                    help="Skip the input/output bubbles (just the 4 stage boxes).")
    ap.add_argument("--labels", action="store_true",
                    help="Show inter-stage data-flow labels above the arrows.")
    return ap.parse_args()


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    render_paper_flowchart(
        output_path=args.output / "pipeline_flowchart.pdf",
        include_io=not args.no_io,
        show_arrow_labels=args.labels,
    )


if __name__ == "__main__":
    main()
