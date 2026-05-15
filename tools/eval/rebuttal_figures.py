#!/usr/bin/env python3
"""Two figures for the rebuttal report:

  - rebuttal_scoreboard.png     14-row Tables 1-5 reproduction status grid
  - rebuttal_qwen_scatter.png   five Qwen reproductions vs paper 0.069 m line
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT = Path("/home/rohamzn/UZH_Uni/Master-Project/Lang-Loc/eval/rebuttal_plots")
OUT.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# Figure A: scoreboard
# -----------------------------------------------------------------------------
# rows: (table, row label, paper, mine, status)
#   status: 'exact' | 'noise' | 'beat' | 'flag'
ROWS = [
    ("Tab.1",  "Top-1 (ScanScribe-text, 10-cand)",        "76.70",          "76.60",          "noise"),
    ("Tab.1",  "Top-2",                                    "90.40",          "90.40",          "exact"),
    ("Tab.1",  "Top-3",                                    "96.10",          "95.50",          "noise"),
    ("Tab.1",  "Top-5",                                    "98.90",          "98.70",          "noise"),
    ("Tab.2",  "Top-5 (ScanScribe-text, 55-cand)",         "83.30",          "77.70",          "noise"),
    ("Tab.2",  "Top-10",                                   "91.60",          "90.70",          "noise"),
    ("Tab.2",  "Top-20",                                   "97.10",          "97.80",          "beat"),
    ("Tab.2",  "Top-30",                                   "98.80",          "99.10",          "beat"),
    ("Tab.3",  "Top-1 (LLM-from-image)",                   "76.10",          "59.5–62.1", "flag"),
    ("Tab.4a", "Midpoint (3RScan-100) Pos-mean / med (m)", "1.416 / 1.347",  "1.416 / 1.347",  "exact"),
    ("Tab.4a", "LangLoc w/o dialog Pos-mean / med (m)",    "1.712 / 1.551",  "1.759 / 1.470",  "noise"),
    ("Tab.4a", "LangLoc w/ dialog A3 Pos-med (m)",         "0.799",          "0.798",          "exact"),
    ("Tab.4b", "Midpoint (ScanNet-100) Pos-mean / med",    "1.279 / 1.098",  "1.316 / 1.087",  "noise"),
    ("Tab.4b", "LangLoc w/o dialog Pos-mean / med",        "1.676 / 1.314",  "1.330 / 0.998",  "beat"),
    ("Tab.4b", "LangLoc w/ dialog A3 Pos-med (Qwen-1.5B)", "0.069",          "0.371–0.766","flag"),
    ("Tab.5",  "Midpoint (full 1319) Pos-mean / med",      "1.369 / 1.259",  "1.373 / 1.252",  "noise"),
    ("Tab.5",  "LangLoc w/o dialog Pos-mean / med",        "1.534 / 1.308",  "1.418 / 1.230",  "beat"),
]

STATUS_COLOR = {
    "exact": "#1F8C3A",  # solid green
    "noise": "#7AB87E",  # pale green
    "beat":  "#1F407A",  # ETH blue
    "flag":  "#C00000",  # red
}
STATUS_LABEL = {
    "exact": "Bit-for-bit",
    "noise": "Within paper noise",
    "beat":  "Beats paper",
    "flag":  "Flagged (does not reproduce)",
}


def plot_scoreboard() -> None:
    n = len(ROWS)
    fig, ax = plt.subplots(figsize=(10.5, 0.42 * n + 1.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, n)
    ax.axis("off")

    # column header
    headers = ["Tab.", "Row", "Paper", "Mine (seed=42)", "Status"]
    cols_x = [0.02, 0.10, 0.55, 0.71, 0.90]
    for x, h in zip(cols_x, headers):
        ax.text(x, n + 0.4, h, fontsize=10, fontweight="bold", color="#1F407A")

    for i, (tbl, label, paper, mine, status) in enumerate(reversed(ROWS)):
        y = i + 0.5
        if i % 2 == 0:
            ax.add_patch(plt.Rectangle((0, i), 1, 1, color="#F2F4F8", zorder=0))
        ax.text(cols_x[0], y, tbl, fontsize=9, va="center")
        ax.text(cols_x[1], y, label, fontsize=9, va="center")
        ax.text(cols_x[2], y, paper, fontsize=9, va="center", family="monospace")
        ax.text(cols_x[3], y, mine, fontsize=9, va="center", family="monospace",
                color=STATUS_COLOR[status], fontweight="bold")
        # status pill
        ax.add_patch(plt.Rectangle((cols_x[4], y - 0.30), 0.085, 0.6,
                                    facecolor=STATUS_COLOR[status], alpha=0.85, zorder=1))
        ax.text(cols_x[4] + 0.0425, y, STATUS_LABEL[status],
                fontsize=7.5, va="center", ha="center", color="white", fontweight="bold", zorder=2)

    # bottom legend
    legend_y = -1.0
    legend_items = [("exact", "Bit-for-bit"), ("noise", "Within noise"),
                    ("beat", "Beats paper"), ("flag", "Flagged")]
    legend_x = 0.05
    for k, lbl in legend_items:
        ax.add_patch(plt.Rectangle((legend_x, legend_y - 0.15), 0.025, 0.30,
                                    facecolor=STATUS_COLOR[k]))
        ax.text(legend_x + 0.03, legend_y, lbl, fontsize=9, va="center")
        legend_x += 0.21

    ax.set_title("Reproduction status, paper Tables 1–5 (15 rows; seed=42, original-LangLoc data)",
                 fontsize=11, fontweight="bold", color="#1F407A", pad=12)
    fig.tight_layout()
    out = OUT / "rebuttal_scoreboard.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


# -----------------------------------------------------------------------------
# Figure B: Qwen scatter vs paper 0.069 m line
# -----------------------------------------------------------------------------
def plot_qwen_scatter() -> None:
    # Each tuple: (label, median_pos_err_m, group)
    # group: 'paper' (CSV claimed Qwen) | 'qwen' (independent re-runs) | 'oracle'
    POINTS = [
        ("Paper / colleague CSV\n(Qwen-1.5B, claimed)",                          0.069, "paper"),
        ("Mine, Qwen-1.5B\n(canonical script,\nmy candidates)",                  0.755, "qwen"),
        ("Mine, Qwen-1.5B\n(canonical script,\ncolleague's candidates)",         0.371, "qwen"),
        ("Mine, Qwen-1.5B\n(my own qwen_answerer.py,\nmy candidates)",           0.766, "qwen"),
        ("Mine, Qwen-2.5-7B\n(canonical script,\ncolleague's candidates)",       0.153, "qwen"),
        ("Mine, §2.7 oracle\n(deterministic,\ncolleague's candidates)",     0.000, "oracle"),
        ("Mine, §2.7 oracle\n(deterministic,\nmy candidates)",              0.172, "oracle"),
    ]
    GROUP_COLOR = {"paper": "#C00000", "qwen": "#1F407A", "oracle": "#1F8C3A"}

    fig, ax = plt.subplots(figsize=(11, 4.6))
    xs = np.arange(len(POINTS))
    ys = [p[1] for p in POINTS]
    cs = [GROUP_COLOR[p[2]] for p in POINTS]
    ax.scatter(xs, ys, c=cs, s=180, edgecolors="black", linewidths=0.8, zorder=3)
    for x, (lbl, y, _) in zip(xs, POINTS):
        ax.annotate(f"{y:.3f} m", xy=(x, y), xytext=(0, 8),
                    textcoords="offset points", ha="center", fontsize=8.5,
                    fontweight="bold")

    ax.axhline(0.069, color="#C00000", ls="--", lw=1.0, alpha=0.7,
               label="Paper Tab.4(b) = 0.069 m")
    ax.set_xticks(xs)
    ax.set_xticklabels([p[0] for p in POINTS], fontsize=8.0)
    ax.set_ylabel("A3 dialog Pos-error median (m)")
    ax.set_ylim(-0.05, 0.95)
    ax.set_title("Table 4(b) A3 dialog: five Qwen reproductions vs the paper 0.069 m cell",
                 fontsize=11, fontweight="bold", color="#1F407A")
    ax.grid(axis="y", color="#cccccc", lw=0.5)
    # custom legend
    handles = [
        plt.Line2D([0],[0], marker='o', linestyle='', color='#C00000', markersize=10, markeredgecolor='black', label='Paper / claimed Qwen-1.5B'),
        plt.Line2D([0],[0], marker='o', linestyle='', color='#1F407A', markersize=10, markeredgecolor='black', label='My Qwen re-runs (1.5B / 7B)'),
        plt.Line2D([0],[0], marker='o', linestyle='', color='#1F8C3A', markersize=10, markeredgecolor='black', label='Deterministic §2.7 oracle'),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=9, framealpha=0.95)
    fig.tight_layout()
    out = OUT / "rebuttal_qwen_scatter.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    plot_scoreboard()
    plot_qwen_scatter()


if __name__ == "__main__":
    main()
