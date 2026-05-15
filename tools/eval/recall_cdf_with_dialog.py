#!/usr/bin/env python3
"""Position-error CDF / Recall@τ_pos with the four series the supervisors
want for the rebuttal:

  - Midpoint  (no language)
  - Qwen baseline (VLM Qwen2.5-VL-2B with topdown image)
  - LangLoc (= LangLoc with dialog, A3 MAP position)
  - LangLoc top-10 oracle (= minimum error among the top-10 highest-scoring grid cells)

Output: eval/rebuttal_plots/recall_cdf_3rscan_subset.png
        eval/rebuttal_plots/recall_cdf_scannet_subset.png
        eval/rebuttal_plots/recall_cdf_combined.png   (figure 5 layout)
"""
from __future__ import annotations

import json
import re
import statistics
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

REPO = Path("/home/rohamzn/UZH_Uni/Master-Project/Lang-Loc")
EVAL_NEW = REPO / "eval" / "new_data"
WHEREAMI_EVAL = REPO / "whereami-text2sgm" / "playground" / "graph_models" / "models" / "eval"
OUT = REPO / "eval" / "rebuttal_plots"
OUT.mkdir(parents=True, exist_ok=True)


def load_pos_errors(path: Path, key: str = "distance_error") -> List[float]:
    """Return per-entry pos errors from a metrics JSON."""
    d = json.loads(path.read_text())
    items = d["metrics"] if isinstance(d, dict) and "metrics" in d else d
    out = []
    for it in items:
        v = it.get(key)
        if v is None or v != v:
            continue
        out.append(float(v))
    return out


DIALOG_RE = re.compile(
    r"^\[(?P<bk>A[123])\]\s+(?P<kind>MAP|Mean)\s*:\s*pos_err=(?P<pos>[-\d.]+)\s*m\s*\|\s*rot_err=(?P<rot>[-\d.]+|nan)\s*deg",
    re.MULTILINE,
)


def parse_dialog_pos(path: Path, backend: str = "A3", kind: str = "MAP") -> List[float]:
    txt = path.read_text()
    out = []
    for m in DIALOG_RE.finditer(txt):
        if m.group("bk") == backend and m.group("kind") == kind:
            out.append(float(m.group("pos")))
    return out


COLOR = {
    "Midpoint":              "#888888",
    "Qwen baseline":         "#2ca02c",
    "LangLoc":               "#1f77b4",
    "LangLoc top-10 oracle": "#ff7f0e",
}


def gather(dataset: str) -> Dict[str, List[float]]:
    if dataset == "3rscan":
        return {
            "Midpoint":              load_pos_errors(EVAL_NEW / "midpoint_3rscan_100_NEW.json"),
            "Qwen baseline":         load_pos_errors(WHEREAMI_EVAL / "baseline_eval_metrics_qwen_3rscan_subset.json"),
            "LangLoc":               parse_dialog_pos(EVAL_NEW / "dialog_3rscan_100_NEW.log", "A3", "MAP"),
            "LangLoc top-10 oracle": load_pos_errors(EVAL_NEW / "eval_metrics_table4_3rscan_parsed_NEW.json", key="topk_min_dist"),
        }
    if dataset == "scannet":
        return {
            "Midpoint":              load_pos_errors(EVAL_NEW / "midpoint_scannet_100_NEW.json"),
            "Qwen baseline":         load_pos_errors(WHEREAMI_EVAL / "baseline_eval_metrics_qwen_scannet.json"),
            "LangLoc":               parse_dialog_pos(EVAL_NEW / "dialog_scannet_100_NEW.log", "A3", "MAP"),
            "LangLoc top-10 oracle": load_pos_errors(EVAL_NEW / "eval_metrics_table4_scannet_parsed_NEW.json", key="topk_min_dist"),
        }
    raise ValueError(dataset)


def plot_cdf_panel(ax, samples: Dict[str, List[float]], title: str, xmax: float = 5.0) -> None:
    grid = np.linspace(0, xmax, 1200)
    for name in ("Midpoint", "Qwen baseline", "LangLoc", "LangLoc top-10 oracle"):
        data = samples.get(name, [])
        if not data:
            continue
        pos = np.array(sorted(data))
        n = len(pos)
        cdf = np.searchsorted(pos, grid, side="right") / n
        ax.plot(grid, cdf, label=name, color=COLOR[name], lw=2)
    for tau in (0.3, 0.5, 1.0):
        ax.axvline(tau, color="grey", ls=":", lw=0.6)
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Position error (m)")
    ax.set_ylabel("Recall (= CDF)")
    ax.set_title(f"Position-error CDF / Recall@τ_pos\n{title}")
    ax.legend(loc="lower right", fontsize=9)


def main() -> None:
    samples_3r = gather("3rscan")
    samples_sn = gather("scannet")

    # individual panels
    for tag, samples, title in [
        ("3rscan_subset", samples_3r, "(a) 3RScan subset"),
        ("scannet_subset", samples_sn, "(b) ScanNet subset"),
    ]:
        fig, ax = plt.subplots(figsize=(7, 5.5))
        plot_cdf_panel(ax, samples, title)
        out = OUT / f"recall_cdf_{tag}.png"
        fig.tight_layout()
        fig.savefig(out, dpi=140)
        plt.close(fig)
        print(f"wrote {out}")
        for k, v in samples.items():
            if v:
                print(f"  {tag:20s} {k:25s}  N={len(v):4d}  mean={statistics.mean(v):.3f}  median={statistics.median(v):.3f}")

    # combined figure (Fig 5 layout: two stacked panels)
    fig, axs = plt.subplots(2, 1, figsize=(7, 11))
    plot_cdf_panel(axs[0], samples_3r, "(a) 3RScan subset")
    plot_cdf_panel(axs[1], samples_sn, "(b) ScanNet subset")
    fig.tight_layout()
    out = OUT / "recall_cdf_combined.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
