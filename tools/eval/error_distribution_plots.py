#!/usr/bin/env python3
"""Error-distribution visualisations for the rebuttal.

Reviewer hook: EEmB —
  "Mean position error is high (~half the midpoint error) — likely
   skewed distribution.  It would be good to have more visualizations
   of this.  I suspect that, when the system identifies the correct
   cluster, the errors are very low; but when the incorrect cluster
   is found, then it is very high."

Produces four panel families per dataset:

  1. Position-error KDE per method (with mean and median annotated).
  2. Position-error CDF per method (Recall@τ as a curve).
  3. Distance-error × angular-error scatter (separates wrong-room
     failures from near-but-wrong-heading failures).
  4. Mean-vs-median bar chart per method to make the skew obvious.

Reads the same cached eval JSONs + dialog stdout logs as
`tools/eval/recall_at_threshold.py`; no new model runs.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
EVAL = REPO / "eval"
OUT = EVAL / "rebuttal_plots"
OUT.mkdir(parents=True, exist_ok=True)


def load_per_query(path: Path) -> List[Tuple[float, float]]:
    d = json.loads(path.read_text())
    items = d["metrics"] if isinstance(d, dict) and "metrics" in d else d
    out: List[Tuple[float, float]] = []
    for it in items:
        de = it.get("distance_error")
        ae = it.get("angular_error_deg")
        if de is None:
            continue
        out.append((float(de), float(ae) if ae is not None else float("nan")))
    return out


DIALOG_RE = re.compile(
    r"^\[(?P<bk>A[123])\]\s+(?P<kind>MAP|Mean)\s*:\s*pos_err=(?P<pos>[-\d.]+)\s*m\s*\|\s*rot_err=(?P<rot>[-\d.]+|nan)\s*deg",
    re.MULTILINE,
)


def parse_dialog(path: Path, backend: str = "A3", kind: str = "MAP") -> List[Tuple[float, float]]:
    txt = path.read_text()
    out = []
    for m in DIALOG_RE.finditer(txt):
        if m.group("bk") != backend or m.group("kind") != kind:
            continue
        pos = float(m.group("pos"))
        rot = float("nan") if m.group("rot") == "nan" else float(m.group("rot"))
        out.append((pos, rot))
    return out


COLOR = {
    "Midpoint":             "#888888",
    "LangLoc w/o dialog":   "#1f77b4",
    "LangLoc w/ dialog":    "#d62728",
}


def gather(dataset: str) -> Dict[str, List[Tuple[float, float]]]:
    if dataset == "3rscan_100":
        return {
            "Midpoint":           load_per_query(EVAL / "midpoint_3rscan_metrics.json"),
            "LangLoc w/o dialog": load_per_query(EVAL / "eval_metrics_table4_parsed.json"),
            "LangLoc w/ dialog":  parse_dialog(Path("/tmp/dialog_3rscan.log"), "A3", "MAP"),
        }
    if dataset == "scannet_100":
        return {
            "Midpoint":           load_per_query(EVAL / "midpoint_scannet_metrics.json"),
            "LangLoc w/o dialog": load_per_query(EVAL / "eval_metrics_table4_scannet_parsed.json"),
            "LangLoc w/ dialog":  parse_dialog(Path("/tmp/dialog_scannet.log"), "A3", "MAP"),
        }
    if dataset == "full_1319":
        return {
            "Midpoint":           load_per_query(EVAL / "baseline_eval_metrics_mid_point.json"),
            "LangLoc w/o dialog": load_per_query(EVAL / "eval_metrics_table5.json"),
        }
    raise ValueError(dataset)


def plot_kde_position(ax, samples: Dict[str, List[Tuple[float, float]]], xmax: float = 6.0) -> None:
    """Position-error KDE per method, with mean (dashed) and median (solid) marks."""
    grid = np.linspace(0, xmax, 600)
    for name, data in samples.items():
        if not data:
            continue
        pos = np.array([p for p, _ in data])
        # Silverman bandwidth
        n = len(pos)
        sigma = pos.std(ddof=1) if n > 1 else 1.0
        bw = max(1.06 * sigma * n ** (-1 / 5), 1e-3)
        kde = np.exp(-((grid[:, None] - pos[None, :]) ** 2) / (2 * bw * bw))
        density = kde.mean(axis=1) / (bw * (2 * np.pi) ** 0.5)
        c = COLOR[name]
        ax.plot(grid, density, label=name, color=c, lw=2)
        ax.axvline(pos.mean(), color=c, ls="--", lw=1, alpha=0.6)
        ax.axvline(np.median(pos), color=c, ls=":", lw=1, alpha=0.9)
    ax.set_xlim(0, xmax)
    ax.set_xlabel("Position error (m)")
    ax.set_ylabel("Density")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("Position-error KDE (--- mean,  · · · median)")


def plot_cdf(ax, samples: Dict[str, List[Tuple[float, float]]], xmax: float = 5.0) -> None:
    grid = np.linspace(0, xmax, 1200)
    for name, data in samples.items():
        if not data:
            continue
        pos = np.array(sorted(p for p, _ in data))
        n = len(pos)
        cdf = np.searchsorted(pos, grid, side="right") / n
        ax.plot(grid, cdf, label=name, color=COLOR[name], lw=2)
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, 1)
    for tau in (0.3, 0.5, 1.0):
        ax.axvline(tau, color="grey", ls=":", lw=0.6)
    ax.set_xlabel("Position error (m)")
    ax.set_ylabel("Recall (= CDF)")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_title("Position-error CDF / Recall@τ_pos")


def plot_scatter(ax, samples: Dict[str, List[Tuple[float, float]]],
                 xmax: float = 5.0) -> None:
    for name, data in samples.items():
        if not data:
            continue
        valid = [(p, r) for p, r in data if r == r]  # drop NaN angles
        if not valid:
            continue
        pos = np.array([p for p, _ in valid])
        rot = np.array([r for _, r in valid])
        ax.scatter(pos, rot, label=name, color=COLOR[name], alpha=0.5, s=14)
    ax.axhline(30, color="grey", ls=":", lw=0.6)
    ax.axvline(0.3, color="grey", ls=":", lw=0.6)
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, 180)
    ax.set_xlabel("Position error (m)")
    ax.set_ylabel("Angular error (°)")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("dist × ang per query (gates: 0.3 m / 30°)")


def plot_mean_median(ax, samples: Dict[str, List[Tuple[float, float]]]) -> None:
    names = list(samples.keys())
    means = [np.mean([p for p, _ in samples[n]]) if samples[n] else float("nan") for n in names]
    meds = [np.median([p for p, _ in samples[n]]) if samples[n] else float("nan") for n in names]
    x = np.arange(len(names))
    w = 0.36
    ax.bar(x - w / 2, means, w, label="Mean", color="#888888")
    ax.bar(x + w / 2, meds, w, label="Median", color="#d62728")
    for i, (m, md) in enumerate(zip(means, meds)):
        ax.text(i - w / 2, m, f"{m:.2f}", ha="center", va="bottom", fontsize=8)
        ax.text(i + w / 2, md, f"{md:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace(" w/", "\nw/") for n in names], fontsize=8)
    ax.set_ylabel("Position error (m)")
    ax.legend()
    ax.set_title("Mean vs median (skew = mean − median)")


def render_dataset(dataset: str, samples: Dict[str, List[Tuple[float, float]]],
                   xmax_kde: float = 6.0) -> None:
    fig, axs = plt.subplots(2, 2, figsize=(12, 9))
    plot_kde_position(axs[0, 0], samples, xmax=xmax_kde)
    plot_cdf(axs[0, 1], samples, xmax=5.0)
    plot_scatter(axs[1, 0], samples, xmax=5.0)
    plot_mean_median(axs[1, 1], samples)
    fig.suptitle(f"Error distribution — {dataset}", fontsize=13)
    fig.tight_layout()
    out = OUT / f"error_distribution_{dataset}.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    for ds in ("3rscan_100", "scannet_100", "full_1319"):
        s = gather(ds)
        # Skip empty rows so the legend isn't cluttered
        s = {k: v for k, v in s.items() if v}
        render_dataset(ds, s)


if __name__ == "__main__":
    main()
