#!/usr/bin/env python3
"""Compute Recall@(τ_pos, τ_ang) over cached eval JSONs and dialog logs.

Reviewer hook: EEmB — "Recall@(0.3 m / 30°) accuracy + AUC sweep".

Output: a table with one row per method × dataset and accuracy at
{(0.1m,10°), (0.3m,30°), (0.5m,30°), (1.0m,30°), (2.0m,60°)}.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List, Tuple

REPO = Path(__file__).resolve().parents[2]
EVAL = REPO / "eval"


def load_per_query(path: Path) -> List[Tuple[float, float]]:
    """Return list of (distance_error, angular_error_deg) per query."""
    d = json.loads(path.read_text())
    if "metrics" in d:
        items = d["metrics"]
    elif isinstance(d, list):
        items = d
    else:
        raise ValueError(f"unknown schema in {path}")
    out = []
    for it in items:
        de = it.get("distance_error")
        ae = it.get("angular_error_deg")
        if de is None:
            continue
        if ae is None:
            ae = float("nan")
        out.append((float(de), float(ae)))
    return out


# Dialog log lines look like:
#   [A3] MAP: pos_err=0.043 m | rot_err=12.3 deg
DIALOG_RE = re.compile(
    r"^\[(?P<bk>A[123])\]\s+(?P<kind>MAP|Mean)\s*:\s*pos_err=(?P<pos>[-\d.]+)\s*m\s*\|\s*rot_err=(?P<rot>[-\d.]+|nan)\s*deg",
    re.MULTILINE,
)


def parse_dialog_log(path: Path, backend: str = "A3", kind: str = "MAP") -> List[Tuple[float, float]]:
    """Extract (pos_err, rot_err) per entry for one backend / aggregation."""
    txt = path.read_text()
    out: List[Tuple[float, float]] = []
    for m in DIALOG_RE.finditer(txt):
        if m.group("bk") != backend or m.group("kind") != kind:
            continue
        pos = float(m.group("pos"))
        rot_s = m.group("rot")
        rot = float("nan") if rot_s == "nan" else float(rot_s)
        out.append((pos, rot))
    return out


def recall_at(samples: List[Tuple[float, float]], tau_pos: float, tau_ang: float) -> float:
    if not samples:
        return float("nan")
    hits = 0
    n_pos_only = 0
    for pos, ang in samples:
        if pos <= tau_pos:
            n_pos_only += 1
            if ang <= tau_ang or ang != ang:  # NaN angle counted-as-fail
                if ang == ang:
                    hits += 1
    return hits / len(samples)


def recall_at_pos_only(samples, tau_pos):
    if not samples:
        return float("nan")
    return sum(1 for p, _ in samples if p <= tau_pos) / len(samples)


def report(rows):
    thresholds = [
        (0.1, 10.0), (0.3, 30.0), (0.5, 30.0), (1.0, 30.0), (2.0, 60.0),
    ]
    print()
    print(f"{'Method':<35} {'N':>5}", end="")
    for tp, ta in thresholds:
        print(f"  R@{tp}m/{int(ta)}°", end="")
    print()
    print("-" * 100)
    for name, samples in rows:
        print(f"{name:<35} {len(samples):>5}", end="")
        for tp, ta in thresholds:
            r = recall_at(samples, tp, ta)
            print(f"  {r*100:>9.1f}%", end="")
        print()


def report_pos_only(rows):
    thresholds = [0.3, 0.5, 1.0, 2.0]
    print()
    print(f"{'Method (pos-only)':<35} {'N':>5}", end="")
    for tp in thresholds:
        print(f"   R@{tp}m", end="")
    print()
    print("-" * 80)
    for name, samples in rows:
        print(f"{name:<35} {len(samples):>5}", end="")
        for tp in thresholds:
            r = recall_at_pos_only(samples, tp)
            print(f"  {r*100:>6.1f}%", end="")
        print()


def main():
    print("=" * 80)
    print("Recall@(τ_pos, τ_ang) — cached LangLoc eval data")
    print("=" * 80)

    rows_3rscan_100 = []
    rows_3rscan_100.append(
        ("Midpoint", load_per_query(EVAL / "midpoint_3rscan_metrics.json"))
    )
    rows_3rscan_100.append(
        ("LangLoc w/o dialog", load_per_query(EVAL / "eval_metrics_table4_parsed.json"))
    )
    rows_3rscan_100.append(
        ("LangLoc w/ dialog (A3 MAP)", parse_dialog_log(Path("/tmp/dialog_3rscan.log"), "A3", "MAP"))
    )
    rows_3rscan_100.append(
        ("LangLoc w/ dialog (A3 Mean)", parse_dialog_log(Path("/tmp/dialog_3rscan.log"), "A3", "Mean"))
    )

    rows_scannet_100 = []
    rows_scannet_100.append(
        ("Midpoint", load_per_query(EVAL / "midpoint_scannet_metrics.json"))
    )
    rows_scannet_100.append(
        ("LangLoc w/o dialog", load_per_query(EVAL / "eval_metrics_table4_scannet_parsed.json"))
    )
    rows_scannet_100.append(
        ("LangLoc w/ dialog (A3 MAP)", parse_dialog_log(Path("/tmp/dialog_scannet.log"), "A3", "MAP"))
    )
    rows_scannet_100.append(
        ("LangLoc w/ dialog (A3 Mean)", parse_dialog_log(Path("/tmp/dialog_scannet.log"), "A3", "Mean"))
    )

    rows_full = []
    # Midpoint full 1319 — saved as eval/baseline_eval_metrics_mid_point.json
    # (this gets overwritten by each midpoint run; latest is full 1319)
    rows_full.append(
        ("Midpoint", load_per_query(EVAL / "baseline_eval_metrics_mid_point.json"))
    )
    rows_full.append(
        ("LangLoc w/o dialog", load_per_query(EVAL / "eval_metrics_table5.json"))
    )

    print("\n### 3RScan 100-scene paper subset")
    report(rows_3rscan_100)
    print("\nPos-only (no angle gate):")
    report_pos_only(rows_3rscan_100)

    print("\n### ScanNet 100-scene paper subset")
    report(rows_scannet_100)
    print("\nPos-only:")
    report_pos_only(rows_scannet_100)

    print("\n### Full LangLoc dataset (3RScan, 1319 scenes)")
    report(rows_full)
    print("\nPos-only:")
    report_pos_only(rows_full)


if __name__ == "__main__":
    main()
