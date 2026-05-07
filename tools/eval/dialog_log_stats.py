#!/usr/bin/env python3
"""Extract dialog episode statistics from runner stdout logs.

Reviewer hook: WxoL — *"the dialog results are not sufficient evidence
of real interactive localization"*.  Per-scene round counts +
posterior-concentration trajectory let us answer "how many questions
does the system actually need" — a question the paper does not
quantitatively report.

Inputs: `/tmp/dialog_3rscan.log` and `/tmp/dialog_scannet.log`
(produced by `python -m langloc.dialogue.cli ... eval_mode=sequential
answer_mode=oracle`).

Outputs: per-backend statistics across the 100-scene paper subset.
"""
from __future__ import annotations

import re
import statistics
from pathlib import Path
from typing import Dict, List, Tuple

ROUND_RE = re.compile(r"^\[(?P<bk>A[123])\]\s+Round\s+(?P<r>\d+)\s*\|\s*topP=(?P<p>[\d.]+)", re.MULTILINE)
QASKED_RE = re.compile(r"^\[(?P<bk>A[123])\]\s+Questions asked:\s*(?P<n>\d+)", re.MULTILINE)


def parse(path: Path) -> Dict[str, Dict[str, List]]:
    """Return per-backend dicts of round counts + final topP per scene."""
    txt = path.read_text()
    # Each scene has 3 backend blocks — split by `[A1] Round 1` boundaries
    # would lose A2/A3 boundaries.  Instead parse linearly and bucket
    # by backend, with scene boundary inferred by "Questions asked".
    backends = {"A1": {"rounds": [], "qasked": [], "final_topP": []},
                "A2": {"rounds": [], "qasked": [], "final_topP": []},
                "A3": {"rounds": [], "qasked": [], "final_topP": []}}

    # All round entries
    rounds_seen: Dict[str, List[Tuple[int, float]]] = {bk: [] for bk in backends}
    for m in ROUND_RE.finditer(txt):
        bk = m.group("bk")
        r = int(m.group("r"))
        p = float(m.group("p"))
        rounds_seen[bk].append((r, p))

    # All "Questions asked: N" entries — one per scene per backend
    for m in QASKED_RE.finditer(txt):
        bk = m.group("bk")
        n = int(m.group("n"))
        backends[bk]["qasked"].append(n)

    # For each "Questions asked" we want the final topP just before it.
    # Linear scan: every time we see `[bk] Round R | topP=p`, remember p
    # for that backend.  When we see `[bk] Questions asked: n`, record
    # (n, last_topP[bk]).
    last_topP: Dict[str, float] = {}
    last_round: Dict[str, int] = {}
    counters: Dict[str, int] = {bk: 0 for bk in backends}
    for line in txt.splitlines():
        m = re.match(r"^\[(?P<bk>A[123])\]\s+Round\s+(?P<r>\d+)\s*\|\s*topP=(?P<p>[\d.]+)", line)
        if m:
            bk = m.group("bk")
            last_round[bk] = int(m.group("r"))
            last_topP[bk] = float(m.group("p"))
            continue
        m = re.match(r"^\[(?P<bk>A[123])\]\s+Questions asked:\s*(?P<n>\d+)", line)
        if m:
            bk = m.group("bk")
            n = int(m.group("n"))
            backends[bk]["rounds"].append(last_round.get(bk, n))
            backends[bk]["final_topP"].append(last_topP.get(bk, float("nan")))
            # Reset round/topP so next scene's first round overwrites
            last_topP[bk] = float("nan")
            last_round[bk] = 0
            counters[bk] += 1

    return backends


def fmt(stats: Dict[str, Dict[str, List]], dataset: str) -> str:
    out = [f"### {dataset}", ""]
    out.append(f"{'Backend':<18}{'N scenes':>10}{'Mean Q':>10}{'Med Q':>8}{'Max Q':>8}{'P(stop early)':>16}{'Mean final topP':>18}")
    out.append("-" * 90)
    for bk in ("A1", "A2", "A3"):
        s = stats[bk]
        n = len(s["qasked"])
        if n == 0:
            continue
        qa = s["qasked"]
        topP = [p for p in s["final_topP"] if p == p]
        # Per Master §A.5.2: τ_conf = 0.85, max_rounds = 12
        early = sum(1 for q in qa if q < 12) / n
        out.append(
            f"{bk:<18}{n:>10}{statistics.mean(qa):>10.2f}{statistics.median(qa):>8.0f}"
            f"{max(qa):>8}{early*100:>15.1f}%"
            f"{statistics.mean(topP) if topP else float('nan'):>18.3f}"
        )
    return "\n".join(out)


def main() -> None:
    print("=" * 90)
    print("Dialog episode statistics (per-backend, 100-scene subsets)")
    print("Source: /tmp/dialog_{3rscan,scannet}.log — answer_mode=oracle, seed=42")
    print("=" * 90)
    print()
    print(fmt(parse(Path("/tmp/dialog_3rscan.log")), "3RScan-100"))
    print()
    print(fmt(parse(Path("/tmp/dialog_scannet.log")), "ScanNet-100"))
    print()
    print("Notes:")
    print("- Mean Q = mean number of dialog questions per scene before stop.")
    print("- P(stop early) = fraction of episodes that hit τ_conf=0.85 before max_rounds=12.")
    print("- Mean final topP = mean posterior mass on the MAP frame at termination.")


if __name__ == "__main__":
    main()
