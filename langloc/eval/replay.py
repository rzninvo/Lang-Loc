#!/usr/bin/env python3
"""Replay cached LLM answers through the dialogue pipeline and compute IoU.

Replays saved Q&A logs (from ``llm_answerer.py``) through the dialogue
pipeline WITHOUT re-running the LLM. Captures predicted pose vectors,
computes 3D View IoU via ``langloc.eval.view_iou``, and writes a metrics CSV.

Usage::

    python -m langloc.eval.replay \
        --qwen_json   results/llm_results.json \
        --candidates  candidates.json \
        --dataset_root /data/scans \
        --dialogue_script path/to/dialogue_entry.py \
        --output_csv  results/replay_metrics.csv
"""

from __future__ import annotations

import argparse
import builtins
import csv
import json
import math
import re
import sys
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from langloc.eval.view_iou import (
    build_iou_context,
    compute_view_iou,
    discover_mesh,
)


# ── regex ────────────────────────────────────────────────────────────────────
RE_SUMMARY = re.compile(
    r"^(A[123]):\s+MAP\(([0-9.]+)m,([0-9.nan]+)°\)\s+\|\s+Mean\(([0-9.]+)m,([0-9.nan]+)°\)",
    re.MULTILINE,
)
RE_BASELINE = re.compile(
    r"^Baseline predicted_pose:\s+pos_err=([0-9.]+)\s+m\s+\|\s+rot_err=([0-9.nan]+)\s+deg",
    re.MULTILINE,
)


def safe_float(s: str) -> Optional[float]:
    try:
        v = float(s)
        return None if v != v else v
    except (TypeError, ValueError):
        return None


# ── module importer ──────────────────────────────────────────────────────────

from langloc.eval import import_module_from_path as import_module


# ── Mesh discovery extension for ScanNet ─────────────────────────────────────

def discover_mesh_extended(scene_dir: Path) -> Path:
    """Extend mesh discovery to also find ScanNet-style *_vh_clean_2.ply files."""
    try:
        return discover_mesh(scene_dir)
    except FileNotFoundError:
        pass
    for pattern in ("*_vh_clean_2.ply", "*_vh_clean.ply", "*.ply"):
        candidates = sorted(scene_dir.glob(pattern))
        non_label = [p for p in candidates if "label" not in p.name.lower()]
        chosen = non_label or candidates
        if chosen:
            return chosen[0]
    raise FileNotFoundError(f"No mesh found in {scene_dir}")


# ── IoU computer ─────────────────────────────────────────────────────────────

class IoUComputer:
    """Cached View IoU computation wrapper."""

    def __init__(self, hfov_deg: float = 39.31, vfov_deg: float = 64.76,
                 near: float = 0.05, far: Optional[float] = None):
        self.hfov_rad = math.radians(hfov_deg)
        self.vfov_rad = math.radians(vfov_deg)
        self.near = near
        self.far = far
        self._cache: Dict[str, Any] = {}

    def _ctx(self, scene_dir: Path):
        key = str(scene_dir)
        if key not in self._cache:
            self._cache[key] = build_iou_context(scene_dir)
        return self._cache[key]

    def compute(self, scene_dir: Path, gt_pos, gt_dir, pred_pos, pred_dir) -> Optional[float]:
        try:
            ray_scene, mesh_id, tri_pts, tri_cen, tri_areas = self._ctx(scene_dir)
            return compute_view_iou(
                gt_cam=gt_pos, gt_dir=gt_dir,
                pred_cam=pred_pos, pred_dir=pred_dir,
                hfov_rad=self.hfov_rad, vfov_rad=self.vfov_rad,
                raycasting_scene=ray_scene, mesh_id=mesh_id,
                tri_points=tri_pts, tri_centroids=tri_cen, tri_areas=tri_areas,
                near=self.near, far=self.far,
            )
        except Exception:
            return None


def _dir_valid(v) -> bool:
    if v is None:
        return False
    v = np.asarray(v, dtype=np.float64).reshape(3)
    return bool(np.isfinite(v).all() and np.linalg.norm(v) > 1e-6)


# ── per-scene replay ─────────────────────────────────────────────────────────

_state: Dict[str, Any] = {
    "answer_queue": deque(),
    "last_q_line": "",
    "output_lines": [],
    "pose_store": {},
    "current_sid": "",
}


def install_pose_hooks(dlg):
    """Patch predict_pose on each backend class to capture predicted poses."""
    def _make_patched(cls, tag: str):
        orig = cls.predict_pose
        def patched(self_inner, *a, **kw):
            result = orig(self_inner, *a, **kw)
            try:
                mp, md, sp, sd = result
                _state["pose_store"][tag] = {
                    "map_pos": np.asarray(mp, dtype=np.float64).tolist() if mp is not None else None,
                    "map_dir": np.asarray(md, dtype=np.float64).tolist() if md is not None else None,
                    "mean_pos": np.asarray(sp, dtype=np.float64).tolist() if sp is not None else None,
                    "mean_dir": np.asarray(sd, dtype=np.float64).tolist() if sd is not None else None,
                }
            except Exception:
                pass
            return result
        cls.predict_pose = patched

    for cls, tag in ((dlg.CandidateBackendA1, "A1"),
                     (dlg.ParticleBackendA2, "A2"),
                     (dlg.FrameBackendA3, "A3")):
        if hasattr(cls, "predict_pose"):
            _make_patched(cls, tag)


def replay_scene(
    sid: str,
    qa_log: List[Dict],
    gt_pos: np.ndarray,
    gt_dir: np.ndarray,
    scene_dir: Path,
    dlg,
    iou_comp: Optional[IoUComputer],
    dataset_root: Path,
    candidates_json: Path,
    orig_print,
) -> Dict:
    """Re-run one scene's dialogue with cached answers; return metrics dict."""

    _state["answer_queue"] = deque(qa["answer"] for qa in qa_log)
    _state["last_q_line"] = ""
    _state["output_lines"] = []
    _state["pose_store"] = {}
    _state["current_sid"] = sid

    orig_print_builtin = builtins.print

    def hooked_print(*a, **kw):
        orig_print_builtin(*a, **kw)
        sep = kw.get("sep", " ")
        end = kw.get("end", "\n")
        line = sep.join(str(x) for x in a) + end
        _state["output_lines"].append(line)

    def hooked_input(prompt=""):
        orig_print_builtin(prompt, end="", flush=True)
        ans = _state["answer_queue"].popleft() if _state["answer_queue"] else "u"
        orig_print_builtin(f"[replay -> {ans}]")
        _state["output_lines"].append(f"[replay -> {ans}]\n")
        return ans

    builtins.print = hooked_print
    builtins.input = hooked_input

    saved_argv = sys.argv[:]
    try:
        sys.argv = [
            "dialogue_replay",
            "--candidates_json", str(candidates_json),
            "--dataset_root", str(dataset_root),
            "--only_scene_id", sid,
            "--limit", "1",
            "--eval_mode", "sequential",
            "--question_strategy", "ig",
            "--answer_mode", "interactive",
            "--max_pool_frames", "30",
            "--rel_min_answerable", "0.1",
            "--auto_relax",
            "--include_predicted_pose",
            "--show_gt_debug",
        ]
        dlg.main()
    except SystemExit:
        pass
    except Exception as exc:
        orig_print(f"  [replay error] {exc}")
    finally:
        sys.argv = saved_argv
        builtins.print = orig_print_builtin
        builtins.input = input

    stdout = "".join(_state["output_lines"])

    # Parse pos/rot from summary lines
    metrics: Dict[str, Dict] = {}
    for m in RE_SUMMARY.finditer(stdout):
        tag = m.group(1)
        metrics[(tag, "MAP")] = {
            "pos": safe_float(m.group(2)), "rot": safe_float(m.group(3)),
            "iou": None, "iou_error": None,
        }
        metrics[(tag, "MEAN")] = {
            "pos": safe_float(m.group(4)), "rot": safe_float(m.group(5)),
            "iou": None, "iou_error": None,
        }
    mb = RE_BASELINE.search(stdout)
    if mb:
        metrics[("Baseline", "predicted_pose")] = {
            "pos": safe_float(mb.group(1)), "rot": safe_float(mb.group(2)),
            "iou": None, "iou_error": None,
        }

    # Compute IoU for each backend MAP/MEAN
    if iou_comp and _dir_valid(gt_dir):
        gt_pos_arr = np.asarray(gt_pos, dtype=np.float64)
        gt_dir_arr = np.asarray(gt_dir, dtype=np.float64)
        for tag in ("A1", "A2", "A3"):
            pdata = _state["pose_store"].get(tag, {})
            for which, pos_key, dir_key in (("MAP", "map_pos", "map_dir"),
                                             ("MEAN", "mean_pos", "mean_dir")):
                p = pdata.get(pos_key)
                d = pdata.get(dir_key)
                key = (tag, which)
                if key not in metrics:
                    continue
                if p is None or not _dir_valid(d):
                    continue
                iou = iou_comp.compute(
                    scene_dir,
                    gt_pos_arr, gt_dir_arr,
                    np.asarray(p, dtype=np.float64),
                    np.asarray(d, dtype=np.float64),
                )
                if iou is not None:
                    metrics[key]["iou"] = float(iou)
                    metrics[key]["iou_error"] = float(1.0 - iou)

    return metrics


# ── aggregate stats ──────────────────────────────────────────────────────────

def stats(vals: List) -> Dict:
    arr = np.array([v for v in vals if v is not None], dtype=np.float64)
    n = len(arr)
    if n == 0:
        return dict(valid=0, mean=np.nan, median=np.nan, std=np.nan,
                    min=np.nan, max=np.nan, p25=np.nan, p75=np.nan)
    return dict(valid=n, mean=float(np.mean(arr)), median=float(np.median(arr)),
                std=float(np.std(arr)), min=float(np.min(arr)), max=float(np.max(arr)),
                p25=float(np.percentile(arr, 25)), p75=float(np.percentile(arr, 75)))


def thresh(vals: List, t: float, above: bool = False) -> Tuple[int, float]:
    v = [x for x in vals if x is not None]
    if not v:
        return 0, float("nan")
    n = sum(1 for x in v if (x >= t if above else x <= t))
    return n, 100.0 * n / len(v)


def build_row(backend: str, which: str, scene_data: List[Dict], total: int) -> Dict:
    ps = stats([d["pos"] for d in scene_data])
    rs = stats([d["rot"] for d in scene_data])
    is_ = stats([d["iou"] for d in scene_data])
    ie = stats([d["iou_error"] for d in scene_data])
    p05, p05p = thresh([d["pos"] for d in scene_data], 0.5)
    p10, p10p = thresh([d["pos"] for d in scene_data], 1.0)
    r15, r15p = thresh([d["rot"] for d in scene_data], 15.0)
    r30, r30p = thresh([d["rot"] for d in scene_data], 30.0)
    i25, i25p = thresh([d["iou"] for d in scene_data], 0.25, above=True)
    i50, i50p = thresh([d["iou"] for d in scene_data], 0.50, above=True)
    return dict(
        backend=backend, which=which, rows=len(scene_data), scenes=total,
        pos_err_m_valid=ps["valid"], pos_err_m_mean=ps["mean"],
        pos_err_m_median=ps["median"], pos_err_m_std=ps["std"],
        pos_err_m_min=ps["min"], pos_err_m_max=ps["max"],
        pos_err_m_p25=ps["p25"], pos_err_m_p75=ps["p75"],
        rot_err_deg_valid=rs["valid"], rot_err_deg_mean=rs["mean"],
        rot_err_deg_median=rs["median"], rot_err_deg_std=rs["std"],
        rot_err_deg_min=rs["min"], rot_err_deg_max=rs["max"],
        rot_err_deg_p25=rs["p25"], rot_err_deg_p75=rs["p75"],
        iou_valid=is_["valid"], iou_mean=is_["mean"],
        iou_median=is_["median"], iou_std=is_["std"],
        iou_min=is_["min"], iou_max=is_["max"],
        iou_p25=is_["p25"], iou_p75=is_["p75"],
        iou_error_valid=ie["valid"], iou_error_mean=ie["mean"],
        iou_error_median=ie["median"], iou_error_std=ie["std"],
        iou_error_min=ie["min"], iou_error_max=ie["max"],
        iou_error_p25=ie["p25"], iou_error_p75=ie["p75"],
        **{
            "pos_le_0.5m": p05, "pos_le_1.0m": p10,
            "rot_le_15deg": r15, "rot_le_30deg": r30,
            "iou_ge_0.25": i25, "iou_ge_0.50": i50,
            "pos_le_0.5m_pct": p05p, "pos_le_1.0m_pct": p10p,
            "rot_le_15deg_pct": r15p, "rot_le_30deg_pct": r30p,
            "iou_ge_0.25_pct": i25p, "iou_ge_0.50_pct": i50p,
        },
    )


COLUMNS = [
    "backend", "which", "rows", "scenes",
    "pos_err_m_valid", "pos_err_m_mean", "pos_err_m_median", "pos_err_m_std",
    "pos_err_m_min", "pos_err_m_max", "pos_err_m_p25", "pos_err_m_p75",
    "rot_err_deg_valid", "rot_err_deg_mean", "rot_err_deg_median", "rot_err_deg_std",
    "rot_err_deg_min", "rot_err_deg_max", "rot_err_deg_p25", "rot_err_deg_p75",
    "iou_valid", "iou_mean", "iou_median", "iou_std", "iou_min", "iou_max", "iou_p25", "iou_p75",
    "iou_error_valid", "iou_error_mean", "iou_error_median", "iou_error_std",
    "iou_error_min", "iou_error_max", "iou_error_p25", "iou_error_p75",
    "pos_le_0.5m", "pos_le_1.0m", "rot_le_15deg", "rot_le_30deg", "iou_ge_0.25", "iou_ge_0.50",
    "pos_le_0.5m_pct", "pos_le_1.0m_pct", "rot_le_15deg_pct", "rot_le_30deg_pct",
    "iou_ge_0.25_pct", "iou_ge_0.50_pct",
]


# ── main ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Replay cached LLM answers and compute View IoU metrics."
    )
    ap.add_argument("--qwen_json", type=Path, required=True,
                    help="Path to saved LLM results JSON (from llm_answerer).")
    ap.add_argument("--candidates", type=Path, required=True,
                    help="Path to evaluation candidates JSON.")
    ap.add_argument("--dataset_root", type=Path, required=True,
                    help="Root directory containing scene scan folders.")
    ap.add_argument("--dialogue_script", type=Path, required=True,
                    help="Path to the dialogue entry-point script.")
    ap.add_argument("--output_csv", type=Path, default=Path("replay_metrics.csv"),
                    help="Output CSV path for aggregated metrics.")
    ap.add_argument("--no_iou", action="store_true",
                    help="Skip IoU computation (faster, no mesh loading).")
    ap.add_argument("--h_fov_deg", type=float, default=39.31,
                    help="Horizontal FOV in degrees for IoU.")
    ap.add_argument("--v_fov_deg", type=float, default=64.76,
                    help="Vertical FOV in degrees for IoU.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    orig_print = builtins.print

    # Load inputs
    qwen_results = json.loads(Path(args.qwen_json).read_text())
    cands_raw = json.loads(Path(args.candidates).read_text())
    dataset_root = Path(args.dataset_root)
    candidates_p = Path(args.candidates)

    # Build scene -> gt_pose lookup
    gt_lookup: Dict[str, Dict] = {}
    for e in cands_raw.get("scenes", []):
        if isinstance(e, dict) and "scene_id" in e:
            gt_lookup[e["scene_id"]] = e.get("gt_pose", {})

    # Import dialogue module
    dlg = import_module(Path(args.dialogue_script), "dlg_replay")

    # Set up IoU computer
    iou_comp: Optional[IoUComputer] = None
    if not args.no_iou:
        try:
            # Patch discover_mesh to support ScanNet mesh naming
            import langloc.eval.view_iou as _viou
            _viou.discover_mesh = discover_mesh_extended
            iou_comp = IoUComputer(
                hfov_deg=args.h_fov_deg, vfov_deg=args.v_fov_deg,
            )
            orig_print("IoU computation enabled (ScanNet mesh fallback active).")
        except Exception as e:
            orig_print(f"IoU disabled ({e})")

    # Patch predict_pose ONCE before the loop
    install_pose_hooks(dlg)

    # Aggregate collectors
    pool: Dict[Tuple, List] = {}
    ok = err = 0

    for i, r in enumerate(qwen_results, 1):
        sid = r["scene_id"]
        qa_log = r.get("qa_log", [])
        returncode = r.get("returncode", 1)

        orig_print(f"[{i:3d}/{len(qwen_results)}] {sid} … ", end="", flush=True)

        if returncode != 0:
            orig_print("SKIP (original run failed)")
            err += 1
            continue

        gt_pose = gt_lookup.get(sid, {})
        gt_pos_raw = gt_pose.get("position")
        gt_dir_raw = gt_pose.get("direction")
        if gt_pos_raw is None or gt_dir_raw is None:
            orig_print("SKIP (no GT pose)")
            err += 1
            continue

        gt_pos = np.asarray(gt_pos_raw, dtype=np.float64)
        gt_dir = np.asarray(gt_dir_raw, dtype=np.float64)
        scene_dir = dataset_root / sid

        metrics = replay_scene(
            sid=sid,
            qa_log=qa_log,
            gt_pos=gt_pos,
            gt_dir=gt_dir,
            scene_dir=scene_dir,
            dlg=dlg,
            iou_comp=iou_comp,
            dataset_root=dataset_root,
            candidates_json=candidates_p,
            orig_print=orig_print,
        )

        for key, vals in metrics.items():
            pool.setdefault(key, []).append(vals)

        n_iou = sum(1 for k in metrics if metrics[k]["iou"] is not None)
        orig_print(f"OK | iou_computed={n_iou}/{len(metrics)}")
        ok += 1

    orig_print(f"\nDone: {ok} OK / {err} skipped")

    # Build CSV rows
    row_order = [
        ("Baseline", "predicted_pose"),
        ("A1", "MAP"), ("A1", "MEAN"),
        ("A2", "MAP"), ("A2", "MEAN"),
        ("A3", "MAP"), ("A3", "MEAN"),
    ]
    rows = []
    for backend, which in row_order:
        data = pool.get((backend, which), [])
        if not data:
            orig_print(f"  WARNING: no data for {backend} {which}")
            continue
        rows.append(build_row(backend, which, data, ok))

    out = Path(args.output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)

    orig_print(f"\nSaved -> {out}")
    orig_print(f"\n{'backend':<12} {'which':<18} {'scenes':>6} {'pos_mean':>9} {'pos_med':>9} "
               f"{'rot_mean':>9} {'iou_mean':>9} {'<=0.5m%':>8} {'<=1m%':>7} "
               f"{'<=15d%':>7} {'<=30d%':>7} {'iou>=.25%':>10}")
    orig_print("-" * 115)
    for row in rows:
        orig_print(f"{row['backend']:<12} {row['which']:<18} {row['scenes']:>6} "
                   f"{row['pos_err_m_mean']:>9.3f} {row['pos_err_m_median']:>9.3f} "
                   f"{row['rot_err_deg_mean']:>9.2f} {row['iou_mean']:>9.4f} "
                   f"{row['pos_le_0.5m_pct']:>8.1f} {row['pos_le_1.0m_pct']:>7.1f} "
                   f"{row['rot_le_15deg_pct']:>7.1f} {row['rot_le_30deg_pct']:>7.1f} "
                   f"{row['iou_ge_0.25_pct']:>10.1f}")


if __name__ == "__main__":
    main()
