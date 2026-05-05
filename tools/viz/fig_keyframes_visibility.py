#!/usr/bin/env python3
"""Per-keyframe visibility figures (one per Stage-2 DPP keyframe).

For each of the (up to) 10 keyframes selected by the two-stage DPP for a given
scene, render two artifacts into ``{output}/{scan_id}/``:

* ``{rank}_{fid}_rgb.jpg`` — copy of the source RGB frame.
* ``{rank}_{fid}_topdown.pdf`` — top-down render of the mesh where only the
  faces actually visible in this keyframe are coloured by their semantic
  instance; everything else is desaturated. The camera frustum is projected
  onto the floor plane as a translucent vector overlay.

Usage::

    python -m tools.viz.fig_keyframes_visibility \\
        --dataset scannet --root ./data/scans \\
        --scan-id scene0002_00 \\
        --output docs/figures/keyframes
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def parse_args():
    ap = argparse.ArgumentParser(
        description="Per-keyframe visibility renders for the dataset slide.")
    ap.add_argument("--dataset", choices=["3rscan", "scannet"], required=True)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--scan-id", "--scan_id", dest="scan_id", required=True)
    ap.add_argument("--output", type=Path,
                    default=Path("docs/figures/keyframes"))
    ap.add_argument("--topdown-size", "--topdown_size",
                    dest="topdown_size", type=int, default=2048)
    ap.add_argument("--desat", type=float, default=0.85,
                    help="Grey-out strength for non-visible mesh regions.")
    ap.add_argument("--fov-h", "--fov_h", dest="fov_h",
                    type=float, default=58.0)
    ap.add_argument("--fov-v", "--fov_v", dest="fov_v",
                    type=float, default=45.0)
    ap.add_argument("--frustum-reach", "--frustum_reach",
                    dest="frustum_reach", type=float, default=2.2)
    ap.add_argument("--frustum-alpha", "--frustum_alpha",
                    dest="frustum_alpha", type=float, default=0.22)
    return ap.parse_args()


def main():
    args = parse_args()

    out_dir = args.output / args.scan_id
    out_dir.mkdir(parents=True, exist_ok=True)

    from tools.viz.visualize_teaser import (
        load_scene_any, detect_up_axis, render_topdown,
        load_semantic_vertex_colors,
    )
    from tools.viz.fig_supp_visibility import (
        color_visible_faces, overlay_frustum_pdf,
    )

    scan_dir = args.root / args.scan_id
    print(f"[1/4] Loading scene mesh: {args.scan_id}")
    mesh, tri2obj, obj2faces = load_scene_any(scan_dir, dataset=args.dataset)
    up_axis = detect_up_axis(mesh)

    # Visibility cache (per-frame visible face IDs and visible_objects)
    cache_path = scan_dir / "output" / "cache" / f"{args.scan_id}.json"
    cache = json.loads(cache_path.read_text())
    cache_by_fid = {entry["fid"]: entry for entry in cache}

    # DPP diagnostics → list of selected keyframes
    diag_path = scan_dir / "output" / "cache" / f"{args.scan_id}_dpp_diag.json"
    diag = json.loads(diag_path.read_text())
    stage2_fids = diag["stage2_fids"]
    print(f"[2/4] {len(stage2_fids)} Stage-2 keyframes: {stage2_fids}")

    sem_colors = load_semantic_vertex_colors(scan_dir, args.scan_id, args.dataset)
    if sem_colors is None:
        raise FileNotFoundError(
            "Semantic vertex colours not found — needed for instance colouring.")

    print(f"[3/4] Rendering {len(stage2_fids)} keyframes into {out_dir}")
    for k_idx, fid in enumerate(stage2_fids, 1):
        print(f"  [{k_idx:02d}/{len(stage2_fids)}] keyframe fid={fid}")

        # -- Copy the RGB frame ------------------------------------------
        src_rgb = scan_dir / "color" / f"{fid}.jpg"
        if not src_rgb.exists():
            src_rgb = scan_dir / "color" / f"{fid}.png"
        if src_rgb.exists():
            dst_rgb = out_dir / f"{k_idx:02d}_{fid}_rgb{src_rgb.suffix}"
            shutil.copy2(src_rgb, dst_rgb)
        else:
            print(f"      [warn] RGB frame not found for {fid}")

        # -- Per-frame visibility + camera pose --------------------------
        entry = cache_by_fid.get(fid)
        if entry is None:
            print(f"      [warn] no visibility cache entry for {fid}; skipping")
            continue
        visible_face_ids = [int(x) for x in entry.get("visible_face_ids", [])]

        desc_path = scan_dir / "output" / "descriptions" / f"{fid}.json"
        if not desc_path.exists():
            print(f"      [warn] no description JSON for {fid}; skipping")
            continue
        desc = json.loads(desc_path.read_text())
        pose = np.array(desc["scene_pose"], dtype=np.float64)

        # Build the face-level coloured mesh: only rasterised faces get
        # instance colours; everything else is desaturated.
        coloured_mesh = color_visible_faces(
            mesh, tri2obj, visible_face_ids,
            sem_colors, desat=args.desat,
        )

        # Top-down render of the coloured mesh (also returns the camera
        # intrinsic / extrinsic that we need to project the 3D frustum).
        topdown_img, intrinsic, extrinsic = render_topdown(
            coloured_mesh, up_axis, args.topdown_size,
        )

        # Translucent frustum overlay (vector polygon + edges + apex marker).
        out_path = out_dir / f"{k_idx:02d}_{fid}_topdown.pdf"
        overlay_frustum_pdf(
            topdown_img, pose, intrinsic, extrinsic,
            fov_h_deg=args.fov_h, fov_v_deg=args.fov_v,
            reach=args.frustum_reach,
            output_path=out_path,
            fill_alpha=args.frustum_alpha,
        )
        print(f"      saved: {out_path.name}")

    print(f"[4/4] Done. {len(stage2_fids)} keyframes in {out_dir}")


if __name__ == "__main__":
    main()
