#!/usr/bin/env python3
"""Generate supp_direction_field.png: predicted viewing directions on top-down view.

Runs the fine-localization pipeline at two grid resolutions and renders
side-by-side direction fields with metric annotations.

Usage::

    python -m tools.viz.fig_supp_direction_field \
        --dataset scannet --root ./data/scans \
        --scan-id scene0002_00 \
        --frame-id 003693 \
        --graphs-3dssg ./data/processed_data/generated/scannet_scene_graphs.pt \
        --output docs/figures
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _compute_metrics(cams, probs, cam_dirs, pred_pos, pred_dir, gt_pos, gt_dir, grid_step):
    """Compute localization metrics for annotation."""
    pos_err = float(np.linalg.norm(pred_pos[:2] - gt_pos[:2])) if pred_pos is not None else float("nan")
    ang_err = float("nan")
    if pred_dir is not None and gt_dir is not None:
        cos_val = np.clip(np.dot(pred_dir, gt_dir), -1.0, 1.0)
        ang_err = math.degrees(math.acos(cos_val))
    n_cells = int((probs > 0).sum())
    n_dirs = int((np.linalg.norm(cam_dirs, axis=1) > 1e-6).sum())
    return {
        "grid_step": grid_step,
        "n_cells": n_cells,
        "n_dirs": n_dirs,
        "pos_err_m": pos_err,
        "ang_err_deg": ang_err,
    }


def _annotate_metrics(img, metrics, dpi=300, save_pdf_path=None):
    """Burn metric text into the bottom-left corner of an image.

    If ``save_pdf_path`` is given, the annotated figure is additionally saved
    as a PDF with the metric text preserved as vector (the background image
    remains an embedded raster).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    H, W = img.shape[:2]
    fig, ax = plt.subplots(1, 1, figsize=(W / dpi, H / dpi), dpi=dpi)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.imshow(img)
    ax.set_axis_off()
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)

    lines = [
        f"Grid: {metrics['grid_step']:.2f} m",
        f"Cells: {metrics['n_cells']}",
        f"Pos err: {metrics['pos_err_m']:.2f} m",
        f"Ang err: {metrics['ang_err_deg']:.1f}°",
    ]
    text = "\n".join(lines)
    fs = max(7, min(W, H) / dpi * 1.2)
    ax.text(W * 0.02, H * 0.98, text, fontsize=fs, fontfamily="monospace",
            verticalalignment="bottom", horizontalalignment="left",
            color="white",
            bbox=dict(boxstyle="round,pad=0.3", fc="black", alpha=0.7, ec="none"),
            zorder=10)

    if save_pdf_path is not None:
        fig.savefig(save_pdf_path, format="pdf", dpi=dpi,
                    bbox_inches="tight", pad_inches=0)

    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    from PIL import Image as PILImage
    if buf.shape[:2] != (H, W):
        buf = np.asarray(PILImage.fromarray(buf).resize((W, H), PILImage.LANCZOS))
    return buf


def parse_args():
    ap = argparse.ArgumentParser(description="Direction field figure for fine localization.")
    ap.add_argument("--dataset", choices=["3rscan", "scannet"], required=True)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--scan-id", "--scan_id", dest="scan_id", required=True)
    ap.add_argument("--frame-id", "--frame_id", dest="frame_id", type=str, default=None)
    ap.add_argument("--graphs-3dssg", "--graphs_3dssg", dest="graphs_3dssg", required=True)
    ap.add_argument("--grid-step", "--grid_step", dest="grid_step",
                    type=float, default=0.25,
                    help="Coarse grid spacing in metres (default: 0.25).")
    ap.add_argument("--fine-grid-step", "--fine_grid_step", dest="fine_grid_step",
                    type=float, default=0.10,
                    help="Fine grid spacing in metres for comparison (default: 0.10).")
    ap.add_argument("--topdown-size", "--topdown_size", dest="topdown_size",
                    type=int, default=2048)
    ap.add_argument("--output", type=Path, default=Path("docs/figures"))
    ap.add_argument("--vector", action="store_true",
                    help="Save outputs as vector PDF instead of raster PNG.")
    return ap.parse_args()


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    from tools.viz.visualize_teaser import (
        load_scene_any, detect_up_axis, render_topdown,
        run_localization, overlay_direction_field,
        _load_frame_description,
    )
    from langloc.localization.frame_io import frame_to_scenegraph, camera_center_from_pose
    from PIL import Image

    scan_dir = args.root / args.scan_id

    print(f"[1/5] Loading scene: {args.scan_id}")
    mesh, tri2obj, obj2faces = load_scene_any(scan_dir, dataset=args.dataset)
    up_axis = detect_up_axis(mesh)

    print(f"[2/5] Loading frame description...")
    desc_dir = scan_dir / "output" / "descriptions"
    frame_id = Path(args.frame_id).stem if args.frame_id else args.frame_id
    frame_data = _load_frame_description(desc_dir, frame_id)
    query_text = frame_data.get("description", "")
    print(f"  Query: \"{query_text}\"")

    query_sg, _ = frame_to_scenegraph(frame_data, embedding_type="word2vec",
                                       use_attributes=True)
    scene_pose = frame_data.get("scene_pose")
    gt_pos = camera_center_from_pose(scene_pose)
    pose_mat = np.array(scene_pose, dtype=np.float64)
    gt_dir = pose_mat[:3, 2]
    gt_dir /= max(float(np.linalg.norm(gt_dir)), 1e-6)

    print(f"[3/5] Running localization (grid={args.grid_step}m)...")
    cams, probs, obj_ids, pred_pos, pred_dir, cam_dirs = run_localization(
        mesh, tri2obj, obj2faces,
        query_sg=query_sg, graphs_3dssg=args.graphs_3dssg,
        scan_id=args.scan_id, up_axis=up_axis,
        grid_step=args.grid_step,
    )
    metrics_coarse = _compute_metrics(
        cams, probs, cam_dirs, pred_pos, pred_dir, gt_pos, gt_dir, args.grid_step)

    print(f"[4/5] Running localization (grid={args.fine_grid_step}m)...")
    cams_f, probs_f, _, pred_pos_f, pred_dir_f, cam_dirs_f = run_localization(
        mesh, tri2obj, obj2faces,
        query_sg=query_sg, graphs_3dssg=args.graphs_3dssg,
        scan_id=args.scan_id, up_axis=up_axis,
        grid_step=args.fine_grid_step,
    )
    metrics_fine = _compute_metrics(
        cams_f, probs_f, cam_dirs_f, pred_pos_f, pred_dir_f, gt_pos, gt_dir,
        args.fine_grid_step)

    print(f"[5/5] Rendering direction fields...")
    topdown_img, intrinsic, extrinsic = render_topdown(
        mesh, up_axis, args.topdown_size)

    ext = "pdf" if args.vector else "png"

    # Coarse grid
    img_coarse = overlay_direction_field(
        topdown_img, intrinsic, extrinsic,
        cams, probs, cam_dirs,
        pred_pos=pred_pos, pred_dir=pred_dir,
        gt_pos=gt_pos, gt_dir=gt_dir,
        up_axis=up_axis, stride=1, dpi=300,
    )
    out_coarse = args.output / f"supp_direction_field.{ext}"
    img_coarse = _annotate_metrics(
        img_coarse, metrics_coarse,
        save_pdf_path=out_coarse if args.vector else None)

    # Fine grid
    img_fine = overlay_direction_field(
        topdown_img, intrinsic, extrinsic,
        cams_f, probs_f, cam_dirs_f,
        pred_pos=pred_pos_f, pred_dir=pred_dir_f,
        gt_pos=gt_pos, gt_dir=gt_dir,
        up_axis=up_axis, stride=1, dpi=300,
    )
    out_fine = args.output / f"supp_direction_field_fine.{ext}"
    img_fine = _annotate_metrics(
        img_fine, metrics_fine,
        save_pdf_path=out_fine if args.vector else None)

    if not args.vector:
        Image.fromarray(img_coarse).save(out_coarse, dpi=(300, 300))
        Image.fromarray(img_fine).save(out_fine, dpi=(300, 300))
    print(f"  Saved: {out_coarse}")
    print(f"  Saved: {out_fine}")

    # Side-by-side comparison
    out_combined = args.output / f"supp_direction_field_comparison.{ext}"
    if args.vector:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(16, 8), dpi=300)
        for ax, img in [(axes[0], img_coarse), (axes[1], img_fine)]:
            ax.imshow(img)
            ax.set_axis_off()
        fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01,
                            wspace=0.03)
        fig.savefig(out_combined, format="pdf", dpi=300,
                    bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
    else:
        h = min(img_coarse.shape[0], img_fine.shape[0])
        w_c, w_f = img_coarse.shape[1], img_fine.shape[1]
        combined = np.ones((h, w_c + w_f + 10, 3), dtype=np.uint8) * 255
        combined[:h, :w_c] = img_coarse[:h]
        combined[:h, w_c + 10:] = img_fine[:h]
        Image.fromarray(combined).save(out_combined, dpi=(300, 300))
    print(f"  Saved: {out_combined}")

    print(f"\n  Coarse ({args.grid_step}m): {metrics_coarse['n_cells']} cells, "
          f"pos_err={metrics_coarse['pos_err_m']:.2f}m, "
          f"ang_err={metrics_coarse['ang_err_deg']:.1f}°")
    print(f"  Fine   ({args.fine_grid_step}m): {metrics_fine['n_cells']} cells, "
          f"pos_err={metrics_fine['pos_err_m']:.2f}m, "
          f"ang_err={metrics_fine['ang_err_deg']:.1f}°")
    print("Done.")


if __name__ == "__main__":
    main()
