"""Dense-grid localization probability visualization.

For every ScanScribe caption graph:

1. Load its ground-truth 3D-SSG scene-graph + mesh.
2. Compute cosine similarity between all nodes; keep Top-K object matches.
3. Run the shared localization pipeline (grid sampling, ray-casting,
   probability computation, rendering).

This module is a thin wrapper that re-exports core utilities from
:mod:`langloc.localization` and provides its own ``main()`` for standalone
demo use.

Usage::

    python tools/viz/visualize_loc_prob.py \\
        localization.show_heatmap=true localization.show_3d=true
"""
from __future__ import annotations

from pathlib import Path

import torch

import hydra
from omegaconf import DictConfig

from langloc.graphs.scene_graph import SceneGraph

# Re-export all public helpers from langloc.localization so that
# existing code importing from this module keeps working.
from langloc.localization.grid import load_scene, sample_grid, first_hit_is_object  # noqa: F401
from langloc.localization.matching import topk_matched_objects  # noqa: F401
from langloc.localization.pipeline import run_loc_pipeline
from langloc.localization.visualization import (  # noqa: F401
    colour_objects,
    colormap,
    dir_to_yaw_pitch,
    best_fov_window,
    average_direction,
)


def run_visualize_loc_prob(cfg: DictConfig) -> None:
    """Standalone dense-grid localization demo over ScanScribe captions.

    Args:
        cfg: Merged Hydra configuration.
    """
    if cfg.paths.rscan_root is None:
        raise ValueError("paths.rscan_root is required. Place 3RScan data in ./data/3rscan or override paths.rscan_root=...")

    rscan_root = Path(cfg.paths.rscan_root)
    loc = cfg.localization

    g3d = torch.load(cfg.paths.graphs_3dssg, map_location="cpu", weights_only=False)
    scenes = {sid: SceneGraph(sid,
                              graph_type="3dssg",
                              graph=g,
                              max_dist=cfg.graph.max_dist,
                              embedding_type=cfg.graph.embedding_type,
                              use_attributes=cfg.graph.use_attributes)
              for sid, g in g3d.items()}

    gtxt = torch.load(cfg.paths.scanscribe_text, map_location="cpu", weights_only=False)
    queries = [SceneGraph(k.split("_")[0],
                          txt_id=None,
                          graph=g,
                          graph_type="scanscribe",
                          embedding_type=cfg.graph.embedding_type,
                          use_attributes=cfg.graph.use_attributes)
               for k, g in gtxt.items()]
    if loc.max_scenes:
        queries = queries[:loc.max_scenes]

    for qi, qg in enumerate(queries, 1):
        sid = qg.scene_id
        sg = scenes[sid]

        obj_ids = topk_matched_objects(qg, sg, k=loc.top_k)
        if not obj_ids:
            print(f"[{qi}] {sid} : no cosine matches — skipped")
            continue

        mesh, tri2obj, obj2faces = load_scene(rscan_root / sid)
        print(f"[{qi}] {sid}: {len(obj_ids)} matched objs")

        run_loc_pipeline(
            scan_dir=rscan_root / sid,
            obj_ids=obj_ids,
            obj2faces=obj2faces,
            mesh=mesh,
            tri2obj=tri2obj,
            grid_step=loc.grid_step,
            show_heatmap=loc.show_heatmap,
            show_arrows=loc.show_arrows,
            show_3d=loc.show_3d,
            h_fov_deg=loc.h_fov_deg,
            v_fov_deg=loc.v_fov_deg,
            arrow_stride=loc.arrow_stride,
            arrow_len=loc.arrow_len,
            title_prefix=f"{sid}  –  ",
        )


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Hydra CLI entry point for localization probability visualization."""
    run_visualize_loc_prob(cfg)


if __name__ == "__main__":
    main()
