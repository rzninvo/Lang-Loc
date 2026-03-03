"""Localise a natural-language query inside a specific 3RScan scene.

Bridges:
  - single_inference.py : builds a text SceneGraph from free text (LLM-backed)
  - localization pipeline : object matching, dense grid casting, visualisation

Requires OPENAI_API_KEY in .env or environment.

Usage::

    python -m src.visualization.visualize_loc_from_query \\
        scan_id=3RScan1234 \\
        inference.query="I can see a sofa facing a TV and a coffee table between them." \\
        localization.top_k=8 localization.show_heatmap=true localization.show_3d=true
"""
from __future__ import annotations

import os
from pathlib import Path

import torch

import hydra
from omegaconf import DictConfig

from langloc.graphs.scene_graph import SceneGraph
from langloc.localization.grid import load_scene
from langloc.localization.matching import topk_matched_objects
from langloc.localization.pipeline import run_loc_pipeline
from langloc.graph_matching.single_inference import text_to_scenegraph


def load_scene_graph_for_scan(graphs_3dssg_path: str, scan_id: str,
                               max_dist: float, embedding_type: str,
                               use_attributes: bool) -> SceneGraph:
    """Load 3DSSG database and return SceneGraph for the requested scan_id.

    Args:
        graphs_3dssg_path: Path to the 3DSSG graphs .pt file.
        scan_id: Target scan ID.
        max_dist: Maximum distance for graph construction.
        embedding_type: Embedding backend name.
        use_attributes: Whether to use attribute embeddings.

    Returns:
        SceneGraph for the requested scan.

    Raises:
        FileNotFoundError: If the graphs file does not exist.
        KeyError: If the scan_id is not found.
    """
    g3d_path = Path(graphs_3dssg_path)
    if not g3d_path.exists():
        raise FileNotFoundError(g3d_path)

    g3d_all = torch.load(g3d_path, map_location="cpu", weights_only=False)
    if scan_id not in g3d_all:
        alts = [scan_id.replace("/", ""), scan_id.replace("3RScan/", ""), scan_id.split("/")[-1]]
        hit = next((a for a in alts if a in g3d_all), None)
        if hit is None:
            raise KeyError(f"scan_id '{scan_id}' not found in 3DSSG file.")
        scan_id = hit

    g = g3d_all[scan_id]
    sg = SceneGraph(scan_id,
                    graph_type="3dssg",
                    graph=g,
                    max_dist=max_dist,
                    embedding_type=embedding_type,
                    use_attributes=use_attributes)
    return sg


def run_visualize_loc_from_query(cfg: DictConfig) -> None:
    """Runs localization visualization from a natural-language query.

    Args:
        cfg: Merged Hydra configuration.
    """
    if cfg.scan_id is None:
        raise ValueError("scan_id is required. Set via CLI: scan_id=3RScan1234")
    if cfg.inference.query is None:
        raise ValueError("inference.query is required. Set via CLI: inference.query='...'")
    if cfg.paths.rscan_root is None:
        raise ValueError("paths.rscan_root is required. Place 3RScan data in ./data/3rscan or override paths.rscan_root=...")
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY not set. Add it to your .env file.")

    loc = cfg.localization

    # 1) Build a query SceneGraph from free text
    qg = text_to_scenegraph(cfg.inference.query,
                            embedding_type=cfg.graph.embedding_type,
                            use_attributes=cfg.graph.use_attributes,
                            scene_id="query_0001",
                            debug=cfg.inference.debug)

    # 2) Load the target scene's 3DSSG
    sg = load_scene_graph_for_scan(cfg.paths.graphs_3dssg,
                                    cfg.scan_id,
                                    max_dist=cfg.graph.max_dist,
                                    embedding_type=cfg.graph.embedding_type,
                                    use_attributes=cfg.graph.use_attributes)

    # 3) Top-K object matches
    obj_ids = topk_matched_objects(qg, sg, k=loc.top_k)
    if not obj_ids:
        print("No cosine matches found between query and scene.")
        return

    # 4) Load mesh and run localization pipeline
    rscan_root = Path(cfg.paths.rscan_root)
    mesh, tri2obj, obj2faces = load_scene(rscan_root / sg.scene_id)
    print(f"[{sg.scene_id}] {len(obj_ids)} matched objs")

    run_loc_pipeline(
        scan_dir=rscan_root / sg.scene_id,
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
        title_prefix=f"{sg.scene_id} – ",
    )


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Hydra CLI entry point for query-based localization visualization."""
    run_visualize_loc_from_query(cfg)


if __name__ == "__main__":
    main()
