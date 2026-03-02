"""Visualizes matched objects for a single 3RScan scene.

Colours the 3D objects that BigGNN (or the DBSCAN overlap matcher) links to
each ScanScribe / Human caption for ONE scene.

Controls:
    SPACE / ENTER - next caption
    q / ESC       - quit
"""

import sys
from pathlib import Path

import torch
import numpy as np
import open3d as o3d

import hydra
from omegaconf import DictConfig

from src.data_processing.scene_graph import SceneGraph
from src.analysis.helper import get_matching_subgraph
from src.localization.grid import load_scene
from src.localization.visualization import colour_objects
try:
    from src.models.model_graph2graph import BigGNN
except ImportError:
    BigGNN = None


def to_legacy_cpu(mesh_t):
    """Convert any Tensor/CUDA mesh to legacy CPU TriangleMesh."""
    if isinstance(mesh_t, o3d.cuda.pybind.t.geometry.TriangleMesh) \
       or isinstance(mesh_t, o3d.t.geometry.TriangleMesh):
        return mesh_t.to_legacy()        # Open3D >=0.15
    return mesh_t                        # already legacy


@torch.inference_mode()
def matched_object_ids(model, qg: SceneGraph, sg: SceneGraph,
                       dbscan_eps: float = 0.5, dbscan_min_samples: int = 1):
    """Returns object IDs matched by BigGNN or the pure-overlap fallback.

    Args:
        model: Trained BigGNN model, or None for overlap-only matching.
        qg: Query (text) scene graph.
        sg: Database (3DSSG) scene graph.
        dbscan_eps: DBSCAN epsilon parameter for cosine distance.
        dbscan_min_samples: DBSCAN minimum samples per cluster.

    Returns:
        List of matched object IDs from the scene graph.
    """
    if model is None:
        print("No GNN model; using pure-overlap matcher")
        _, sub3d = get_matching_subgraph(qg, sg, dbscan_eps, dbscan_min_samples)
        return [] if sub3d is None else list(sub3d.nodes)
    device = next(model.parameters()).device
    def prep(g):
        n, e, f = g.to_pyg()
        return (
          torch.tensor(np.array(n), dtype=torch.float32, device=device),
          torch.tensor(np.array(e[0:2]), dtype=torch.int64, device=device),
          torch.tensor(np.array(f), dtype=torch.float32, device=device),
        )
    x_n, x_e, x_f = prep(qg)
    p_n, p_e, p_f = prep(sg)
    _ = model(x_n, p_n, x_e, p_e, x_f, p_f)
    _, sub3d = get_matching_subgraph(qg, sg, dbscan_eps, dbscan_min_samples)
    return [] if sub3d is None else list(sub3d.nodes)


def run_visualization_minimal(cfg: DictConfig) -> None:
    """Runs the interactive single-scene matched-object viewer.

    Args:
        cfg: Merged Hydra configuration.
    """
    device = cfg.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if cfg.scan_id is None:
        raise ValueError("scan_id is required. Set via CLI: scan_id=XXXX")
    if cfg.paths.rscan_root is None:
        raise ValueError("paths.rscan_root is required. Place 3RScan data in ./data/3rscan or override paths.rscan_root=...")

    scan_id = cfg.scan_id
    scan_dir = Path(cfg.paths.rscan_root) / scan_id

    # load 3D scene-graph
    scenes = torch.load(cfg.paths.graphs_3dssg, map_location=device, weights_only=False)
    if scan_id not in scenes:
        raise ValueError(f"{scan_id} not found in 3dssg graphs")
    sg = SceneGraph(scan_id,
                    graph_type="3dssg",
                    graph=scenes[scan_id],
                    max_dist=cfg.graph.max_dist,
                    embedding_type=cfg.graph.embedding_type,
                    use_attributes=cfg.graph.use_attributes)

    # load captions
    caps_raw = torch.load(cfg.paths.scanscribe_text, map_location=device, weights_only=False)
    captions = {
      k: SceneGraph(scan_id,
                    graph_type="scanscribe",
                    graph=g,
                    embedding_type=cfg.graph.embedding_type,
                    use_attributes=cfg.graph.use_attributes)
      for k, g in caps_raw.items() if k.startswith(scan_id)
    }
    if not captions:
        raise RuntimeError(f"No captions for {scan_id}")

    # optional BigGNN
    model = None
    ckpt_dir = Path(cfg.paths.checkpoint_dir)
    if cfg.eval.model_name is not None:
        if BigGNN is None:
            raise ImportError("BigGNN not available")
        ckpt_path = ckpt_dir / f"{cfg.eval.model_name}.pt"
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model = BigGNN(cfg.model.N, cfg.model.heads, cfg.model.embed_dim, cfg.model.dropout).to(device)
        model.load_state_dict(ckpt)
        model.eval()

    # load mesh + obj2faces
    mesh, _, obj2faces = load_scene(scan_dir)

    for cap_key, qg in captions.items():
        matched = matched_object_ids(model, qg, sg,
                                     cfg.graph.dbscan_eps,
                                     cfg.graph.dbscan_min_samples)
        print(f"\n─────── {cap_key} ───────")
        print("Matched object IDs:", matched)
        print("SPACE/ENTER → next    |    q/Esc → quit")

        # fresh CPU copy via constructor
        cpu_mesh = to_legacy_cpu(mesh)          # guarantees legacy-CPU
        vis_mesh = colour_objects(cpu_mesh, obj2faces, matched, base=(0.6, 0.6, 0.6))

        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window(window_name=cap_key, width=1280, height=720)
        vis.add_geometry(vis_mesh)
        vis.register_key_callback(32,  lambda v: v.close())  # SPACE
        vis.register_key_callback(257, lambda v: v.close())  # ENTER
        vis.register_key_callback(81,  lambda v: sys.exit(0))  # q
        vis.register_key_callback(113, lambda v: sys.exit(0))  # Q
        vis.register_key_callback(256, lambda v: sys.exit(0))  # ESC
        vis.run()
        vis.destroy_window()


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Hydra CLI entry point for single-scene visualization."""
    run_visualization_minimal(cfg)


if __name__ == "__main__":
    main()
