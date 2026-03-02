"""Visualizes best/worst/ground-truth scene matches for each caption.

For each human-text caption, finds the best-matched scene, worst-matched scene,
and its own (ground-truth) scene, then shows all three with matched objects highlighted.

Controls:
    SPACE / ENTER - next caption
    q / ESC       - quit
"""

import sys
from pathlib import Path

import numpy as np
import torch
import open3d as o3d
import torch.nn.functional as F

import hydra
from omegaconf import DictConfig

from src.data_processing.scene_graph import SceneGraph
from src.analysis.helper import get_matching_subgraph
from src.models.model_graph2graph import BigGNN
from src.localization.grid import load_scene
from src.localization.visualization import colour_objects


@torch.inference_mode()
def compute_match_score(model, qg: SceneGraph, sg: SceneGraph, device: str,
                        score_blend_weight: float = 0.5,
                        dbscan_eps: float = 0.5,
                        dbscan_min_samples: int = 1):
    """Computes a blended matching score between a query graph and a scene graph.

    Extracts matching subgraphs, converts to PyG format, runs through the model,
    and returns a blended score in [0, 1].

    Args:
        model: Trained BigGNN model.
        qg: Query (text) scene graph.
        sg: Database (3DSSG) scene graph.
        device: Torch device string.
        score_blend_weight: Weight for matching_prob; cosine gets ``(1 - weight)``.
        dbscan_eps: DBSCAN epsilon parameter for cosine distance.
        dbscan_min_samples: DBSCAN minimum samples per cluster.

    Returns:
        Blended score in [0, 1].
    """
    q_sub, s_sub = get_matching_subgraph(qg, sg, dbscan_eps, dbscan_min_samples)

    def is_degenerate(g: SceneGraph):
        return (g is None or
                len(g.nodes) <= 1 or
                (hasattr(g, 'edge_idx') and len(g.edge_idx[0]) < 1))
    if is_degenerate(q_sub) or is_degenerate(s_sub):
        q_sub, s_sub = qg, sg

    def prep(g: SceneGraph):
        n, e, f = g.to_pyg()
        return (
            torch.tensor(np.array(n), dtype=torch.float32, device=device),
            torch.tensor(np.array(e[0:2]), dtype=torch.int64, device=device),
            torch.tensor(np.array(f), dtype=torch.float32, device=device),
        )

    x_n, x_e, x_f = prep(q_sub)
    p_n, p_e, p_f = prep(s_sub)

    x_p, p_p, m_p = model(x_n, p_n, x_e, p_e, x_f, p_f)

    cos_sim = (F.cosine_similarity(x_p, p_p, dim=0).item() + 1.0) / 2.0
    w = score_blend_weight
    return w * m_p.item() + (1 - w) * cos_sim


def visualize_match(scan_root: Path, qg: SceneGraph, sg: SceneGraph,
                    dbscan_eps: float = 0.5, dbscan_min_samples: int = 1):
    """Visualizes a single query-graph vs one scene-graph.

    Loads the mesh, finds matched objects via DBSCAN overlap, colours them,
    and shows an interactive Open3D window.

    Args:
        scan_root: Parent directory containing scan subdirectories.
        qg: Query (text) scene graph.
        sg: Database (3DSSG) scene graph.
        dbscan_eps: DBSCAN epsilon parameter for cosine distance.
        dbscan_min_samples: DBSCAN minimum samples per cluster.
    """
    _, sub3d = get_matching_subgraph(qg, sg, dbscan_eps, dbscan_min_samples)
    matched = list(sub3d.nodes) if sub3d else []

    mesh, _, obj2faces = load_scene(scan_root / sg.scene_id)
    vis_mesh = colour_objects(mesh, obj2faces, matched, base=(0.6, 0.6, 0.6))

    print(f"\n>>> Query scene {qg.scene_id} vs scene {sg.scene_id}")
    print("Matched object IDs:", matched)
    print(" SPACE/ENTER -> close window;  q/Esc -> quit")

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name=f"{qg.scene_id}->{sg.scene_id}", width=1024, height=768)
    vis.add_geometry(vis_mesh)
    vis.register_key_callback(32,  lambda v: v.close())   # SPACE
    vis.register_key_callback(257, lambda v: v.close())   # ENTER
    vis.register_key_callback(81,  lambda v: sys.exit(0)) # q
    vis.register_key_callback(256, lambda v: sys.exit(0)) # ESC
    vis.run()
    vis.destroy_window()


def run_visualization_graph_object(cfg: DictConfig) -> None:
    """Runs the interactive best/worst/GT visualization loop for all captions.

    Args:
        cfg: Merged Hydra configuration.
    """
    device = cfg.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if cfg.paths.rscan_root is None:
        raise ValueError("paths.rscan_root is required. Place 3RScan data in ./data/3rscan or override paths.rscan_root=...")

    # load scene-graphs
    raw3d = torch.load(cfg.paths.graphs_3dssg, map_location="cpu", weights_only=False)
    database_3dssg = {
      sid: SceneGraph(sid, graph_type="3dssg", graph=g,
                      max_dist=cfg.graph.max_dist,
                      embedding_type=cfg.graph.embedding_type,
                      use_attributes=cfg.graph.use_attributes)
      for sid, g in raw3d.items()
    }

    # load text-graphs
    rawtxt = torch.load(cfg.paths.scanscribe_text, map_location="cpu", weights_only=False)
    dataset = [
      SceneGraph(k.split("_")[0], txt_id=None,
                 graph=g, graph_type="human",
                 embedding_type=cfg.graph.embedding_type,
                 use_attributes=cfg.graph.use_attributes)
      for k, g in rawtxt.items()
    ]

    # optional GNN
    model = None
    ckpt_dir = Path(cfg.paths.checkpoint_dir)
    if cfg.eval.model_name is not None:
        ckpt_path = ckpt_dir / f"{cfg.eval.model_name}.pt"
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model = BigGNN(cfg.model.N, cfg.model.heads, cfg.model.embed_dim, cfg.model.dropout).to(device)
        model.load_state_dict(ckpt)
        model.eval()

    scan_root = Path(cfg.paths.rscan_root)

    # for each caption, find best, worst, and gt scenes, then show them
    for qg in dataset:
        scores = {
          sid: compute_match_score(model, qg, sg, device,
                                   cfg.inference.score_blend_weight,
                                   cfg.graph.dbscan_eps,
                                   cfg.graph.dbscan_min_samples)
          for sid, sg in database_3dssg.items()
        }
        best_sid  = max(scores,  key=scores.get)
        worst_sid = min(scores,  key=scores.get)
        gt_sid    = qg.scene_id

        print("\n" + "="*60)
        print(f"Query caption for scene: {gt_sid}")
        print(f" -> Best match:  {best_sid}  (score={scores[best_sid]:.4f})")
        print(f" -> Worst match: {worst_sid} (score={scores[worst_sid]:.4f})")
        print(f" -> GroundTruth: {gt_sid}  (score={scores[gt_sid]:.4f})")

        db_eps = cfg.graph.dbscan_eps
        db_ms = cfg.graph.dbscan_min_samples
        visualize_match(scan_root, qg, database_3dssg[best_sid], db_eps, db_ms)
        visualize_match(scan_root, qg, database_3dssg[worst_sid], db_eps, db_ms)
        visualize_match(scan_root, qg, database_3dssg[gt_sid], db_eps, db_ms)


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Hydra CLI entry point for best/worst/GT visualization."""
    run_visualization_graph_object(cfg)


if __name__ == "__main__":
    main()
