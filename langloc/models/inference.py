"""Batch text-to-scene retrieval: ScanScribe caption graph to top-k matching 3D-SSG scenes."""

from __future__ import annotations

import json
import time

import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

import hydra
from omegaconf import DictConfig

from langloc.data_processing.scene_graph import SceneGraph
from langloc.analysis.helper import get_matching_subgraph
from langloc.models.model_graph2graph import BigGNN


@torch.inference_mode()
def compute_match_score(model: BigGNN | None,
                        qg: SceneGraph,
                        sg: SceneGraph,
                        device: str = "cpu",
                        score_blend_weight: float = 0.5,
                        dbscan_eps: float = 0.5,
                        dbscan_min_samples: int = 1) -> float:
    """Computes a blended matching score between a query graph and a scene graph.

    Extracts matching subgraphs, converts to PyG format, runs through the model
    (or falls back to cosine-only), and returns a score in [0, 1].

    Args:
        model: Trained BigGNN model, or None for cosine-only scoring.
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

    def bad(g):
        return (g is None or len(g.nodes) <= 1
                or (hasattr(g, "edge_idx") and len(g.edge_idx[0]) < 1))

    if bad(q_sub) or bad(s_sub):
        q_sub, s_sub = qg, sg

    def prep(g: SceneGraph):
        n, e, f = g.to_pyg()
        return (torch.tensor(np.array(n), dtype=torch.float32, device=device),
                torch.tensor(np.array(e[0:2]), dtype=torch.int64, device=device),
                torch.tensor(np.array(f), dtype=torch.float32, device=device))

    q_n, q_e, q_f = prep(q_sub)
    s_n, s_e, s_f = prep(s_sub)

    if model is None:
        cos = F.cosine_similarity(q_n.mean(0, keepdim=True),
                                  s_n.mean(0, keepdim=True), dim=1).item()
        return (cos + 1) / 2

    q_emb, s_emb, m_p = model(q_n, s_n, q_e, s_e, q_f, s_f)
    cos = (F.cosine_similarity(q_emb, s_emb, dim=0).item() + 1) / 2
    w = score_blend_weight
    return w * m_p.item() + (1 - w) * cos


def run_inference(cfg: DictConfig) -> None:
    """Runs batch text-to-scene retrieval over all ScanScribe captions.

    Args:
        cfg: Merged Hydra configuration.
    """
    device = cfg.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    t0 = time.perf_counter()

    # Load graphs
    g3d_raw = torch.load(cfg.paths.graphs_3dssg, map_location="cpu", weights_only=False)
    scans_raw = torch.load(cfg.paths.scanscribe_text, map_location="cpu", weights_only=False)

    database_3dssg = {
        sid: SceneGraph(sid, graph_type="3dssg", graph=g,
                        max_dist=cfg.graph.max_dist,
                        embedding_type=cfg.graph.embedding_type,
                        use_attributes=cfg.graph.use_attributes)
        for sid, g in g3d_raw.items()
    }
    queries = [
        SceneGraph(k.split("_")[0], txt_id=None,
                   graph=g, graph_type="scanscribe",
                   embedding_type=cfg.graph.embedding_type,
                   use_attributes=cfg.graph.use_attributes)
        for k, g in scans_raw.items()
    ]
    print(f"Loaded {len(queries)} ScanScribe captions, "
          f"{len(database_3dssg)} 3D-SSG scenes.")

    # Load model
    ckpt_dir = Path(cfg.paths.checkpoint_dir)
    if cfg.eval.model_name is None:
        raise ValueError("eval.model_name is required. Set via CLI: eval.model_name=my_model")
    ckpt_path = ckpt_dir / f"{cfg.eval.model_name}.pt"

    model = BigGNN(cfg.model.N, cfg.model.heads, cfg.model.embed_dim, cfg.model.dropout).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False))
    model.eval()

    # For each caption, rank all scenes
    top_k = cfg.inference.top_k
    jsonl_out = cfg.inference.jsonl_out
    jsonl_fh = open(jsonl_out, "w") if jsonl_out else None
    try:
        for qi, qg in enumerate(queries, 1):
            scores = {
                sid: compute_match_score(model, qg, sg, device,
                                         cfg.inference.score_blend_weight,
                                         cfg.graph.dbscan_eps,
                                         cfg.graph.dbscan_min_samples)
                for sid, sg in database_3dssg.items()
            }
            best = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]

            print(f"\nQuery {qi:>4}/{len(queries)}  (scene_id={qg.scene_id})")
            for rank, (sid, sc) in enumerate(best, 1):
                gt_tag = "  *GT*" if sid == qg.scene_id else ""
                print(f"  {rank:>2}. {sid:<18}  score={sc:5.3f}{gt_tag}")

            if jsonl_fh:
                jsonl_fh.write(json.dumps({
                    "query_scene_id": qg.scene_id,
                    "top_k": best
                }) + "\n")
    finally:
        if jsonl_fh:
            jsonl_fh.close()

    if jsonl_out:
        print(f"\nWrote ranked lists to {jsonl_out}")

    print(f"\nFinished in {(time.perf_counter()-t0):.1f}s.")


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Hydra CLI entry point for batch inference."""
    run_inference(cfg)


if __name__ == "__main__":
    main()
