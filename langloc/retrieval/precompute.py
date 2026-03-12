"""Precompute embeddings for cache-based evaluation (Table 3).

Encodes all database and query graphs through the model, saves caches
for fast evaluation with ``eval.py retrieval.eval.protocol=table3``.
"""

import torch
import torch.nn.functional as F
import numpy as np
import clip
from tqdm import tqdm

import hydra
from omegaconf import DictConfig

from langloc.graphs.scene_graph import SceneGraph
from langloc.retrieval.models.dual_scene_aligner import DualSceneAligner
from langloc.retrieval.eval import (
    get_clip_embedding, get_base_label, build_single_batch, build_clip_caches,
)


@hydra.main(config_path="../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Hydra CLI entry point for precomputing embeddings.

    Args:
        cfg: Merged Hydra configuration.
    """
    rcfg = cfg.retrieval
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    assert rcfg.checkpoint, "Set retrieval.checkpoint"
    assert rcfg.cache_dir, "Set retrieval.cache_dir"
    cache_dir = rcfg.cache_dir

    print("Loading CLIP...")
    clip_model, _ = clip.load(rcfg.clip_model, device=device)

    print("Loading graphs...")
    raw_3dssg = torch.load(
        f"{cache_dir}/3dssg_graphs_518D.pt", weights_only=False, map_location="cpu"
    )
    db_graphs = {}
    for sid in tqdm(raw_3dssg, desc="3DSSG"):
        db_graphs[sid] = SceneGraph(
            sid, graph_type="3dssg", graph=raw_3dssg[sid],
            max_dist=rcfg.max_dist, embedding_type=rcfg.embedding_type,
            use_attributes=True,
        )

    raw_test = torch.load(
        f"{cache_dir}/scanscribe_graphs_test_518D.pt",
        weights_only=False, map_location="cpu",
    )
    query_graphs = {}
    for sid in tqdm(raw_test, desc="Queries"):
        for tid in raw_test[sid]:
            key = f"{sid}_{str(tid).zfill(5)}"
            query_graphs[key] = SceneGraph(
                sid, txt_id=tid, graph_type="scanscribe",
                graph=raw_test[sid][tid],
                embedding_type=rcfg.embedding_type, use_attributes=True,
            )
    query_graphs = {k: v for k, v in query_graphs.items() if len(v.edge_idx[0]) >= 1}
    print(f"DB: {len(db_graphs)}, Queries: {len(query_graphs)}")

    all_graphs = {**query_graphs, **{sid: g for sid, g in db_graphs.items()}}
    node_clip, rel_clip, scene_clip = build_clip_caches(all_graphs, clip_model, device)

    torch.save(
        {"node_clip": node_clip, "rel_clip": rel_clip, "scene_clip": scene_clip},
        f"{cache_dir}/clip_embedding_cache.pt",
    )
    print(f"Saved clip_embedding_cache.pt")

    print("Loading model...")
    model = DualSceneAligner(
        node_input_dim=rcfg.node_input_dim,
        hidden_dim=rcfg.hidden_dim,
        dropout=0.0,
    ).to(device)
    ckpt = torch.load(rcfg.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Model: {sum(p.numel() for p in model.parameters()):,} parameters")

    print("Precomputing DB embeddings...")
    db_emb_cache = {}
    with torch.no_grad():
        for sid, g in tqdm(db_graphs.items(), desc="DB"):
            batch = build_single_batch(g, node_clip, rel_clip, scene_clip, sid, device)
            out = model(batch)
            db_emb_cache[sid] = {
                "emb": F.normalize(out["ref_emb"], dim=-1).cpu(),
                "scene_clip": batch["scene_clip_ref"].cpu(),
                "labels": {get_base_label(g.nodes[nid].label) for nid in g.nodes},
            }
    torch.save(db_emb_cache, f"{cache_dir}/db_emb_cache.pt")
    print(f"Saved db_emb_cache.pt ({len(db_emb_cache)} scenes)")

    print("Precomputing query embeddings...")
    query_emb_cache = {}
    with torch.no_grad():
        for key, g in tqdm(query_graphs.items(), desc="Queries"):
            batch = build_single_batch(g, node_clip, rel_clip, scene_clip, key, device)
            out = model(batch)
            query_emb_cache[key] = {
                "emb": F.normalize(out["src_emb"], dim=-1).cpu(),
                "scene_clip": batch["scene_clip_src"].cpu(),
                "labels": {get_base_label(g.nodes[nid].label) for nid in g.nodes},
                "scene_id": g.scene_id,
            }
    torch.save(query_emb_cache, f"{cache_dir}/query_emb_cache.pt")
    print(f"Saved query_emb_cache.pt ({len(query_emb_cache)} queries)")

    print("\nDone. Run evaluation with:")
    print(f"  python -m langloc.retrieval.eval retrieval.eval.protocol=table3 "
          f"retrieval.cache_dir={cache_dir}")


if __name__ == "__main__":
    main()
