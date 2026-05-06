"""
Precompute embeddings for cache-based evaluation (Table 3).

Encodes all database and query graphs through the model, saves caches
for fast evaluation with eval.py retrieval.eval.protocol=table3.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import clip
from tqdm import tqdm
from torch_scatter import scatter_mean

import hydra
from omegaconf import DictConfig

from langloc.graphs.scene_graph import SceneGraph
from langloc.retrieval.models.dual_scene_aligner import DualSceneAligner
from langloc.retrieval.models.networks.edge_gat import MultiGAT_Edge
from langloc.retrieval.eval import (
    get_clip_embedding, get_base_label, build_single_batch, build_clip_caches,
)


class LegacyDualSceneAligner(nn.Module):
    """
    Checkpoint architecture: base_model (GatedFusion, final_proj=Linear(256,256))
    + top-level fusion(cat([gnn_out, scene_clip])) → 256.
    """

    def __init__(self, node_input_dim=518, hidden_dim=256, dropout=0.1):
        super().__init__()
        # base_model mirrors current DualSceneAligner but final_proj takes 256 (no scene_clip)
        self.base_model = DualSceneAligner(
            node_input_dim=node_input_dim, hidden_dim=hidden_dim, dropout=dropout)
        # override final_proj to match checkpoint (input=hidden_dim, not hidden_dim+512)
        self.base_model.final_proj = nn.Sequential(
            nn.Linear(hidden_dim, 256), nn.LayerNorm(256), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(256, 256),
        )
        # top-level fusion: cat([gnn_out(256), scene_clip(512)]) → 256
        self.fusion = nn.Sequential(
            nn.LayerNorm(hidden_dim + 512),
            nn.Linear(hidden_dim + 512, hidden_dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim),
        )

    def _encode(self, node_feats, geom_edges, geom_attr,
                text_edges, text_attr, batch, scene_clip):
        bm = self.base_model
        x = bm.node_encoder(node_feats)
        g = bm.norm_geom(bm.gat_geom(x, geom_edges, geom_attr.float()))
        t = bm.norm_text(bm.gat_text(x, text_edges, text_attr.float())) \
            if text_edges.size(1) > 0 else g
        h = bm.fusion(g, t)
        pooled = scatter_mean(h, batch, dim=0)
        gnn_out = bm.final_proj(pooled)          # Linear(256→256), no scene_clip
        return self.fusion(torch.cat([gnn_out, scene_clip], dim=-1))

    def forward(self, batch):
        return {
            "src_emb": self._encode(
                batch["node_feats_src"], batch["geom_edges_src"], batch["geom_attr_src"],
                batch["text_edges_src"], batch["text_attr_src"],
                batch["src_batch"], batch["scene_clip_src"],
            ),
            "ref_emb": self._encode(
                batch["node_feats_ref"], batch["geom_edges_ref"], batch["geom_attr_ref"],
                batch["text_edges_ref"], batch["text_attr_ref"],
                batch["ref_batch"], batch["scene_clip_ref"],
            ),
        }


@hydra.main(config_path="../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    rcfg = cfg.retrieval
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    assert rcfg.checkpoint, "Set retrieval.checkpoint"
    assert rcfg.cache_dir, "Set retrieval.cache_dir"
    cache_dir = rcfg.cache_dir

    # Load CLIP
    print("Loading CLIP...")
    clip_model, _ = clip.load("ViT-B/32", device=device)

    # Load graphs
    print("Loading graphs...")
    graphs_3dssg_path = rcfg.get("graphs_3dssg", None) or f"{cache_dir}/3dssg_graphs_518D.pt"
    raw_3dssg = torch.load(graphs_3dssg_path, weights_only=False, map_location="cpu")
    db_graphs = {}
    for sid in tqdm(raw_3dssg, desc="3DSSG"):
        db_graphs[sid] = SceneGraph(
            sid, graph_type="3dssg", graph=raw_3dssg[sid],
            max_dist=rcfg.max_dist, embedding_type=rcfg.embedding_type,
            use_attributes=True,
        )

    graphs_scanscribe_img = rcfg.get("graphs_scanscribe_img", None)
    query_graphs_path = graphs_scanscribe_img or f"{cache_dir}/scanscribe_graphs_test_518D.pt"
    raw_test = torch.load(query_graphs_path, weights_only=False, map_location="cpu")
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

    # Build CLIP caches
    all_graphs = {**query_graphs, **{sid: g for sid, g in db_graphs.items()}}
    node_clip, rel_clip, scene_clip = build_clip_caches(all_graphs, clip_model, device)

    torch.save(
        {"node_clip": node_clip, "rel_clip": rel_clip, "scene_clip": scene_clip},
        f"{cache_dir}/clip_embedding_cache.pt",
    )
    print(f"Saved clip_embedding_cache.pt")

    # Load model — auto-detect legacy vs current architecture from checkpoint
    print("Loading model...")
    ckpt = torch.load(rcfg.checkpoint, map_location=device, weights_only=False)
    state = ckpt["model_state_dict"]
    # Detect legacy: has top-level fusion.0 (wrapper) AND base_model.* keys
    is_legacy = any(k.startswith("fusion.0") for k in state)
    print(f"  Architecture: {'legacy wrapper' if is_legacy else 'current'} DualSceneAligner")
    model = LegacyDualSceneAligner(
        node_input_dim=rcfg.node_input_dim,
        hidden_dim=rcfg.hidden_dim,
        dropout=0.0,
    ).to(device) if is_legacy else DualSceneAligner(
        node_input_dim=rcfg.node_input_dim,
        hidden_dim=rcfg.hidden_dim,
        dropout=0.0,
    ).to(device)
    if not is_legacy and any(k.startswith("base_model.") for k in state):
        state = {k.replace("base_model.", "", 1): v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()
    print(f"Model: {sum(p.numel() for p in model.parameters()):,} parameters")

    # Precompute DB embeddings
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

    # Precompute query embeddings
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
