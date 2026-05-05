"""Precompute embedding caches for retrieval evaluation.

Ported from Shirley's `precompute_eval_embeddings.py` (transcribed from
``VLSG.ipynb`` cells 34/38/40). Loads ``DualSceneAlignerV2 +
SimpleGraphMatcher`` from a checkpoint and writes:

  - ``clip_embedding_cache.pt`` — node/relation/scene-level CLIP embeddings
  - ``db_emb_cache.pt`` — model-projected DB scene embeddings + scene-CLIP + base-label sets
  - ``query_emb_cache.pt`` — same fields per query

These caches are consumed by ``langloc/retrieval/eval.py`` (Tables 1+2).
With ``epoch_70_163_cliprel.pth`` as the checkpoint the produced caches
reproduce paper Tables 1-2 numbers within reporting noise.

Run from repo root::

    python -m scripts.retrieval.precompute_eval_embeddings \
        --checkpoint VLSG_TEXT_v2/VLSG_Files/checkpoints/epoch_70_163_cliprel.pth \
        --cache_dir   VLSG_TEXT_v2/VLSG_Files \
        --device cuda
"""
from __future__ import annotations

import argparse
from pathlib import Path

import clip
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from langloc.graphs.scene_graph import SceneGraph
from langloc.retrieval.models.dual_scene_aligner_v2 import DualSceneAlignerV2
from langloc.retrieval.models.simple_graph_matcher import SimpleGraphMatcher


def get_clip_embedding(text: str, clip_model: nn.Module, device: torch.device) -> torch.Tensor:
    """L2-normalized CLIP text embedding (Shirley's exact form)."""
    with torch.no_grad():
        tokens = clip.tokenize([text]).to(device)
        emb = clip_model.encode_text(tokens)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb[0].cpu()


def get_base_label(label: str) -> str:
    """Strip spatial qualifiers from unique labels (Shirley's exact form)."""
    parts = label.split("_")
    spatial = {"north", "south", "east", "west", "center", "upper", "middle", "lower"}
    base: list[str] = []
    for part in parts:
        if part in spatial:
            break
        base.append(part)
    return "_".join(base) if base else label


def build_batch_from_cache(
    graph: SceneGraph,
    node_clip_cache: dict[str, torch.Tensor],
    rel_clip_cache: dict[str, torch.Tensor],
    scene_clip_cache: dict[str, torch.Tensor],
    graph_key: str,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Build a single-graph batch with cached CLIP embeddings (matches the form
    used inside ``DualSceneGraphDataset`` so the model sees identical inputs at
    train and eval)."""
    node_feats: list[torch.Tensor] = []
    for nid in graph.nodes:
        label = graph.nodes[nid].label
        node_clip = node_clip_cache.get(label, torch.zeros(512))
        # ScanScribe / 3DSSG nodes have no usable centroid/color in eval graphs;
        # zero-pad those dims (matches Shirley's precompute).
        feat = torch.cat([torch.zeros(6), node_clip])
        node_feats.append(feat)
    node_feats = torch.stack(node_feats)

    edge_idx = graph.edge_idx
    num_nodes = len(graph.nodes)

    if len(edge_idx) > 0 and len(edge_idx[0]) > 0:
        edges = torch.tensor(edge_idx, dtype=torch.long)
        valid_mask = (edges[0] < num_nodes) & (edges[1] < num_nodes)
        edges = edges[:, valid_mask]
        num_edges = edges.size(1)
        geom_attr = torch.zeros(num_edges, 8)
        if hasattr(graph, "edge_relations") and graph.edge_relations:
            valid_indices = valid_mask.nonzero(as_tuple=True)[0].tolist()
            rel_embs = [
                rel_clip_cache.get(str(graph.edge_relations[i]).lower(), torch.zeros(512))
                for i in valid_indices
            ]
            text_attr = torch.stack(rel_embs) if rel_embs else torch.zeros(0, 512)
        else:
            text_attr = torch.zeros(num_edges, 512)
    else:
        edges = torch.zeros(2, 0, dtype=torch.long)
        geom_attr = torch.zeros(0, 8)
        text_attr = torch.zeros(0, 512)

    scene_clip = scene_clip_cache.get(graph_key, torch.zeros(512))

    return {
        "node_feats_src": node_feats.to(device),
        "geom_edges_src": edges.to(device),
        "geom_attr_src": geom_attr.to(device),
        "text_edges_src": edges.clone().to(device),
        "text_attr_src": text_attr.to(device),
        "node_feats_ref": node_feats.to(device),
        "geom_edges_ref": edges.to(device),
        "geom_attr_ref": geom_attr.to(device),
        "text_edges_ref": edges.clone().to(device),
        "text_attr_ref": text_attr.to(device),
        "src_batch": torch.zeros(node_feats.size(0), dtype=torch.long, device=device),
        "ref_batch": torch.zeros(node_feats.size(0), dtype=torch.long, device=device),
        "batch_size": 1,
        "scene_clip_src": scene_clip.unsqueeze(0).to(device),
        "scene_clip_ref": scene_clip.unsqueeze(0).to(device),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--checkpoint",
        required=True,
        help="Path to a DualSceneAlignerV2 + SimpleGraphMatcher checkpoint "
        "(e.g. epoch_70_163_cliprel.pth from the Colab artifact).",
    )
    ap.add_argument(
        "--cache_dir",
        required=True,
        help="Directory holding 3dssg_graphs_518D.pt and "
        "scanscribe_graphs_test_518D.pt; will receive the produced caches.",
    )
    ap.add_argument("--clip_model", default="ViT-B/32")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    args = ap.parse_args()

    if args.device == "auto":
        device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load CLIP
    # ------------------------------------------------------------------
    print("Loading CLIP…")
    clip_model, _ = clip.load(args.clip_model, device=device)

    # ------------------------------------------------------------------
    # Load graphs
    # ------------------------------------------------------------------
    print("\nLoading 3DSSG database graphs…")
    raw_3dssg = torch.load(
        cache_dir / "3dssg_graphs_518D.pt",
        weights_only=False, map_location="cpu",
    )
    db_graphs = {}
    for sid in tqdm(raw_3dssg, desc="3DSSG"):
        db_graphs[sid] = SceneGraph(
            sid, graph_type="3dssg", graph=raw_3dssg[sid],
            max_dist=1.0, embedding_type="word2vec", use_attributes=True,
        )

    print("\nLoading ScanScribe test queries…")
    raw_test = torch.load(
        cache_dir / "scanscribe_graphs_test_518D.pt",
        weights_only=False, map_location="cpu",
    )
    query_graphs: dict[str, SceneGraph] = {}
    for sid in tqdm(raw_test, desc="ScanScribe test"):
        for tid in raw_test[sid]:
            key = f"{sid}_{str(tid).zfill(5)}"
            try:
                g = SceneGraph(
                    sid, txt_id=tid, graph_type="scanscribe",
                    graph=raw_test[sid][tid],
                    embedding_type="word2vec", use_attributes=True,
                )
                if len(g.edge_idx[0]) >= 1:
                    query_graphs[key] = g
            except Exception:
                continue

    # ------------------------------------------------------------------
    # Step 1: per-label / per-relation / per-scene CLIP caches
    # ------------------------------------------------------------------
    print("\nCollecting unique labels and relations…")
    all_labels: set[str] = set()
    all_relations: set[str] = set()
    for g in list(query_graphs.values()) + list(db_graphs.values()):
        for nid in g.nodes:
            all_labels.add(g.nodes[nid].label)
        if hasattr(g, "edge_relations") and g.edge_relations:
            for r in g.edge_relations:
                all_relations.add(str(r).lower())
    print(f"  unique labels:    {len(all_labels)}")
    print(f"  unique relations: {len(all_relations)}")

    node_clip_cache: dict[str, torch.Tensor] = {}
    for label in tqdm(all_labels, desc="Node CLIP"):
        node_clip_cache[label] = get_clip_embedding(label, clip_model, device)

    rel_clip_cache: dict[str, torch.Tensor] = {}
    for rel in tqdm(all_relations, desc="Rel CLIP"):
        rel_clip_cache[rel] = get_clip_embedding(rel, clip_model, device)

    # Scene-CLIP per graph: ``"A room with l_1, ..., l_K"`` (first 10 unique
    # labels). Matches Shirley's notebook exactly.
    scene_clip_cache: dict[str, torch.Tensor] = {}
    all_for_scene = list(query_graphs.items()) + [(sid, g) for sid, g in db_graphs.items()]
    for key, g in tqdm(all_for_scene, desc="Scene CLIP"):
        labels = [g.nodes[nid].label for nid in g.nodes]
        unique_labels = list(set(labels))[:10]
        scene_desc = f"A room with {', '.join(unique_labels)}"
        scene_clip_cache[key] = get_clip_embedding(scene_desc, clip_model, device)

    torch.save(
        {"node_clip": node_clip_cache, "rel_clip": rel_clip_cache, "scene_clip": scene_clip_cache},
        cache_dir / "clip_embedding_cache.pt",
    )
    print(f"\nSaved clip_embedding_cache.pt ({len(node_clip_cache)} labels, "
          f"{len(rel_clip_cache)} relations, {len(scene_clip_cache)} scene CLIPs)")

    # ------------------------------------------------------------------
    # Step 2: load model and produce DB / query embedding caches
    # ------------------------------------------------------------------
    print("\nLoading model…")
    base_model = DualSceneAlignerV2(node_input_dim=518, hidden_dim=256, dropout=0.1).to(device)
    model = SimpleGraphMatcher(base_model, scene_clip_dim=512, hidden_dim=256).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(sd, strict=True)
    model.eval()
    print(f"  loaded checkpoint epoch={ckpt.get('epoch','?')}, "
          f"params={sum(p.numel() for p in model.parameters()):,}")

    # ------------------------------------------------------------------
    # DB embeddings
    # ------------------------------------------------------------------
    print("\nPrecomputing 3DSSG database embeddings…")
    db_emb_cache: dict[str, dict] = {}
    with torch.no_grad():
        for scene_id, g in tqdm(db_graphs.items(), desc="DB"):
            batch = build_batch_from_cache(
                g, node_clip_cache, rel_clip_cache, scene_clip_cache,
                scene_id, device,
            )
            out = model(
                batch,
                scene_clip_src=batch["scene_clip_src"],
                scene_clip_ref=batch["scene_clip_ref"],
            )
            db_emb_cache[scene_id] = {
                "emb": F.normalize(out["ref_emb"], dim=-1).cpu(),
                "scene_clip": batch["scene_clip_ref"].cpu(),
                "labels": {get_base_label(g.nodes[nid].label) for nid in g.nodes},
            }
    torch.save(db_emb_cache, cache_dir / "db_emb_cache.pt")
    print(f"Saved db_emb_cache.pt ({len(db_emb_cache)} scenes)")

    # ------------------------------------------------------------------
    # Query embeddings
    # ------------------------------------------------------------------
    print("\nPrecomputing query embeddings…")
    query_emb_cache: dict[str, dict] = {}
    with torch.no_grad():
        for key, g in tqdm(query_graphs.items(), desc="Queries"):
            batch = build_batch_from_cache(
                g, node_clip_cache, rel_clip_cache, scene_clip_cache, key, device,
            )
            out = model(
                batch,
                scene_clip_src=batch["scene_clip_src"],
                scene_clip_ref=batch["scene_clip_ref"],
            )
            query_emb_cache[key] = {
                "emb": F.normalize(out["src_emb"], dim=-1).cpu(),
                "scene_clip": batch["scene_clip_src"].cpu(),
                "labels": {get_base_label(g.nodes[nid].label) for nid in g.nodes},
                "scene_id": g.scene_id,
            }
    torch.save(query_emb_cache, cache_dir / "query_emb_cache.pt")
    print(f"Saved query_emb_cache.pt ({len(query_emb_cache)} queries)")

    print("\nAll caches written. Run reproduction with:")
    print(f"  python -m langloc.retrieval.eval --cache_dir {cache_dir} --mode both")


if __name__ == "__main__":
    main()
