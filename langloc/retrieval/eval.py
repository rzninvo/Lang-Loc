"""
Unified evaluation script for scene retrieval (Tables 1, 2, 3).

Usage:
  python -m langloc.retrieval.eval retrieval.eval.protocol=table1 retrieval.checkpoint=<path>
  python -m langloc.retrieval.eval retrieval.eval.protocol=table2 retrieval.checkpoint=<path>
  python -m langloc.retrieval.eval retrieval.eval.protocol=table3 retrieval.cache_dir=<path>
"""

import torch
import torch.nn.functional as F
import numpy as np
import random
import clip
from tqdm import tqdm

import hydra
from omegaconf import DictConfig

from langloc.graphs.scene_graph import SceneGraph
from langloc.retrieval.models.dual_scene_aligner import DualSceneAligner


# ── Helpers ───────────────────────────────────────────────────

def get_base_label(label):
    """Remove spatial modifiers from label."""
    parts = label.split("_")
    spatial = {"north", "south", "east", "west", "center", "upper", "middle", "lower"}
    base = []
    for part in parts:
        if part in spatial:
            break
        base.append(part)
    return "_".join(base) if base else label


def get_clip_embedding(text, clip_model, device):
    with torch.no_grad():
        tokens = clip.tokenize([text]).to(device)
        emb = clip_model.encode_text(tokens)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb[0].cpu()


def score_pair(q_cache, db_cache, w_emb, w_scene, w_jac):
    """Compute Eq.8 scoring between query and database entry."""
    emb_sim = (q_cache["emb"] * db_cache["emb"]).sum().item()
    scene_sim = F.cosine_similarity(
        q_cache["scene_clip"], db_cache["scene_clip"]
    ).item()
    overlap = len(q_cache["labels"] & db_cache["labels"])
    if len(q_cache["labels"]) > 0 and len(db_cache["labels"]) > 0:
        prec = overlap / len(db_cache["labels"])
        rec = overlap / len(q_cache["labels"])
        f1 = (2 * prec * rec) / (prec + rec + 1e-8)
    else:
        f1 = 0
    return w_emb * emb_sim + w_scene * scene_sim + w_jac * f1


# ── Build batch for single graph ─────────────────────────────

def build_single_batch(graph, node_clip_cache, rel_clip_cache,
                       scene_clip_cache, graph_key, device):
    """Build a batch dict for a single graph (for on-the-fly embedding)."""
    node_feats = []
    for nid in graph.nodes:
        label = graph.nodes[nid].label
        nc = node_clip_cache.get(label, torch.zeros(512))
        feat = torch.cat([torch.zeros(6), nc])
        node_feats.append(feat)
    node_feats = torch.stack(node_feats)

    edge_idx = graph.edge_idx
    num_nodes = len(graph.nodes)

    if len(edge_idx) > 0 and len(edge_idx[0]) > 0:
        edges = torch.tensor(edge_idx, dtype=torch.long)
        valid = (edges[0] < num_nodes) & (edges[1] < num_nodes)
        edges = edges[:, valid]
        ne = edges.size(1)
        geom_attr = torch.zeros(ne, 8)
        valid_idx = valid.nonzero(as_tuple=True)[0].tolist()
        if hasattr(graph, "edge_relations") and graph.edge_relations and valid_idx:
            rel_embs = [
                rel_clip_cache.get(str(graph.edge_relations[i]).lower(), torch.zeros(512))
                for i in valid_idx
            ]
            text_attr = torch.stack(rel_embs)
        else:
            text_attr = torch.zeros(ne, 512)
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
        "scene_clip_src": scene_clip.unsqueeze(0).to(device),
        "scene_clip_ref": scene_clip.unsqueeze(0).to(device),
    }


# ── CLIP caching ──────────────────────────────────────────────

def build_clip_caches(graphs_dict, clip_model, device):
    """Build node CLIP, relation CLIP, and scene CLIP caches."""
    all_labels, all_relations = set(), set()
    for g in graphs_dict.values():
        for nid in g.nodes:
            all_labels.add(g.nodes[nid].label)
        if hasattr(g, "edge_relations") and g.edge_relations:
            for r in g.edge_relations:
                all_relations.add(str(r).lower())

    node_clip = {}
    for label in tqdm(all_labels, desc="Node CLIP"):
        node_clip[label] = get_clip_embedding(label, clip_model, device)

    rel_clip = {}
    for rel in tqdm(all_relations, desc="Rel CLIP"):
        rel_clip[rel] = get_clip_embedding(rel, clip_model, device)

    scene_clip = {}
    for key, g in tqdm(graphs_dict.items(), desc="Scene CLIP"):
        labels = list({g.nodes[nid].label for nid in g.nodes})[:10]
        desc = f"A room with {', '.join(labels)}"
        scene_clip[key] = get_clip_embedding(desc, clip_model, device)

    return node_clip, rel_clip, scene_clip


# ── Embed all graphs ──────────────────────────────────────────

def embed_graphs(model, graphs, node_clip, rel_clip, scene_clip, device):
    """Compute embeddings for all graphs, return cache dict."""
    cache = {}
    with torch.no_grad():
        for key, g in tqdm(graphs.items(), desc="Embedding"):
            batch = build_single_batch(g, node_clip, rel_clip, scene_clip, key, device)
            out = model(batch)
            cache[key] = {
                "emb": F.normalize(out["src_emb"], dim=-1).cpu(),
                "scene_clip": batch["scene_clip_src"].cpu(),
                "labels": {get_base_label(g.nodes[nid].label) for nid in g.nodes},
            }
    return cache


# ── Evaluation loop ───────────────────────────────────────────

def eval_sampled(query_emb, db_emb, query_buckets, pool_buckets,
                 out_of, valid_top_k, eval_iters, eval_iter_count,
                 w_emb, w_scene, w_jac):
    """Sampled evaluation: pick 1 correct + (out_of-1) random candidates."""
    all_valid = {k: [] for k in valid_top_k}

    for _ in tqdm(range(eval_iters), desc="Eval rounds"):
        valid = {k: [] for k in valid_top_k}
        for _ in range(eval_iter_count):
            qsid = random.choice(list(query_buckets.keys()))
            qkey = random.choice(query_buckets[qsid])
            qc = query_emb[qkey]

            others = [s for s in pool_buckets if s != qsid]
            if len(others) < out_of - 1:
                continue
            sampled = random.sample(others, out_of - 1)
            candidates = [qsid] + sampled

            scores, sids = [], []
            for sid in candidates:
                if sid not in db_emb:
                    continue
                scores.append(score_pair(qc, db_emb[sid], w_emb, w_scene, w_jac))
                sids.append(sid)

            if not scores:
                continue

            order = np.argsort(scores)[::-1]
            for k in valid_top_k:
                if k <= len(order):
                    top_k = [sids[idx] for idx in order[:k]]
                    valid[k].append(1 if qsid in top_k else 0)

        for k in valid_top_k:
            if valid[k]:
                all_valid[k].append(np.mean(valid[k]))

    return {k: (np.mean(v), np.std(v)) for k, v in all_valid.items() if v}


def eval_full_pool(query_emb, db_emb, query_buckets,
                   valid_top_k, eval_iters, eval_iter_count,
                   w_emb, w_scene, w_jac):
    """Full-pool evaluation: rank against ALL database scenes."""
    all_valid = {k: [] for k in valid_top_k}
    db_keys = list(db_emb.keys())

    for _ in tqdm(range(eval_iters), desc="Eval rounds"):
        valid = {k: [] for k in valid_top_k}
        for _ in range(eval_iter_count):
            qsid = random.choice(list(query_buckets.keys()))
            qkey = random.choice(query_buckets[qsid])
            qc = query_emb[qkey]

            scores = []
            for sid in db_keys:
                scores.append(score_pair(qc, db_emb[sid], w_emb, w_scene, w_jac))

            order = np.argsort(scores)[::-1]
            for k in valid_top_k:
                if k <= len(order):
                    top_k = [db_keys[idx] for idx in order[:k]]
                    valid[k].append(1 if qsid in top_k else 0)

        for k in valid_top_k:
            if valid[k]:
                all_valid[k].append(np.mean(valid[k]))

    return {k: (np.mean(v), np.std(v)) for k, v in all_valid.items() if v}


# ── Main ──────────────────────────────────────────────────────

@hydra.main(config_path="../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    rcfg = cfg.retrieval
    ecfg = rcfg.eval
    protocol = ecfg.protocol

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Protocol: {protocol}")

    random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    proto = ecfg[protocol]

    if protocol == "table3":
        # ── Cache-based evaluation (Table 3) ──
        cache_dir = rcfg.cache_dir
        assert cache_dir, "Set retrieval.cache_dir for table3 evaluation"

        db_emb = torch.load(f"{cache_dir}/db_emb_cache.pt", weights_only=False)
        query_emb = torch.load(f"{cache_dir}/query_emb_cache.pt", weights_only=False)
        print(f"DB: {len(db_emb)} scenes, Queries: {len(query_emb)}")

        # Load pool graphs for bucket building
        pool_data = torch.load(
            f"{cache_dir}/scanscribe_graphs_test_518D.pt",
            weights_only=False, map_location="cpu",
        )
        pool_graphs = {}
        for sid in pool_data:
            for tid in pool_data[sid]:
                key = f"{sid}_{str(tid).zfill(5)}"
                pool_graphs[key] = SceneGraph(
                    sid, txt_id=tid, graph_type="scanscribe",
                    graph=pool_data[sid][tid],
                    embedding_type="word2vec", use_attributes=True,
                )
        pool_graphs = {k: v for k, v in pool_graphs.items() if len(v.edge_idx[0]) >= 1}

        query_buckets = {}
        for key, c in query_emb.items():
            sid = c["scene_id"]
            query_buckets.setdefault(sid, []).append(key)

        pool_buckets = {}
        for key, g in pool_graphs.items():
            pool_buckets.setdefault(g.scene_id, []).append(key)

        results = eval_sampled(
            query_emb, db_emb, query_buckets, pool_buckets,
            out_of=proto.out_of, valid_top_k=proto.valid_top_k,
            eval_iters=proto.eval_iters, eval_iter_count=proto.eval_iter_count,
            w_emb=rcfg.w_emb, w_scene=rcfg.w_scene, w_jac=rcfg.w_jac,
        )
    else:
        # ── Model-based evaluation (Tables 1 & 2) ──
        assert rcfg.checkpoint, "Set retrieval.checkpoint"

        print("Loading CLIP...")
        clip_model, _ = clip.load("ViT-B/32", device=device)

        # Load model
        model = DualSceneAligner(
            node_input_dim=rcfg.node_input_dim,
            hidden_dim=rcfg.hidden_dim,
            dropout=0.0,
        ).to(device)
        ckpt = torch.load(rcfg.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} params")

        # Load 3DSSG database graphs
        assert rcfg.graphs_3dssg, "Set retrieval.graphs_3dssg"
        raw_3dssg = torch.load(rcfg.graphs_3dssg, weights_only=False, map_location="cpu")
        db_graphs = {}
        for sid in tqdm(raw_3dssg, desc="3DSSG"):
            db_graphs[sid] = SceneGraph(
                sid, graph_type="3dssg", graph=raw_3dssg[sid],
                max_dist=rcfg.max_dist, embedding_type=rcfg.embedding_type,
                use_attributes=True,
            )

        # Load query graphs
        assert rcfg.graphs_scanscribe_test, "Set retrieval.graphs_scanscribe_test"
        raw_test = torch.load(rcfg.graphs_scanscribe_test, weights_only=False, map_location="cpu")
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
        all_graphs = {**db_graphs, **query_graphs}
        node_clip, rel_clip, scene_clip = build_clip_caches(all_graphs, clip_model, device)

        # Embed all graphs
        db_emb = embed_graphs(model, db_graphs, node_clip, rel_clip, scene_clip, device)
        query_emb = embed_graphs(model, query_graphs, node_clip, rel_clip, scene_clip, device)
        # Add scene_id to query cache
        for key, g in query_graphs.items():
            query_emb[key]["scene_id"] = g.scene_id

        query_buckets = {}
        for key in query_emb:
            sid = query_emb[key]["scene_id"]
            query_buckets.setdefault(sid, []).append(key)

        pool_buckets = {}
        for key, g in query_graphs.items():
            pool_buckets.setdefault(g.scene_id, []).append(key)

        if protocol == "table1":
            results = eval_sampled(
                query_emb, db_emb, query_buckets, pool_buckets,
                out_of=proto.out_of, valid_top_k=proto.valid_top_k,
                eval_iters=proto.eval_iters, eval_iter_count=proto.eval_iter_count,
                w_emb=rcfg.w_emb, w_scene=rcfg.w_scene, w_jac=rcfg.w_jac,
            )
        elif protocol == "table2":
            results = eval_full_pool(
                query_emb, db_emb, query_buckets,
                valid_top_k=proto.valid_top_k,
                eval_iters=proto.eval_iters, eval_iter_count=proto.eval_iter_count,
                w_emb=rcfg.w_emb, w_scene=rcfg.w_scene, w_jac=rcfg.w_jac,
            )

    # Print results
    table_name = {"table1": "Table 1 (10-scene pool)",
                  "table2": "Table 2 (full pool)",
                  "table3": "Table 3 (image-generated)"}
    print(f"\n{'='*60}")
    print(f"RESULTS — {table_name[protocol]}")
    print(f"{'='*60}")
    print(f"Weights: emb={rcfg.w_emb:.2f}, scene={rcfg.w_scene:.2f}, jac={rcfg.w_jac:.2f}")
    for k in sorted(results.keys()):
        mean, std = results[k]
        print(f"  Top-{k}: {mean*100:.2f}% +/- {std*100:.2f}%")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
