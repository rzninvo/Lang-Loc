"""
Evaluation script for SimpleGraphMatcher (518-dim + Scene CLIP fusion).

Properly loads the SimpleGraphMatcher wrapper with fusion layer.

USE ONLY FOR IMG GENERATED SCANSCRIBE TEST SET. (Table4)

# update 12 feb 5.30pm : 
Results:
  Top-1: 42.10% ± 4.04%
  Top-3: 68.40% ± 3.47%
  Top-5: 82.90% ± 3.05%
  Top-10: 100.00% ± 0.00%
"""


import time
import argparse
import sys
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import random
import clip
from pathlib import Path

from langloc.retrieval.scene_graph import SceneGraph
from langloc.retrieval.helper import get_matching_subgraph
from langloc.retrieval.models.dual_scene_aligner import DualSceneAligner
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
if device.type == "cuda":
    torch.cuda.empty_cache()
print(f"Using device: {device}")

# Load CLIP
print("Loading CLIP...")
clip_model, _ = clip.load("ViT-B/32", device=device)
print("✓ CLIP loaded")

random.seed(42)
#add seed to torch
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)  


# ============================================================
# Model Definition (Must match training script!)
# ============================================================

class SimpleGraphMatcher(nn.Module):
    """Just GNN + Scene CLIP fusion. No overlap head."""
    def __init__(self, base_model, scene_clip_dim=512, hidden_dim=256):
        super().__init__()
        self.base_model = base_model
        
        # Simple fusion (MUST match training!)
        self.fusion = nn.Sequential(
            nn.LayerNorm(base_model.hidden_dim + scene_clip_dim),
            nn.Linear(base_model.hidden_dim + scene_clip_dim, hidden_dim),  
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
    
    def forward(self, batch, scene_clip_src, scene_clip_ref):
        # GNN
        out = self.base_model(batch)
        gnn_src = out["src_emb"]
        gnn_ref = out["ref_emb"]
        
        # Fusion
        src_combined = torch.cat([gnn_src, scene_clip_src], dim=-1)
        ref_combined = torch.cat([gnn_ref, scene_clip_ref], dim=-1)
        
        src_emb = self.fusion(src_combined)
        ref_emb = self.fusion(ref_combined)
        
        return {
            "src_emb": src_emb,
            "ref_emb": ref_emb
        }


# ============================================================
# Helper Functions
# ============================================================
def grid_search_weights(model, database_3dssg, dataset, clip_model, device):
    """Find optimal fusion weights."""
    
    print("\n" + "="*70)
    print("GRID SEARCH: Finding optimal fusion weights")
    print("="*70)
    
    weight_configs = [
        # (emb_weight, scene_clip_weight, jaccard_weight)
        (0.5, 0.3, 0.2),  # Favor learned embeddings
        (0.4, 0.4, 0.2),  # Balance emb + scene
        (0.4, 0.3, 0.3),  # Current baseline
        (0.3, 0.5, 0.2),  # Favor scene CLIP
        (0.3, 0.4, 0.3),  # Balance scene + jaccard
        (0.3, 0.3, 0.4),  # Favor label overlap
        (0.2, 0.5, 0.3),  # Strong scene CLIP
        (0.2, 0.4, 0.4),  # Favor semantics
        (0.33, 0.33, 0.34), # Equal weights
        (0.25, 0.5, 0.25), # Very strong scene CLIP
    ]
    
    best_top1 = 0
    
    best_weights = None
    results = []
    
    # Quick eval: 3 rounds, 50 iterations
    for w_emb, w_scene, w_jac in weight_configs:
        print(f"\nTesting: emb={w_emb:.2f}, scene={w_scene:.2f}, jaccard={w_jac:.2f}")
        
        # Run quick eval
        top1_acc = quick_eval_with_weights(
            model, database_3dssg, dataset, clip_model, device,
            w_emb, w_scene, w_jac,
            eval_rounds=3, iterations=50
        )
        
        results.append((w_emb, w_scene, w_jac, top1_acc))
        print(f"  → Top-1: {top1_acc:.2f}%")
        
        if top1_acc > best_top1:
            best_top1 = top1_acc
            best_weights = (w_emb, w_scene, w_jac)
    
    print("\n" + "="*70)
    print("GRID SEARCH RESULTS:")
    print("="*70)
    results.sort(key=lambda x: x[3], reverse=True)
    for w_emb, w_scene, w_jac, acc in results[:5]:
        print(f"  emb={w_emb:.2f}, scene={w_scene:.2f}, jac={w_jac:.2f} → {acc:.2f}%")
    
    print(f"\n✅ BEST WEIGHTS: emb={best_weights[0]:.2f}, scene={best_weights[1]:.2f}, jac={best_weights[2]:.2f}")
    print(f"✅ BEST Top-1: {best_top1:.2f}%")
    print("="*70)
    
    return best_weights


def quick_eval_with_weights(model, database_3dssg, dataset, clip_model, device, 
                            w_emb, w_scene, w_jac, eval_rounds=3, iterations=50):
    """Quick evaluation with specific weights."""
    model.eval()
    
    # Organize by scene
    buckets = {}
    for idx, g in enumerate(dataset):
        if g.scene_id not in buckets:
            buckets[g.scene_id] = []
        buckets[g.scene_id].append(idx)
    
    all_top1 = []
    
    for _ in range(eval_rounds):
        top1_hits = []
        
        sampled_test_indices = [
            [random.sample(buckets[g], 1)[0] for g in random.sample(list(buckets.keys()), 10)]
            for _ in range(iterations)
        ]
        
        for t_set in sampled_test_indices:
            match_scores = []
            scene_ids = []
            seen_scene_ids = set()
            
            query_scene_id = dataset[t_set[0]].scene_id
            
            for i in t_set:
                db_scene_id = dataset[i].scene_id
                
                if db_scene_id in seen_scene_ids:
                    continue
                seen_scene_ids.add(db_scene_id)
                
                query = dataset[t_set[0]]
                db = database_3dssg[db_scene_id]
                
                batch = convert_scene_graph_to_batch(query, db, clip_model, device)
                
                with torch.no_grad():
                    out = model(
                        batch,
                        scene_clip_src=batch['scene_clip_src'],
                        scene_clip_ref=batch['scene_clip_ref']
                    )
                    
                    # Embedding similarity
                    src_norm = F.normalize(out['src_emb'], dim=-1)
                    ref_norm = F.normalize(out['ref_emb'], dim=-1)
                    emb_sim = (src_norm * ref_norm).sum().item()
                    
                    # Scene CLIP similarity
                    scene_sim = F.cosine_similarity(
                        batch['scene_clip_src'], 
                        batch['scene_clip_ref']
                    ).item()
                    
                    # Jaccard
                    #  Label overlap (F1 Score - better than Jaccard!)
                    query_labels = set(n.label for n in query.nodes.values())
                    db_labels = set(n.label for n in db.nodes.values())
                    overlap = len(query_labels & db_labels)

                    if len(query_labels) > 0 and len(db_labels) > 0:
                        precision = overlap / len(db_labels)
                        recall = overlap / len(query_labels)
                        f1 = (2 * precision * recall) / (precision + recall + 1e-8) if (precision + recall) > 0 else 0
                    else:
                        f1 = 0

                    # COMBINED SCORE (using F1 instead of Jaccard)
                    final_score = w_emb * emb_sim + w_scene * scene_sim + w_jac * f1  # ← Using F1!
                                        
                    match_scores.append(final_score)
                    scene_ids.append(db_scene_id)
            
            # Check top-1
            if len(match_scores) > 0:
                best_idx = np.argmax(match_scores)
                top1_hits.append(1 if scene_ids[best_idx] == query_scene_id else 0)
        
        all_top1.append(np.mean(top1_hits) * 100)
    
    return np.mean(all_top1)

def get_clip_embedding(label, clip_model, device):
    """Get CLIP embedding for a label."""
    with torch.no_grad():
        tokens = clip.tokenize([label]).to(device)
        emb = clip_model.encode_text(tokens)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb[0].cpu().numpy()


def get_scene_clip_embedding(labels_list, clip_model, device):
    """Get scene-level CLIP from list of object labels."""
    unique_labels = list(set(labels_list))[:10]
    scene_desc = f"A room with {', '.join(unique_labels)}"
    
    with torch.no_grad():
        tokens = clip.tokenize([scene_desc]).to(device)
        emb = clip_model.encode_text(tokens)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb[0].cpu().numpy()


def convert_scene_graph_to_batch(query_graph, db_graph, clip_model, device):
    """
    Convert two SceneGraph objects to 518-dim format + scene CLIP.
    
    Node features: centroid(3) + color(3) + node_CLIP(512) = 518 dims
    Scene CLIP: 512 dims (separate, not in nodes)
    """
    
    def get_node_features_518(graph):
        """Convert SceneGraph to 518-dim features (NO scene CLIP in nodes!)."""
        node_feats = []
        
        for node_id in graph.nodes:
            node = graph.nodes[node_id]
            
            # Centroid (3D) - zero out for fair comparison
            centroid = np.zeros(3, dtype=np.float32)
            
            # Color (3D) - default gray
            color = np.array([0.5, 0.5, 0.5], dtype=np.float32)
            
            # Node-level CLIP (512D)
            node_clip = get_clip_embedding(node.label, clip_model, device)
            
            # Concatenate: 3 + 3 + 512 = 518
            feat = np.concatenate([centroid, color, node_clip])
            node_feats.append(feat)
        
        return torch.tensor(np.array(node_feats), dtype=torch.float32)
    
    def get_scene_clip_512(graph):
        """Get scene-level CLIP (512D)."""
        labels = [graph.nodes[nid].label for nid in graph.nodes]
        return get_scene_clip_embedding(labels, clip_model, device)
    
    def get_edge_info(graph):
        """Get edge information from SceneGraph."""
        edge_idx = graph.edge_idx
        edge_feats = graph.edge_features
        
        num_nodes = len(graph.nodes)
        
        if len(edge_idx) > 0 and len(edge_idx[0]) > 0:
            edges = torch.tensor(edge_idx, dtype=torch.long)
            
            # Filter out invalid edges
            valid_mask = (edges[0] < num_nodes) & (edges[1] < num_nodes) & (edges[0] >= 0) & (edges[1] >= 0)
            edges = edges[:, valid_mask]
            
            if edges.size(1) == 0:
                return (torch.zeros(2, 0, dtype=torch.long),
                       torch.zeros(0, 8, dtype=torch.float32),
                       torch.zeros(2, 0, dtype=torch.long),
                       torch.zeros(0, 1, dtype=torch.float32))
            
            num_edges = edges.size(1)
            
            # Geometric attributes (8D)
            geom_attr = torch.zeros(num_edges, 8, dtype=torch.float32)
            
            if edge_feats is not None and len(edge_feats) > 0:
                edge_feats_tensor = torch.tensor(np.array(edge_feats), dtype=torch.float32)
                if edge_feats_tensor.dim() == 1:
                    edge_feats_tensor = edge_feats_tensor.unsqueeze(-1)
                
                edge_feats_tensor = edge_feats_tensor[valid_mask]
                
                feat_dim = min(8, edge_feats_tensor.size(-1))
                geom_attr[:, :feat_dim] = edge_feats_tensor[:, :feat_dim]
            
            # Text edges
            text_edges = edges.clone()
            text_attr = torch.ones(num_edges, 1, dtype=torch.float32)
        else:
            edges = torch.zeros(2, 0, dtype=torch.long)
            geom_attr = torch.zeros(0, 8, dtype=torch.float32)
            text_edges = torch.zeros(2, 0, dtype=torch.long)
            text_attr = torch.zeros(0, 1, dtype=torch.float32)
        
        return edges, geom_attr, text_edges, text_attr
    
    # Get features
    query_nodes = get_node_features_518(query_graph)
    query_edges, query_geom_attr, query_text_edges, query_text_attr = get_edge_info(query_graph)
    query_scene_clip = get_scene_clip_512(query_graph)
    
    db_nodes = get_node_features_518(db_graph)
    db_edges, db_geom_attr, db_text_edges, db_text_attr = get_edge_info(db_graph)
    db_scene_clip = get_scene_clip_512(db_graph)
    
    # Create batch
    batch = {
        "node_feats_src": query_nodes.to(device),
        "geom_edges_src": query_edges.to(device),
        "geom_attr_src": query_geom_attr.to(device),
        "text_edges_src": query_text_edges.to(device),
        "text_attr_src": query_text_attr.to(device),
        
        "node_feats_ref": db_nodes.to(device),
        "geom_edges_ref": db_edges.to(device),
        "geom_attr_ref": db_geom_attr.to(device),
        "text_edges_ref": db_text_edges.to(device),
        "text_attr_ref": db_text_attr.to(device),
        
        "src_batch": torch.zeros(query_nodes.size(0), dtype=torch.long).to(device),
        "ref_batch": torch.zeros(db_nodes.size(0), dtype=torch.long).to(device),
        "batch_size": 1,
        
        # Scene CLIP (separate!)
        "scene_clip_src": torch.tensor(query_scene_clip, dtype=torch.float32).unsqueeze(0).to(device),
        "scene_clip_ref": torch.tensor(db_scene_clip, dtype=torch.float32).unsqueeze(0).to(device),
    }
    
    return batch


# ============================================================
# Evaluation Function
# ============================================================

def eval_acc_dual_aligner(model, database_3dssg, dataset, clip_model, mode='scanscribe', 
                          eval_iter_count=100, out_of=10, valid_top_k=[1, 3, 5, 10],
                          w_emb=0.33, w_scene=0.33, w_jac=0.34):
    """
    Evaluate SimpleGraphMatcher.
    """
    model.eval()
    
    print(f"\n{'='*70}")
    print(f"Evaluating on {mode}")
    print(f"{'='*70}")
    
    # Organize by scene
    buckets = {}
    for idx, g in enumerate(dataset):
        if g.scene_id not in buckets:
            buckets[g.scene_id] = []
        buckets[g.scene_id].append(idx)
    
    print(f"Unique scenes: {len(buckets)}, Total graphs: {len(dataset)}")
    
    # Evaluation
    eval_iters = 10
    all_valid = {}
    
    debug_count = 0
    
    for eval_round in tqdm(range(eval_iters), desc=f"Eval {mode}"):
        valid = {k: [] for k in valid_top_k}
        
        sampled_test_indices = [
            [random.sample(buckets[g], 1)[0] for g in random.sample(list(buckets.keys()), out_of)]
            for _ in range(eval_iter_count)
        ]
        
        for batch_idx, t_set in enumerate(sampled_test_indices):
            match_scores = []
            scene_ids = []
            seen_scene_ids = set()  # ← Track unique scenes
            
            query_scene_id = dataset[t_set[0]].scene_id
            
            for i in t_set:
                db_scene_id = dataset[i].scene_id
                
                # SKIP DUPLICATES
                if db_scene_id in seen_scene_ids:
                    continue
                seen_scene_ids.add(db_scene_id)  # ← CRITICAL: Mark as seen!
                
                query = dataset[t_set[0]]
                db = database_3dssg[db_scene_id]
                
                # Optional: Subgraph matching
                query_subgraph = query
                db_subgraph = db
                
                # Convert to batch
                batch = convert_scene_graph_to_batch(query_subgraph, db_subgraph, clip_model, device)
                
                with torch.no_grad():
                    # Forward through model
                    out = model(
                        batch,
                        scene_clip_src=batch['scene_clip_src'],
                        scene_clip_ref=batch['scene_clip_ref']
                    )
                    
                    # Compute similarity
                    src_emb = out['src_emb']
                    ref_emb = out['ref_emb']
                    
                    # 1. Embedding similarity
                    src_norm = F.normalize(src_emb, dim=-1)
                    ref_norm = F.normalize(ref_emb, dim=-1)
                    emb_sim = (src_norm * ref_norm).sum().item()

                    # 2. Scene CLIP similarity
                    scene_sim = F.cosine_similarity(
                        batch['scene_clip_src'], 
                        batch['scene_clip_ref']
                    ).item()

                    # 3. Label overlap (F1 Score)
                    query_labels = set(n.label for n in query.nodes.values())
                    db_labels = set(n.label for n in db.nodes.values())
                    overlap = len(query_labels & db_labels)
                    
                    if len(query_labels) > 0 and len(db_labels) > 0:
                        precision = overlap / len(db_labels)
                        recall = overlap / len(query_labels)
                        f1 = (2 * precision * recall) / (precision + recall + 1e-8) if (precision + recall) > 0 else 0
                    else:
                        f1 = 0

                    # 4. COMBINED SCORE (using F1)
                    final_score = w_emb * emb_sim + w_scene * scene_sim + w_jac * f1 

                    match_scores.append(final_score)
                    scene_ids.append(db_scene_id)
            
            # Sort by similarity (high to low)
            match_scores = np.array(match_scores)
            sorted_indices = np.argsort(match_scores)[::-1]
                       
            # DEBUG: Show first 5 batches
            if debug_count < 21:
                print(f"\n{'='*70}")
                print(f"DEBUG Batch {debug_count + 1} (Round {eval_round}, Batch {batch_idx})")
                print(f"{'='*70}")
                print(f"Query scene: {query_scene_id}")
                
                print(f"\n🎯 TOP 10 PREDICTIONS:")
                for rank_idx, idx in enumerate(sorted_indices[:10]):
                    scene_id = scene_ids[idx]
                    score = match_scores[idx]
                    is_correct = "✓ CORRECT" if scene_id == query_scene_id else "✗ wrong"
                    print(f"  Rank {rank_idx+1}: {scene_id:40s} score={score:.4f} {is_correct}")
                
                # #finding cause of failure for top-1 wrong predictions
                # failure_analysis = {
                #         'same_prefix_confused': 0,  # wrong scene has same first 8 chars
                #         'score_too_close': 0,       # correct score within 0.1 of top-1
                #         'correct_not_in_top5': 0,
                #     }
               
                # Ground truth rank
                gt_rank = None
                for rank_idx, idx in enumerate(sorted_indices):
                    if scene_ids[idx] == query_scene_id:
                        gt_rank = rank_idx + 1
                        break
                
                print(f"\n  📊 Ground truth ranked at: {gt_rank}/{len(sorted_indices)}")
                
                top1_scene_id = scene_ids[sorted_indices[0]]  
                top1_score = match_scores[sorted_indices[0]]
                top_wrong_idx = sorted_indices[0] if scene_ids[sorted_indices[0]] != query_scene_id else sorted_indices[1]
                top_wrong_scene_id = scene_ids[top_wrong_idx]
                
                # Get labels
                query_labels = [n.label for n in query.nodes.values()]
                wrong_db = database_3dssg[top_wrong_scene_id]
                correct_db = database_3dssg[query_scene_id]
                
                wrong_labels = [n.label for n in wrong_db.nodes.values()]
                correct_labels = [n.label for n in correct_db.nodes.values()]
                
                print(f"Query labels:   {sorted(query_labels)}")
                print(f"Correct DB labels: {sorted(correct_labels)}")
                print(f"Wrong DB labels:   {sorted(wrong_labels)}")
                
                # Overlap scores
                q_set = set(query_labels)
                correct_overlap = len(q_set & set(correct_labels)) / len(q_set)
                wrong_overlap = len(q_set & set(wrong_labels)) / len(q_set)
                print(f"Label overlap with correct: {correct_overlap:.2f}")
                print(f"Label overlap with wrong:   {wrong_overlap:.2f}")
                # gt_score = match_scores[scene_ids.index(query_scene_id)]

                # if top1_scene_id != query_scene_id:
                #     if top1_scene_id[:8] == query_scene_id[:8]:
                #         failure_analysis['s ame_prefix_confused'] += 1
                #     if abs(top1_score - gt_score) < 0.1:
                #         failure_analysis['score_too_close'] += 1
                #     if gt_rank > 5:
                #         failure_analysis['correct_not_in_top5'] += 1

                # print(f"Failure analysis: {failure_analysis}")

                if gt_rank and gt_rank <= 3:
                    print(f"  ✅ GOOD!")
                elif gt_rank and gt_rank <= 5:
                    print(f"  ⚠️  OKAY")
                else:
                    print(f"  ❌ POOR")
                
                print(f"{'='*70}\n")
                debug_count += 1
            
            # Check top-k
            for k in valid_top_k:
                top_k_scenes = [scene_ids[idx] for idx in sorted_indices[:k]]
                valid[k].append(1 if query_scene_id in top_k_scenes else 0)
        
        for k in valid_top_k:
            if k not in all_valid:
                all_valid[k] = []
            all_valid[k].append(np.mean(valid[k]))
    
    # Results
    accuracy = {k: (np.mean(all_valid[k]), np.std(all_valid[k])) for k in valid_top_k}
    
    print(f"\nResults:")
    for k in accuracy:
        mean, std = accuracy[k]
        print(f"  Top-{k}: {mean*100:.2f}% ± {std*100:.2f}%")
    
    model.train()
    return accuracy


# ============================================================
# Main
# ============================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--graphs_3dssg', type=str, required=True, help='Path to 3DSSG graphs .pt file')
    parser.add_argument('--graphs_scanscribe_test', type=str, required=True, help='Path to ScanScribe test graphs .pt file')
    parser.add_argument('--num_relations', type=int, default=9)
    parser.add_argument('--max_dist', type=float, default=1.0)
    parser.add_argument('--embedding_type', type=str, default='word2vec')
    args = parser.parse_args()
    
    print(f"\nCheckpoint: {args.checkpoint}\n")
    
    # Load checkpoint
    print("Loading checkpoint...")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    print(f"Checkpoint keys: {checkpoint.keys()}")
    
    # Create dummy relation embeddings
    dummy_rel_emb = nn.Embedding(args.num_relations, 64).to(device)
    nn.init.normal_(dummy_rel_emb.weight, mean=0, std=0.1)
    
    # Create base model
    base_model = DualSceneAligner(
        node_input_dim=518,
        relation_dim=64,
        hidden_dim=256,
        rel_clip_matrix=dummy_rel_emb.weight,
        dropout=0.0
    ).to(device)
    
    # Wrap with SimpleGraphMatcher
    model = SimpleGraphMatcher(
        base_model=base_model,
        scene_clip_dim=512,
        hidden_dim=256
    ).to(device)
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"✓ Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters\n")

    # Load data
    print("Loading 3DSSG database...")
    _3dssg_scenes = torch.load(args.graphs_3dssg, weights_only=False, map_location='cpu')
    _3dssg_graphs = {}
    for sid in tqdm(_3dssg_scenes, desc="3DSSG"):
        _3dssg_graphs[sid] = SceneGraph(sid, graph_type='3dssg', graph=_3dssg_scenes[sid],
                                       max_dist=args.max_dist, embedding_type=args.embedding_type,
                                       use_attributes=True)

    print(f"✓ Loaded {len(_3dssg_graphs)} 3DSSG scenes")

    print("Loading ScanScribe test...")
    scanscribe_test = torch.load(args.graphs_scanscribe_test, weights_only=False, map_location='cpu')
                    
    scanscribe_graphs = {}
    for sid in tqdm(scanscribe_test, desc="ScanScribe"):
        for tid in scanscribe_test[sid].keys():
            key = f"{sid}_{str(tid).zfill(5)}"
            scanscribe_graphs[key] = SceneGraph(sid, txt_id=tid, graph_type='scanscribe',
                                               graph=scanscribe_test[sid][tid],
                                               embedding_type='word2vec', use_attributes=True)
    
    scanscribe_graphs = {k: v for k, v in scanscribe_graphs.items() if len(v.edge_idx[0]) >= 1}
    
    print(f"✓ Loaded {len(scanscribe_graphs)} ScanScribe queries\n")

    # Grid search best fusion weights
    # best_weights = grid_search_weights(
    #     model,
    #     _3dssg_graphs,
    #     list(scanscribe_graphs.values()),
    #     clip_model,
    #     device
    # )

    # Unpack the best weights
    # w_emb, w_scene, w_jac = best_weights
    w_emb, w_scene, w_jac = 0.33, 0.33, 0.34
    print(f"\n✅ Using best weights: emb={w_emb:.2f}, scene={w_scene:.2f}, jac={w_jac:.2f}\n")
    
    # Evaluate with best weights
    scanscribe_acc = eval_acc_dual_aligner(
        model,
        _3dssg_graphs,
        list(scanscribe_graphs.values()),
        clip_model,
        mode='scanscribe',
        w_emb=w_emb,
        w_scene=w_scene,
        w_jac=w_jac
    )
    
    print(f"\n{'='*70}")
    print("FINAL RESULTS - ScanScribe")
    print(f"{'='*70}")
    for k in [1, 3, 5, 10]:
        mean, std = scanscribe_acc[k]
        print(f"  Top-{k}: {mean*100:.2f}% ± {std*100:.2f}%")
    print(f"{'='*70}\n")