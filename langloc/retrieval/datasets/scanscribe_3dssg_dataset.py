"""
Dataset for fine-tuning: ScanScribe text graphs paired with 3DSSG scene graphs.

Source: sparse ScanScribe text graph (from .pt file)
Reference: dense 3DSSG scene graph (from JSON files)
Positive pair: same scene ID, Negative pair: different scene ID
"""

import os
import json
import torch
import random
import numpy as np
from torch.utils.data import Dataset
import clip
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
clip_model, _ = clip.load("ViT-B/32", device=device)

def get_clip_embedding(label):
    with torch.no_grad():
        tokens = clip.tokenize([label]).to(device)
        emb = clip_model.encode_text(tokens)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb[0].cpu().numpy()

def get_scene_clip_embedding(labels_list):
    unique_labels = list(set(labels_list))[:10]
    scene_desc = f"A room with {', '.join(unique_labels)}"
    with torch.no_grad():
        tokens = clip.tokenize([scene_desc]).to(device)
        emb = clip_model.encode_text(tokens)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb[0].cpu().numpy()


class ScanScribe3DSSGDataset(Dataset):
    """
    Pairs ScanScribe text graphs with 3DSSG scene graphs.
    50% positive pairs (same scene), 50% negative pairs (different scene).
    """
    
    def __init__(self, scanscribe_pt_path, dssg_json_dir, negative_ratio=0.5):
        self.negative_ratio = negative_ratio
        self.dssg_json_dir = dssg_json_dir
        
        # Load ScanScribe text graphs
        print("Loading ScanScribe text graphs...")
        scanscribe_data = torch.load(scanscribe_pt_path, 
                                     weights_only=False, map_location='cpu')
        
        # Build list of (scene_id, txt_id, graph) tuples
        self.samples = []
        for scene_id in scanscribe_data:
            for txt_id in scanscribe_data[scene_id]:
                graph = scanscribe_data[scene_id][txt_id]
                # Only use graphs with at least 1 edge
                if hasattr(graph, 'edge_idx') and len(graph.edge_idx[0]) >= 1:
                    self.samples.append((scene_id, txt_id, graph))
                elif isinstance(graph, dict):
                    edges = graph.get('edges', [[], []])
                    if len(edges) >= 2 and len(edges[0]) >= 1:
                        self.samples.append((scene_id, txt_id, graph))
        
        # Get all scene IDs
        self.scene_ids = list(scanscribe_data.keys())
        
        # Build scene_id → sample indices mapping
        self.scene_to_samples = {}
        for idx, (scene_id, txt_id, graph) in enumerate(self.samples):
            if scene_id not in self.scene_to_samples:
                self.scene_to_samples[scene_id] = []
            self.scene_to_samples[scene_id].append(idx)
        
        # Build relation vocabulary from 3DSSG JSONs
        self.rel2id = {"unknown": 0}
        rel_idx = 1
        json_files = [f for f in os.listdir(dssg_json_dir) if f.endswith('.json')][:50]
        for jf in json_files:
            with open(os.path.join(dssg_json_dir, jf)) as f:
                data = json.load(f)
            for edge in data.get('edges_text', []):
                rel = edge.get('relation', 'unknown')
                if rel not in self.rel2id:
                    self.rel2id[rel] = rel_idx
                    rel_idx += 1
        
        # Also collect relation strings from ScanScribe graphs for CLIP cache
        all_relations = set(self.rel2id.keys())
        for scene_id, txt_id, graph in self.samples:
            if hasattr(graph, 'edge_relations') and graph.edge_relations:
                for r in graph.edge_relations:
                    all_relations.add(str(r).lower())

        # Build CLIP cache for all relation strings (512D per relation)
        print(f"Building CLIP relation cache for {len(all_relations)} relations...")
        self.rel_clip_cache = {}
        rel_strings = [r for r in all_relations if r != "unknown"]
        if rel_strings:
            with torch.no_grad():
                # Batch encode in chunks
                for i in range(0, len(rel_strings), 64):
                    chunk = rel_strings[i:i+64]
                    tokens = clip.tokenize(chunk).to(device)
                    embs = clip_model.encode_text(tokens)
                    embs = embs / embs.norm(dim=-1, keepdim=True)
                    for j, rel in enumerate(chunk):
                        self.rel_clip_cache[rel] = embs[j].cpu().numpy().astype(np.float32)
        self.rel_clip_cache["unknown"] = np.zeros(512, dtype=np.float32)

        print(f"  {len(self.samples)} ScanScribe text graphs")
        print(f"  {len(self.scene_ids)} unique scenes")
        print(f"  {len(self.rel2id)} relation types, {len(self.rel_clip_cache)} CLIP-cached")

        print("Precomputing CLIP embeddings for all ScanScribe text graphs...")
        self.clip_cache = {}
        for scene_id, txt_id, graph in tqdm(self.samples):
            if hasattr(graph, 'nodes'):
                for nid, node in graph.nodes.items():
                    label = node.label if hasattr(node, 'label') else str(node)
                    if label not in self.clip_cache:
                        self.clip_cache[label] = get_clip_embedding(label)
        print(f"  Cached {len(self.clip_cache)} unique CLIP embeddings")
        print("Precomputing scene CLIP embeddings...")
        self.scene_clip_cache = {}
        for scene_id, txt_id, graph in tqdm(self.samples):
            cache_key = f"{scene_id}_{txt_id}"
            if cache_key not in self.scene_clip_cache and hasattr(graph, 'nodes'):
                labels = [graph.nodes[nid].label if hasattr(graph.nodes[nid], 'label') 
                        else str(graph.nodes[nid]) for nid in graph.nodes]
                self.scene_clip_cache[cache_key] = get_scene_clip_embedding(labels)
        print(f"  Cached {len(self.scene_clip_cache)} scene CLIP embeddings")
        
        # Pre-load all 3DSSG JSONs into memory (much faster than disk I/O per-batch)
        print("Pre-loading 3DSSG JSON files...")
        self.dssg_cache = {}
        for filename in os.listdir(dssg_json_dir):
            if filename.endswith('.json'):
                scene_id = filename.replace('.json', '')
                json_path = os.path.join(dssg_json_dir, filename)
                try:
                    with open(json_path) as f:
                        self.dssg_cache[scene_id] = json.load(f)
                except:
                    pass
        print(f"  Pre-loaded {len(self.dssg_cache)} 3DSSG scenes into memory")
        
        # Pre-compute all features to avoid expensive computation during training
        print("Pre-computing all ScanScribe features...")
        self.scanscribe_features_cache = {}
        for idx, (scene_id, txt_id, graph) in enumerate(tqdm(self.samples)):
            cache_key = f"{scene_id}_{txt_id}"
            feats = self._text_graph_to_features(graph, scene_id, txt_id)
            if feats is not None:
                self.scanscribe_features_cache[cache_key] = feats
        print(f"  Pre-computed {len(self.scanscribe_features_cache)} ScanScribe features")
        
        print("Pre-computing all 3DSSG features...")
        self.dssg_features_cache = {}
        for scene_id in tqdm(self.dssg_cache.keys()):
            feats = self._dssg_json_to_features(self.dssg_cache[scene_id])
            self.dssg_features_cache[scene_id] = feats
        print(f"  Pre-computed {len(self.dssg_features_cache)} 3DSSG features")
            
    def _load_dssg_json(self, scene_id):
        """Load 3DSSG scene graph from cache (already in memory)."""
        return self.dssg_cache.get(scene_id, None)
    
    def _text_graph_to_features(self, graph, scene_id=None, txt_id=None):
        """Convert ScanScribe SceneGraph object to 518D features."""
        # Get node labels
        if hasattr(graph, 'nodes'):
            nodes = graph.nodes
            node_ids = list(nodes.keys())
        else:
            return None
        
        node_feats = []
        labels = []
        for nid in node_ids:
            node = nodes[nid]
            label = node.label if hasattr(node, 'label') else str(node)
            labels.append(label)
            
            # 518D: zeros(3) + zeros(3) + CLIP(512)
            centroid = np.zeros(3, dtype=np.float32)
            color = np.array([0.5, 0.5, 0.5], dtype=np.float32)
            node_clip = self.clip_cache.get(label, get_clip_embedding(label))

            feat = np.concatenate([centroid, color, node_clip])
            node_feats.append(feat)
        
        node_feats = torch.tensor(np.array(node_feats), dtype=torch.float32)
        
        # Scene CLIP
        cache_key = f"{scene_id}_{txt_id}"
        scene_clip_np = self.scene_clip_cache.get(cache_key, get_scene_clip_embedding(labels))
        scene_clip = torch.tensor(scene_clip_np, dtype=torch.float32)
            
        # Edges from text graph
        if hasattr(graph, 'edge_idx') and len(graph.edge_idx[0]) > 0:
            edges = torch.tensor(graph.edge_idx, dtype=torch.long)
            num_nodes = len(node_ids)
            valid = (edges[0] < num_nodes) & (edges[1] < num_nodes)
            edges = edges[:, valid]
            valid_indices = valid.nonzero(as_tuple=True)[0].tolist()
            num_edges = edges.size(1)
            geom_attr = torch.zeros(num_edges, 8, dtype=torch.float32)

            # 512D CLIP text_attr from edge relations
            rel_embs = []
            has_rels = hasattr(graph, 'edge_relations') and graph.edge_relations
            for idx in valid_indices:
                if has_rels and idx < len(graph.edge_relations):
                    rel_str = str(graph.edge_relations[idx]).lower()
                    emb = self.rel_clip_cache.get(rel_str, np.zeros(512, dtype=np.float32))
                else:
                    emb = np.zeros(512, dtype=np.float32)
                rel_embs.append(emb)

            if rel_embs:
                text_attr = torch.tensor(np.array(rel_embs), dtype=torch.float32)
            else:
                text_attr = torch.zeros(num_edges, 512, dtype=torch.float32)
        else:
            edges = torch.zeros(2, 0, dtype=torch.long)
            geom_attr = torch.zeros(0, 8, dtype=torch.float32)
            text_attr = torch.zeros(0, 512, dtype=torch.float32)
        
        return node_feats, edges, geom_attr, edges, text_attr, scene_clip
    
    def _dssg_json_to_features(self, data):
        """Convert 3DSSG JSON to 518D features."""
        nodes = data['nodes']
        node_ids = list(nodes.keys())
        id_to_idx = {str(nid): i for i, nid in enumerate(node_ids)}
        
        node_feats = []
        for nid in node_ids:
            node = nodes[nid]
            centroid = np.array(node.get('centroid', [0, 0, 0]), dtype=np.float32)
            color = np.array(node.get('mean_color', [128, 128, 128]), dtype=np.float32) / 255.0
            node_clip = np.array(node.get('clip_text_emb', 
                                  get_clip_embedding(node.get('label', 'object'))), 
                                  dtype=np.float32)
            feat = np.concatenate([centroid, color, node_clip])
            node_feats.append(feat)
        
        node_feats = torch.tensor(np.array(node_feats), dtype=torch.float32)
        
        # Scene CLIP from JSON
        scene_clip = torch.tensor(
            data.get('scene_clip_emb', [0.0] * 512), dtype=torch.float32
        )
        
        # Edges with 512D CLIP text_attr
        edges_text = data.get('edges_text', [])
        edge_index = []
        rel_embs = []
        for e in edges_text:
            s = id_to_idx.get(str(e['subject']))
            o = id_to_idx.get(str(e['object']))
            if s is not None and o is not None:
                edge_index.append([s, o])
                rel_str = e.get('relation', 'unknown').lower()
                emb = self.rel_clip_cache.get(rel_str, np.zeros(512, dtype=np.float32))
                rel_embs.append(emb)

        if edge_index:
            edges = torch.tensor(edge_index, dtype=torch.long).t()
            geom_attr = torch.zeros(len(edge_index), 8, dtype=torch.float32)
            text_attr = torch.tensor(np.array(rel_embs), dtype=torch.float32)
        else:
            edges = torch.zeros(2, 0, dtype=torch.long)
            geom_attr = torch.zeros(0, 8, dtype=torch.float32)
            text_attr = torch.zeros(0, 512, dtype=torch.float32)
        
        return node_feats, edges, geom_attr, edges, text_attr, scene_clip
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        src_scene_id, src_txt_id, src_graph = self.samples[idx]
        
        # Get pre-computed features instead of computing on-the-fly
        src_cache_key = f"{src_scene_id}_{src_txt_id}"
        src = self.scanscribe_features_cache.get(src_cache_key)
        
        if src is None:
            # Fallback (shouldn't happen if cache is complete)
            src = self._text_graph_to_features(src_graph, src_scene_id, src_txt_id)
        
        # Decide positive or negative
        is_negative = random.random() < self.negative_ratio
        
        if is_negative:
            # Pick a different scene
            other_scenes = [s for s in self.scene_ids if s != src_scene_id]
            ref_scene_id = random.choice(other_scenes)
        else:
            ref_scene_id = src_scene_id
        
        # Get pre-computed features for reference
        ref = self.dssg_features_cache.get(ref_scene_id)
        
        if ref is None:
            # Try src scene instead
            ref_scene_id = src_scene_id
            ref = self.dssg_features_cache.get(ref_scene_id)
        
        if ref is None:
            # Last resort: find any valid scene
            for sid in self.scene_ids:
                ref = self.dssg_features_cache.get(sid)
                if ref is not None:
                    ref_scene_id = sid
                    break
        
        if src is None:
            # Fallback: return same as ref for both
            src = ref
        
        return {
            "node_feats_src": src[0],
            "geom_edges_src": src[1],
            "geom_attr_src": src[2],
            "text_edges_src": src[3],
            "text_attr_src": src[4],
            "scene_clip_src": src[5],
            
            "node_feats_ref": ref[0],
            "geom_edges_ref": ref[1],
            "geom_attr_ref": ref[2],
            "text_edges_ref": ref[3],
            "text_attr_ref": ref[4],
            "scene_clip_ref": ref[5],
            
            "is_positive": (src_scene_id == ref_scene_id),
            "src_scene_id": src_scene_id,
            "ref_scene_id": ref_scene_id,
        }