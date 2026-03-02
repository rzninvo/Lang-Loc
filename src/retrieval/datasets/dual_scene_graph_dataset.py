"""
Clean 518D Dataset - Uses existing scene graphs correctly!

Your scene graphs already have the right structure:
- scene_clip_emb at root (not per node)
- nodes have centroid + color + clip_text_emb

This dataset:
- Returns 518D node features (no scene CLIP in nodes)
- Returns scene_clip_emb separately for fusion layer
- Supports subgraph augmentation
"""

import os
import json
import torch
import random
import numpy as np
from torch.utils.data import Dataset


def build_node_features(node_dict):
    """
    Build 518D node features.
    
    Returns: centroid(3) + color(3) + node_CLIP(512) = 518 dims
    """
    centroid = np.array(node_dict["centroid"], dtype=np.float32)
    color = np.array(node_dict["mean_color"], dtype=np.float32) / 255.0
    node_clip = np.array(node_dict.get("clip_text_emb", np.zeros(512)), dtype=np.float32)
    
    feat = np.concatenate([centroid, color, node_clip])  # 518D
    return torch.tensor(feat, dtype=torch.float32)


def extract_centroids_and_radii(nodes):
    obj_ids = list(nodes.keys())
    centroids = np.array([nodes[o]["centroid"] for o in obj_ids], dtype=float)
    radii = np.array([nodes[o]["radius"] for o in obj_ids], dtype=float)
    return centroids, radii, obj_ids


def build_geometric_edges_knn(nodes, k=5):
    """Build k-NN geometric edges."""
    centroids, radii, obj_ids = extract_centroids_and_radii(nodes)
    N = len(obj_ids)
    
    if N <= 1:
        return torch.zeros(2, 0, dtype=torch.long), torch.zeros(0, 8, dtype=torch.float32)
    
    # Distance matrix
    dmat = np.linalg.norm(centroids[:, None, :] - centroids[None, :, :], axis=2)
    np.fill_diagonal(dmat, np.inf)
    
    # k-NN
    knn_idx = np.argsort(dmat, axis=1)[:, :min(k, N-1)]
    
    edge_index = []
    edge_attr = []
    
    for i in range(N):
        ci = centroids[i]
        ri = radii[i]
        
        for j in knn_idx[i]:
            cj = centroids[j]
            rj = radii[j]
            
            vec = cj - ci
            dist = float(np.linalg.norm(vec))
            
            # 8D geometric features
            feat = np.array([
                vec[0], vec[1], vec[2],  # direction
                dist,                     # distance
                ri, rj,                   # radii
                0.0, 0.0                  # padding
            ], dtype=np.float32)
            
            edge_index.append([i, j])
            edge_attr.append(feat)
    
    if not edge_index:
        return torch.zeros(2, 0, dtype=torch.long), torch.zeros(0, 8, dtype=torch.float32)
    
    return (
        torch.tensor(edge_index, dtype=torch.long).t(),
        torch.tensor(edge_attr, dtype=torch.float32)
    )


def build_text_edges(relations, rel2id, id_to_idx):
    """Build text edges with relation IDs (learned embeddings)."""
    
    if len(relations) > 1000:
        relations = relations[:500]
    
    edge_index = []
    rel_ids = []
    
    for r in relations:
        subj = str(r.get("subject", ""))
        obj = str(r.get("object", ""))
        rel_name = r.get("relation", "unknown")
        
        s = id_to_idx.get(subj)
        o = id_to_idx.get(obj)
        
        if s is None or o is None:
            continue
        
        rel_id = rel2id.get(rel_name, 0)
        
        edge_index.append([s, o])
        rel_ids.append(rel_id)
    
    if not edge_index:
        return (
            torch.zeros((2, 0), dtype=torch.long),
            torch.zeros((0, 1), dtype=torch.long)
        )
    
    return (
        torch.tensor(edge_index, dtype=torch.long).t(),
        torch.tensor(rel_ids, dtype=torch.long).unsqueeze(-1)
    )


class DualSceneGraphDataset(Dataset):
    """
    Dataset for scene graph matching with scene-level CLIP fusion.
    
    Returns:
    - 518D node features (centroid + color + node_CLIP)
    - scene_clip_emb separately (for fusion after GNN)
    - Supports subgraph augmentation
    """
    
    def __init__(self, dataset_dir, metadata_path, augment_ratio=0.0):
        self.dataset_dir = dataset_dir
        self.augment_ratio = augment_ratio
        
        # Load scene files
        self.scene_files = sorted([
            os.path.join(dataset_dir, f)
            for f in os.listdir(dataset_dir)
            if f.endswith('.json')
        ])
        
        # Build filename to scene_id mapping
        self.file_to_scene_id = {}
        self.scene_id_to_file = {}
        
        for filepath in self.scene_files:
            filename = os.path.basename(filepath)
            # Extract scene ID from filename (assumes format: scene_id.json or scene_id_text_N.json)
            scene_id = filename.replace('.json', '')
            if '_text_' in scene_id:
                scene_id = scene_id.split('_text_')[0]
            
            self.file_to_scene_id[filepath] = scene_id
            
            # Map scene_id to all files
            if scene_id not in self.scene_id_to_file:
                self.scene_id_to_file[scene_id] = []
            self.scene_id_to_file[scene_id].append(filepath)
        
        # Load metadata for room grouping
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Build scene-to-room mapping
        self.scene_to_group = {}
        self.group_to_scenes = {}
        
        for entry in metadata:
            group_id = entry['reference']
            self.scene_to_group[group_id] = group_id
            
            if group_id not in self.group_to_scenes:
                self.group_to_scenes[group_id] = []
            self.group_to_scenes[group_id].append(group_id)
            
            for scan in entry.get('scans', []):
                scan_id = scan['reference']
                self.scene_to_group[scan_id] = group_id
                self.group_to_scenes[group_id].append(scan_id)
        
        # Build relation vocabulary
        self.rel2id = {"unknown": 0}
        rel_idx = 1
        
        for scene_file in self.scene_files[:50]:  # Sample to build vocab
            with open(scene_file) as f:
                data = json.load(f)
                for edge in data.get('edges_text', []):
                    rel = edge.get('relation', 'unknown')
                    if rel not in self.rel2id:
                        self.rel2id[rel] = rel_idx
                        rel_idx += 1
        
        print(f"✓ Loaded {len(self.scene_files)} scenes")
        print(f"✓ {len(self.group_to_scenes)} unique rooms")
        print(f"✓ {len(self.rel2id)} relation types")
    
    def _load_scene_from_data(self, data):
        """Load scene from data dict."""
        nodes = data["nodes"]
        text_relations = data.get("edges_text", [])
        
        node_ids = list(nodes.keys())
        id_to_idx = {str(nid): i for i, nid in enumerate(node_ids)}
        
        # Build 518D node features (no scene CLIP!)
        feats = []
        for nid in node_ids:
            feat = build_node_features(nodes[nid])
            feats.append(feat)
        
        node_feats = torch.stack(feats, dim=0)
        
        # Build edges
        geom_edges, geom_attr = build_geometric_edges_knn(nodes)
        text_edges, text_attr = build_text_edges(text_relations, self.rel2id, id_to_idx)
        
        return node_feats, geom_edges, geom_attr, text_edges, text_attr
    
    def _create_subgraph(self, scene_data):
        """
        Create subgraph with 40-70% of nodes.
        Preserves scene_clip_emb!
        """
        nodes = scene_data['nodes']
        edges = scene_data.get('edges_text', [])
        
        ratio = random.uniform(0.4, 0.7)
        num_nodes = len(nodes)
        num_keep = max(3, int(num_nodes * ratio))
        
        all_node_ids = list(nodes.keys())
        keep_node_ids = set(random.sample(all_node_ids, num_keep))
        
        # Filter nodes
        subgraph_nodes = {nid: nodes[nid] for nid in keep_node_ids}
        
        # Filter edges
        subgraph_edges = []
        for edge in edges:
            if edge['subject'] in keep_node_ids and edge['object'] in keep_node_ids:
                subgraph_edges.append(edge)
        
        return {
            'scene_id': scene_data['scene_id'] + '_subgraph',
            'nodes': subgraph_nodes,
            'edges_text': subgraph_edges,
            'scene_clip_emb': scene_data.get('scene_clip_emb', [0.0] * 512),
            'scene_description': scene_data.get('scene_description', '')
        }
    
    def __getitem__(self, idx):
        """
        Returns scene pair with scene CLIP separate from node features.
        """
        src_path = self.scene_files[idx]
        src_scene_id = self.file_to_scene_id[src_path]
        
        # Find positive pair (same room)
        group_id = self.scene_to_group.get(src_scene_id)
        
        if group_id and group_id in self.group_to_scenes:
            same_room_scenes = self.group_to_scenes[group_id]
            candidates = [s for s in same_room_scenes if s != src_scene_id]
            
            if candidates:
                # Pick random scene from same room
                ref_scene_id = random.choice(candidates)
                
                # Get actual file path for this scene
                if ref_scene_id in self.scene_id_to_file:
                    ref_path = random.choice(self.scene_id_to_file[ref_scene_id])
                else:
                    # Fallback: pick random different scene
                    ref_idx = random.randint(0, len(self.scene_files) - 1)
                    while ref_idx == idx:
                        ref_idx = random.randint(0, len(self.scene_files) - 1)
                    ref_path = self.scene_files[ref_idx]
                    ref_scene_id = self.file_to_scene_id[ref_path]
            else:
                # No candidates: pick random different scene
                ref_idx = random.randint(0, len(self.scene_files) - 1)
                while ref_idx == idx:
                    ref_idx = random.randint(0, len(self.scene_files) - 1)
                ref_path = self.scene_files[ref_idx]
                ref_scene_id = self.file_to_scene_id[ref_path]
        else:
            # No group info: pick random different scene
            ref_idx = random.randint(0, len(self.scene_files) - 1)
            while ref_idx == idx:
                ref_idx = random.randint(0, len(self.scene_files) - 1)
            ref_path = self.scene_files[ref_idx]
            ref_scene_id = self.file_to_scene_id[ref_path]
        
        # Load scene data
        with open(src_path) as f:
            src_data = json.load(f)
        with open(ref_path) as f:
            ref_data = json.load(f)
        
        # Extract scene CLIP (from root, not nodes!)
        src_scene_clip = torch.tensor(
            src_data.get('scene_clip_emb', [0.0] * 512),
            dtype=torch.float32
        )
        ref_scene_clip = torch.tensor(
            ref_data.get('scene_clip_emb', [0.0] * 512),
            dtype=torch.float32
        )
        
        # Subgraph augmentation (50% of time)
        if random.random() < 0.5:
            src_data = self._create_subgraph(src_data)
        if random.random() < 0.5:
            ref_data = self._create_subgraph(ref_data)
        
        # Load features
        src = self._load_scene_from_data(src_data)
        ref = self._load_scene_from_data(ref_data)
        
        # Room labels
        src_group = self.scene_to_group.get(src_scene_id, src_scene_id)
        ref_group = self.scene_to_group.get(ref_scene_id, ref_scene_id)
        
        return {
            "node_feats_src": src[0],  # 518D
            "geom_edges_src": src[1],
            "geom_attr_src": src[2],
            "text_edges_src": src[3],
            "text_attr_src": src[4],
            
            "node_feats_ref": ref[0],  # 518D
            "geom_edges_ref": ref[1],
            "geom_attr_ref": ref[2],
            "text_edges_ref": ref[3],
            "text_attr_ref": ref[4],
            
            "scene_clip_src": src_scene_clip,  # 512D (separate!)
            "scene_clip_ref": ref_scene_clip,  # 512D (separate!)
            "room_id": src_group,
            "is_positive": (src_group == ref_group),
        }
    
    def __len__(self):
        return len(self.scene_files)