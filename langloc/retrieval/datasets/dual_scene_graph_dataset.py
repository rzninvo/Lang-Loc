"""Scene graph dataset for dual-branch retrieval (518D node features).

Loads scene graphs with:
- 518D node features (centroid + color + node_CLIP)
- Scene-level CLIP embedding (separate, for fusion after GNN)
- Relation vocabulary to CLIP embedding matrix for the model
- Subgraph augmentation support
"""

import os
import json
import torch
import random
import numpy as np
from torch.utils.data import Dataset
import clip


def build_rel_clip_matrix(rel2id: dict[str, int], device: str = "cpu", clip_model_name: str = "ViT-B/32") -> torch.Tensor:
    """Builds a CLIP embedding matrix from relation strings.

    Each relation string (e.g. ``"above"``, ``"near"``) is encoded with CLIP.
    The ``"unknown"`` relation (ID 0) gets a zero vector.

    Args:
        rel2id: Mapping from relation string to integer ID.
        device: Device for CLIP inference.
        clip_model_name: CLIP model variant to load.

    Returns:
        Float32 tensor of shape ``(num_relations, 512)`` on CPU.
    """
    import clip

    num_relations = max(rel2id.values()) + 1
    matrix = torch.zeros(num_relations, 512, dtype=torch.float32)

    rel_strings = []
    rel_indices = []
    for rel_str, rel_id in rel2id.items():
        if rel_str == "unknown":
            continue
        rel_strings.append(rel_str)
        rel_indices.append(rel_id)

    if not rel_strings:
        return matrix

    clip_model, _ = clip.load(clip_model_name, device=device)
    with torch.no_grad():
        tokens = clip.tokenize(rel_strings).to(device)
        embs = clip_model.encode_text(tokens)
        embs = embs / embs.norm(dim=-1, keepdim=True)
        embs = embs.cpu().float()

    for i, rel_id in enumerate(rel_indices):
        matrix[rel_id] = embs[i]

    del clip_model
    if device != "cpu" and torch.cuda.is_available():
        torch.cuda.empty_cache()

    return matrix


def build_node_features(node_dict: dict) -> torch.Tensor:
    """Builds 518D node features from a node dictionary.

    Concatenates centroid (3D), normalized color (3D), and node CLIP embedding
    (512D) into a single 518D feature vector.

    Args:
        node_dict: Node dictionary with ``"centroid"``, ``"mean_color"``,
            and optionally ``"clip_text_emb"`` keys.

    Returns:
        Feature tensor of shape ``(518,)``.
    """
    centroid = np.array(node_dict["centroid"], dtype=np.float32)
    color = np.array(node_dict["mean_color"], dtype=np.float32) / 255.0
    node_clip = np.array(node_dict.get("clip_text_emb", np.zeros(512)), dtype=np.float32)
    
    feat = np.concatenate([centroid, color, node_clip])
    return torch.tensor(feat, dtype=torch.float32)


def extract_centroids_and_radii(nodes: dict) -> tuple[np.ndarray, np.ndarray, list]:
    """Extracts centroid positions and radii from a node dictionary.

    Args:
        nodes: Dictionary mapping node IDs to node attribute dicts, each
            containing ``"centroid"`` and ``"radius"`` keys.

    Returns:
        Tuple of (centroids, radii, obj_ids) where centroids and radii are
        arrays and obj_ids is the ordered list of node keys.
    """
    obj_ids = list(nodes.keys())
    centroids = np.array([nodes[o]["centroid"] for o in obj_ids], dtype=float)
    radii = np.array([nodes[o]["radius"] for o in obj_ids], dtype=float)
    return centroids, radii, obj_ids


def build_geometric_edges_knn(nodes: dict, k: int = 5) -> tuple[torch.Tensor, torch.Tensor]:
    """Builds k-NN geometric edges with 8D spatial features.

    For each node, connects to its k nearest neighbors with edge features
    containing direction vector, distance, and radii.

    Args:
        nodes: Dictionary mapping node IDs to node attribute dicts.
        k: Number of nearest neighbors per node.

    Returns:
        Tuple of (edge_index, edge_attr) where edge_index has shape
        ``(2, num_edges)`` and edge_attr has shape ``(num_edges, 8)``.
    """
    centroids, radii, obj_ids = extract_centroids_and_radii(nodes)
    N = len(obj_ids)

    if N <= 1:
        return torch.zeros(2, 0, dtype=torch.long), torch.zeros(0, 8, dtype=torch.float32)

    dmat = np.linalg.norm(centroids[:, None, :] - centroids[None, :, :], axis=2)
    np.fill_diagonal(dmat, np.inf)

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

            feat = np.array([
                vec[0], vec[1], vec[2],
                dist,
                ri, rj,
                0.0, 0.0
            ], dtype=np.float32)

            edge_index.append([i, j])
            edge_attr.append(feat)

    if not edge_index:
        return torch.zeros(2, 0, dtype=torch.long), torch.zeros(0, 8, dtype=torch.float32)

    return (
        torch.tensor(edge_index, dtype=torch.long).t(),
        torch.tensor(edge_attr, dtype=torch.float32)
    )


def build_text_edges(
    relations: list[dict],
    rel2id: dict[str, int],
    id_to_idx: dict[str, int],
    rel_clip_cache: dict[str, np.ndarray] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Builds text relation edges with 512D CLIP embeddings.

    Args:
        relations: List of relation dicts with ``"subject"``, ``"object"``,
            and ``"relation"`` keys.
        rel2id: Relation string to integer ID mapping.
        id_to_idx: Node ID string to contiguous index mapping.
        rel_clip_cache: Optional pre-computed CLIP embeddings per relation string.

    Returns:
        Tuple of (edge_index, text_attr) where edge_index has shape
        ``(2, num_edges)`` and text_attr has shape ``(num_edges, 512)``.
    """
    # Truncate excessively large relation lists to prevent memory issues
    if len(relations) > 1000:
        relations = relations[:500]

    edge_index = []
    rel_embs = []

    for r in relations:
        subj = str(r.get("subject", ""))
        obj = str(r.get("object", ""))
        rel_name = r.get("relation", "unknown")

        s = id_to_idx.get(subj)
        o = id_to_idx.get(obj)

        if s is None or o is None:
            continue

        if rel_clip_cache is not None and rel_name in rel_clip_cache:
            rel_emb = rel_clip_cache[rel_name]
        else:
            rel_emb = np.zeros(512, dtype=np.float32)

        edge_index.append([s, o])
        rel_embs.append(rel_emb)

    if not edge_index:
        return (
            torch.zeros((2, 0), dtype=torch.long),
            torch.zeros((0, 512), dtype=torch.float32)
        )

    return (
        torch.tensor(edge_index, dtype=torch.long).t(),
        torch.tensor(np.array(rel_embs), dtype=torch.float32)
    )


class DualSceneGraphDataset(Dataset):
    """Dataset for scene graph matching with scene-level CLIP fusion.

    Yields pairs of scene graphs (positive or negative) with 518D node
    features and separate scene-level CLIP embeddings. Supports subgraph
    augmentation during training.

    Args:
        dataset_dir: Directory containing per-scene JSON files.
        metadata_path: Path to metadata JSON with room grouping information.
        augment_ratio: Ratio of samples to augment (currently unused).
        negative_ratio: Probability of returning a negative pair.
        clip_model: Pre-loaded CLIP model for relation embedding, or None.
        device: Device string for CLIP inference.
    """

    def __init__(
        self,
        dataset_dir: str,
        metadata_path: str,
        augment_ratio: float = 0.0,
        negative_ratio: float = 0.5,
        clip_model: torch.nn.Module | None = None,
        device: str = 'cpu',
        clip_model_name: str = "ViT-B/32",
    ) -> None:
        self.dataset_dir = dataset_dir
        self.augment_ratio = augment_ratio
        self.negative_ratio = negative_ratio
        self.device = device
        self.clip_model_name = clip_model_name
        self.clip_model = clip_model
        
        self.scene_files = sorted([
            os.path.join(dataset_dir, f)
            for f in os.listdir(dataset_dir)
            if f.endswith('.json')
        ])
        
        self.file_to_scene_id = {}
        self.scene_id_to_file = {}

        for filepath in self.scene_files:
            filename = os.path.basename(filepath)
            scene_id = filename.replace('.json', '')
            if '_text_' in scene_id:
                scene_id = scene_id.split('_text_')[0]
            
            self.file_to_scene_id[filepath] = scene_id

            if scene_id not in self.scene_id_to_file:
                self.scene_id_to_file[scene_id] = []
            self.scene_id_to_file[scene_id].append(filepath)
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
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
        
        self.rel2id = {"unknown": 0}
        rel_idx = 1

        # Sample first 50 files to build the relation vocabulary
        for scene_file in self.scene_files[:50]:
            with open(scene_file) as f:
                data = json.load(f)
                for edge in data.get('edges_text', []):
                    rel = edge.get('relation', 'unknown')
                    if rel not in self.rel2id:
                        self.rel2id[rel] = rel_idx
                        rel_idx += 1
        
        self.rel_clip_matrix = build_rel_clip_matrix(self.rel2id, clip_model_name=self.clip_model_name)

        self.rel_clip_cache = {}
        if self.clip_model is not None:
            for rel_name in self.rel2id.keys():
                self.rel_clip_cache[rel_name] = self._get_clip_embedding(rel_name)
        else:
            for rel_name in self.rel2id.keys():
                self.rel_clip_cache[rel_name] = np.zeros(512, dtype=np.float32)

        print(f"✓ Loaded {len(self.scene_files)} scenes")
        print(f"✓ {len(self.group_to_scenes)} unique rooms")
        print(f"✓ {len(self.rel2id)} relation types")
        print(f"✓ Built rel_clip_matrix: {self.rel_clip_matrix.shape}")
        print(f"✓ Cached {len(self.rel_clip_cache)} relation CLIP embeddings")
    
    def _get_clip_embedding(self, text: str) -> np.ndarray:
        """Computes a CLIP embedding for a relation text string.

        Args:
            text: Relation string to embed.

        Returns:
            L2-normalized CLIP embedding as a float32 array of shape ``(512,)``.
        """
        if self.clip_model is None:
            return np.zeros(512, dtype=np.float32)

        with torch.no_grad():
            tokens = clip.tokenize([text]).to(self.device)
            emb = self.clip_model.encode_text(tokens)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb[0].cpu().numpy().astype(np.float32)

    def _load_scene_from_data(self, data: dict) -> tuple[torch.Tensor, ...]:
        """Converts a scene data dict to tensors for node features and edges.

        Args:
            data: Scene dictionary with ``"nodes"`` and optionally ``"edges_text"``.

        Returns:
            Tuple of (node_feats, geom_edges, geom_attr, text_edges, text_attr).
        """
        nodes = data["nodes"]
        text_relations = data.get("edges_text", [])

        node_ids = list(nodes.keys())
        id_to_idx = {str(nid): i for i, nid in enumerate(node_ids)}

        feats = []
        for nid in node_ids:
            feat = build_node_features(nodes[nid])
            feats.append(feat)

        node_feats = torch.stack(feats, dim=0)

        geom_edges, geom_attr = build_geometric_edges_knn(nodes)
        text_edges, text_attr = build_text_edges(text_relations, self.rel2id, id_to_idx, self.rel_clip_cache)

        return node_feats, geom_edges, geom_attr, text_edges, text_attr
    
    def _create_subgraph(self, scene_data: dict) -> dict:
        """Creates a random subgraph by keeping 40--70% of nodes.

        Preserves the scene-level CLIP embedding unchanged.

        Args:
            scene_data: Full scene dictionary with ``"nodes"`` and ``"edges_text"``.

        Returns:
            New scene dictionary with a subset of nodes and filtered edges.
        """
        nodes = scene_data['nodes']
        edges = scene_data.get('edges_text', [])
        
        ratio = random.uniform(0.4, 0.7)
        num_nodes = len(nodes)
        num_keep = max(3, int(num_nodes * ratio))
        
        all_node_ids = list(nodes.keys())
        keep_node_ids = set(random.sample(all_node_ids, num_keep))
        
        subgraph_nodes = {nid: nodes[nid] for nid in keep_node_ids}

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
    
    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str | bool]:
        """Returns a scene pair with scene CLIP separate from node features.

        Args:
            idx: Index into the scene file list.

        Returns:
            Dictionary with source/reference node features, edges, scene CLIP
            embeddings, room ID, and a positive-pair flag.
        """
        src_path = self.scene_files[idx]
        src_scene_id = self.file_to_scene_id[src_path]
        
        group_id = self.scene_to_group.get(src_scene_id)
        
        if group_id and group_id in self.group_to_scenes:
            same_room_scenes = self.group_to_scenes[group_id]
            candidates = [s for s in same_room_scenes if s != src_scene_id]
            
            if candidates:
                ref_scene_id = random.choice(candidates)

                if ref_scene_id in self.scene_id_to_file:
                    ref_path = random.choice(self.scene_id_to_file[ref_scene_id])
                else:
                    ref_idx = random.randint(0, len(self.scene_files) - 1)
                    while ref_idx == idx:
                        ref_idx = random.randint(0, len(self.scene_files) - 1)
                    ref_path = self.scene_files[ref_idx]
                    ref_scene_id = self.file_to_scene_id[ref_path]
            else:
                ref_idx = random.randint(0, len(self.scene_files) - 1)
                while ref_idx == idx:
                    ref_idx = random.randint(0, len(self.scene_files) - 1)
                ref_path = self.scene_files[ref_idx]
                ref_scene_id = self.file_to_scene_id[ref_path]
        else:
            ref_idx = random.randint(0, len(self.scene_files) - 1)
            while ref_idx == idx:
                ref_idx = random.randint(0, len(self.scene_files) - 1)
            ref_path = self.scene_files[ref_idx]
            ref_scene_id = self.file_to_scene_id[ref_path]
        
        with open(src_path) as f:
            src_data = json.load(f)
        with open(ref_path) as f:
            ref_data = json.load(f)
        
        # Scene CLIP is stored at the root level, not per-node
        src_scene_clip = torch.tensor(
            src_data.get('scene_clip_emb', [0.0] * 512),
            dtype=torch.float32
        )
        ref_scene_clip = torch.tensor(
            ref_data.get('scene_clip_emb', [0.0] * 512),
            dtype=torch.float32
        )
        
        if random.random() < 0.5:
            src_data = self._create_subgraph(src_data)
        if random.random() < 0.5:
            ref_data = self._create_subgraph(ref_data)
        
        src = self._load_scene_from_data(src_data)
        ref = self._load_scene_from_data(ref_data)

        src_group = self.scene_to_group.get(src_scene_id, src_scene_id)
        ref_group = self.scene_to_group.get(ref_scene_id, ref_scene_id)
        
        return {
            "node_feats_src": src[0],
            "geom_edges_src": src[1],
            "geom_attr_src": src[2],
            "text_edges_src": src[3],
            "text_attr_src": src[4],

            "node_feats_ref": ref[0],
            "geom_edges_ref": ref[1],
            "geom_attr_ref": ref[2],
            "text_edges_ref": ref[3],
            "text_attr_ref": ref[4],

            "scene_clip_src": src_scene_clip,
            "scene_clip_ref": ref_scene_clip,
            "room_id": src_group,
            "is_positive": (src_group == ref_group),
        }
    
    def __len__(self) -> int:
        """Returns the number of scene files in the dataset."""
        return len(self.scene_files)