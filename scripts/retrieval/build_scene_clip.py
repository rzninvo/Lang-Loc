"""
Scene Graph Builder with Scene-Level CLIP Embeddings
Processes ALL scenes in the dataset directory (not just 100)

Features:
- Unique spatially-aware node labels
- Per-node CLIP embeddings
- Scene-level CLIP embedding (NEW!)
- Relationship-aware scene descriptions
"""

import json
import numpy as np
from plyfile import PlyData
import argparse
import clip
import torch
import os
from pathlib import Path
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Loading CLIP model on {device}...")
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
print("✓ CLIP loaded\n")

# Thresholds
DIR_THRESHOLD = 0.20
HEIGHT_THRESHOLD = 0.15
NEAR_THRESHOLD = 0.80


# ============================================================
# Spatial Descriptors
# ============================================================

def get_spatial_descriptor(centroid, bbox_size, scene_bounds):
    """Create spatial descriptor: e.g., 'north_upper', 'south_lower'"""
    x, y, z = centroid
    min_x, max_x, min_y, max_y, min_z, max_z = scene_bounds
    
    # Horizontal position
    x_range = max_x - min_x
    z_range = max_z - min_z
    
    x_norm = (x - min_x) / x_range if x_range > 0 else 0.5
    z_norm = (z - min_z) / z_range if z_range > 0 else 0.5
    
    if abs(x_norm - 0.5) > 0.3:
        horizontal = "east" if x_norm > 0.5 else "west"
    elif abs(z_norm - 0.5) > 0.3:
        horizontal = "north" if z_norm > 0.5 else "south"
    else:
        horizontal = "center"
    
    # Vertical position
    y_range = max_y - min_y
    y_norm = (y - min_y) / y_range if y_range > 0 else 0.5
    
    if y_norm > 0.66:
        vertical = "upper"
    elif y_norm < 0.33:
        vertical = "lower"
    else:
        vertical = "middle"
    
    return f"{horizontal}_{vertical}"


def get_size_descriptor(bbox_size):
    """Classify object by size."""
    max_dim = max(bbox_size)
    if max_dim > 2.0:
        return "large"
    elif max_dim > 0.8:
        return "medium"
    else:
        return "small"


def get_color_descriptor(mean_color):
    """Get dominant color name."""
    r, g, b = mean_color
    
    # Grayscale
    if max(r, g, b) - min(r, g, b) < 30:
        if sum([r, g, b]) / 3 < 80:
            return "dark"
        elif sum([r, g, b]) / 3 > 180:
            return "white"
        else:
            return "gray"
    
    # Colors
    if r > g and r > b:
        if r > 180:
            return "red"
        else:
            return "brown"
    elif g > r and g > b:
        return "green"
    elif b > r and b > g:
        return "blue"
    else:
        return "mixed"


def create_unique_label(base_label, centroid, bbox_size, mean_color, scene_bounds, label_counts, node_id):
    """Create unique label with spatial and node ID info"""
    
    if base_label.startswith("obj") or base_label == "object" or base_label == "unknown":
        size = get_size_descriptor(bbox_size)
        color = get_color_descriptor(mean_color)
        spatial = get_spatial_descriptor(centroid, bbox_size, scene_bounds)
        unique_label = f"{size}_{color}_object_{node_id}_{spatial}"
    else:
        spatial = get_spatial_descriptor(centroid, bbox_size, scene_bounds)
        base_with_spatial = f"{base_label}_{spatial}"
        
        if base_with_spatial in label_counts:
            count = label_counts[base_with_spatial]
            label_counts[base_with_spatial] += 1
            unique_label = f"{base_with_spatial}_{count}"
        else:
            label_counts[base_with_spatial] = 1
            unique_label = base_with_spatial
    
    return unique_label


# ============================================================
# PLY and Semseg Loading
# ============================================================

def load_ply_instances(ply_path):
    """Load point cloud and group by instance ID"""
    ply = PlyData.read(ply_path)
    v = ply["vertex"].data

    xyz = np.vstack([v["x"], v["y"], v["z"]]).T
    rgb = np.vstack([v["red"], v["green"], v["blue"]]).T
    
    # The labels.instances.annotated.v2.ply uses objectId as instance labels
    # These are the ACTUAL object instances we want
    inst = v["objectId"]

    instance_ids = np.unique(inst)
    
    # Filter out background (0)
    instance_ids = instance_ids[instance_ids > 0]

    instance_points = {}
    for iid in instance_ids:
        mask = inst == iid
        pts = xyz[mask]
        colors = rgb[mask]
        instance_points[str(int(iid))] = (pts, colors)

    return instance_points, xyz


def load_semseg(semseg_path):
    """
    Load semantic segmentation labels.
    Returns mapping from objectId (instance) to label.
    """
    with open(semseg_path, "r") as f:
        data = json.load(f)

    # Map objectId to label directly
    # The objectId in semseg corresponds to the objectId in the PLY file
    id_to_label = {}
    for group in data["segGroups"]:
        object_id = str(group["objectId"])
        label = group["label"].lower()
        id_to_label[object_id] = label
    
    return id_to_label


def compute_node_attributes(points, colors):
    """Compute centroid, color, radius"""
    centroid = points.mean(axis=0)
    d = np.linalg.norm(points - centroid, axis=1)
    radius = float(min(d.max(), 0.40))
    mean_color = colors.mean(axis=0).tolist()
    return centroid.tolist(), mean_color, radius


def compute_geometric_features(points):
    """Compute geometric features for the object"""
    centered = points - points.mean(axis=0)
    cov = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    eigenvalues = np.maximum(eigenvalues, 1e-8)
    eigenvalues = np.sort(eigenvalues)[::-1]
    
    l1, l2, l3 = eigenvalues
    linearity = (l1 - l2) / l1 if l1 > 0 else 0
    planarity = (l2 - l3) / l1 if l1 > 0 else 0
    sphericity = l3 / l1 if l1 > 0 else 0
    
    std_dev = np.sqrt(eigenvalues).tolist()
    extent = (2 * np.sqrt(eigenvalues)).tolist()
    
    bbox_min = points.min(axis=0)
    bbox_max = points.max(axis=0)
    bbox_size = (bbox_max - bbox_min).tolist()
    
    return {
        "std_dev": std_dev,
        "std_color": [0, 0, 0],
        "extent": extent,
        "linearity": float(linearity),
        "planarity": float(planarity),
        "sphericity": float(sphericity),
        "bbox_size": bbox_size,
        "num_points": int(len(points))
    }


# ============================================================
# Relations
# ============================================================

def directional_rel(ci, cj):
    """Compute directional relationships"""
    dx, dy, dz = cj - ci
    out = []
    if abs(dx) > DIR_THRESHOLD:
        out.append("right_of" if dx > 0 else "left_of")
    if abs(dz) > DIR_THRESHOLD:
        out.append("in_front_of" if dz > 0 else "behind")
    if abs(dy) > HEIGHT_THRESHOLD:
        out.append("above" if dy > 0 else "below")
    return out


def distance_rel(ci, cj, ri, rj):
    """Compute distance relationships"""
    d = np.linalg.norm(cj - ci)
    if d <= (ri + rj) * 0.65:
        return ["touching"]
    if d < NEAR_THRESHOLD:
        return ["near"]
    return []


def symmetric_rel(sub, obj, rel):
    """Get symmetric relationship"""
    inverses = {
        "left_of": "right_of",
        "right_of": "left_of",
        "in_front_of": "behind",
        "behind": "in_front_of",
        "above": "below",
        "below": "above",
        "near": "near"
    }
    if rel in inverses:
        return obj, sub, inverses[rel]
    return None


# ============================================================
# Scene-Level CLIP Embedding (NEW!)
# ============================================================

def create_scene_description(nodes, edges, max_objects=10, max_relations=5):
    """
    Create a natural language description of the scene.
    
    Example output:
    "A room with desk, chair, monitor, lamp, bookshelf. desk near chair; monitor on desk; lamp on desk"
    """
    # Get object labels (base labels, not unique labels)
    object_labels = []
    node_id_to_label = {}
    
    for node_id, node_data in nodes.items():
        base_label = node_data['base_label']
        object_labels.append(base_label)
        node_id_to_label[node_id] = base_label
    
    # Limit to max_objects (most common objects)
    object_counts = {}
    for label in object_labels:
        object_counts[label] = object_counts.get(label, 0) + 1
    
    # Sort by frequency and take top objects
    sorted_objects = sorted(object_counts.items(), key=lambda x: x[1], reverse=True)
    top_objects = [obj for obj, count in sorted_objects[:max_objects]]
    
    # Create object list string
    object_str = ", ".join(top_objects)
    
    # Add relationships - EXCLUDE floor/wall/ceiling (they dominate)
    # Focus on meaningful furniture-to-furniture relationships
    relation_strs = []
    subject_counts = {}
    
    # Objects to exclude from relationships (structural elements)
    exclude_from_relations = {'floor', 'wall', 'ceiling', 'walls'}
    
    for edge in edges:
        if len(relation_strs) >= max_relations:
            break
            
        subj_id = edge['subject']
        obj_id = edge['object']
        rel = edge['relation']
        
        # Get base labels
        if subj_id in node_id_to_label and obj_id in node_id_to_label:
            subj_label = node_id_to_label[subj_id]
            obj_label = node_id_to_label[obj_id]
            
            # SKIP if either subject or object is floor/wall/ceiling
            if subj_label in exclude_from_relations or obj_label in exclude_from_relations:
                continue
            
            # Skip if subject already has 2+ relationships (diversify)
            if subject_counts.get(subj_label, 0) >= 2:
                continue
                
            relation_strs.append(f"{subj_label} {rel} {obj_label}")
            subject_counts[subj_label] = subject_counts.get(subj_label, 0) + 1
    
    # Combine into full description
    if relation_strs:
        description = f"A room with {object_str}. {'; '.join(relation_strs)}"
    else:
        description = f"A room with {object_str}"
    
    return description


def get_scene_clip_embedding(description, clip_model, device):
    """Get CLIP embedding for scene description"""
    with torch.no_grad():
        tokens = clip.tokenize([description]).to(device)
        clip_emb = clip_model.encode_text(tokens)[0]
        clip_emb = clip_emb / clip_emb.norm()
    return clip_emb.cpu().tolist()


# ============================================================
# Main Scene Graph Builder
# ============================================================

def build_scene_graph(ply_path, semseg_path, output_path, scene_id, debug=False):
    """Build scene graph with scene-level CLIP embedding"""
    try:
        # Load data
        inst_points, all_points = load_ply_instances(ply_path)
        id_to_label = load_semseg(semseg_path)
        
        if debug:
            print(f"\n  DEBUG: Instance IDs from PLY: {list(inst_points.keys())[:10]}")
            print(f"  DEBUG: Label mapping from semseg: {id_to_label}")

        # Compute scene bounds
        scene_bounds = [
            all_points[:, 0].min(), all_points[:, 0].max(),
            all_points[:, 1].min(), all_points[:, 1].max(),
            all_points[:, 2].min(), all_points[:, 2].max(),
        ]

        # Build nodes
        nodes = {}
        clip_cache = {}
        label_counts = {}

        for iid, (pts, colors) in inst_points.items():
            # Skip background (objectId = 0)
            if iid == '0':
                if debug:
                    print(f"  Skipping objectId=0 (background)")
                continue
                
            base_label = id_to_label.get(iid, f"obj_{iid}")
            
            if debug and base_label.startswith("obj_"):
                print(f"  ⚠️  No label for objectId={iid}, using {base_label}")
            
            centroid, mean_color, radius = compute_node_attributes(pts, colors)
            geom_features = compute_geometric_features(pts)
            
            unique_label = create_unique_label(
                base_label, centroid, geom_features["bbox_size"],
                mean_color, scene_bounds, label_counts, iid
            )
            
            # Per-node CLIP embedding
            if unique_label not in clip_cache:
                tokens = clip.tokenize(unique_label).to(device)
                with torch.no_grad():
                    clip_emb = clip_model.encode_text(tokens)[0]
                    clip_emb = clip_emb / clip_emb.norm()
                clip_cache[unique_label] = clip_emb.cpu().tolist()

            nodes[iid] = {
                "label": unique_label,
                "base_label": base_label,
                "centroid": centroid,
                "mean_color": mean_color,
                "radius": radius,
                "clip_text_emb": clip_cache[unique_label],
                "geometric_features": geom_features
            }

        # Build relations
        K = 5
        obj_ids = list(nodes.keys())
        N = len(obj_ids)
        
        centroids = np.array([nodes[obj]["centroid"] for obj in obj_ids])
        dmat = np.linalg.norm(centroids[:, None, :] - centroids[None, :, :], axis=2)
        knn_idx = np.argsort(dmat, axis=1)[:, 1:K+1]

        edges = []
        seen = set()

        for i, oi in enumerate(obj_ids):
            ci = np.array(nodes[oi]["centroid"])
            ri = nodes[oi]["radius"]

            for j_idx in knn_idx[i]:
                oj = obj_ids[j_idx]
                cj = np.array(nodes[oj]["centroid"])
                rj = nodes[oj]["radius"]

                dirs = directional_rel(ci, cj)
                dist = distance_rel(ci, cj, ri, rj)

                if "touching" in dist:
                    dirs = [r for r in dirs if r in ["above", "below"]]

                rels = dirs + dist
                if not rels:
                    rels = ["close_by"]
                
                primary_rel = rels[0]
                
                key = (oi, oj, primary_rel)
                if key not in seen:
                    edges.append({"subject": oi, "object": oj, "relation": primary_rel})
                    seen.add(key)
                
                if primary_rel in ["above", "below", "left_of", "right_of", "in_front_of", "behind"]:
                    sym = symmetric_rel(oi, oj, primary_rel)
                    if sym:
                        sub, obj, rr = sym
                        skey = (sub, obj, rr)
                        if skey not in seen:
                            edges.append({"subject": sub, "object": obj, "relation": rr})
                            seen.add(skey)

        # Create scene-level description and CLIP embedding
        scene_description = create_scene_description(nodes, edges)
        scene_clip_emb = get_scene_clip_embedding(scene_description, clip_model, device)

        # Build output
        out = {
            "scene_id": scene_id,
            "scene_description": scene_description,  # NEW!
            "scene_clip_emb": scene_clip_emb,  # NEW!
            "nodes": nodes,
            "edges_text": edges
        }

        # Save
        with open(output_path, "w") as f:
            json.dump(out, f, indent=2)

        return True, len(nodes), len(edges), scene_description

    except Exception as e:
        print(f"  ❌ Error processing {scene_id}: {str(e)}")
        return False, 0, 0, None


# ============================================================
# Batch Processing
# ============================================================

def process_all_scenes(dataset_dir, output_dir):
    """Process all scenes in the dataset directory"""
    
    print(f"{'='*70}")
    print("SCENE GRAPH GENERATION WITH SCENE-LEVEL CLIP")
    print(f"{'='*70}\n")
    
    print(f"Dataset directory: {dataset_dir}")
    print(f"Output directory: {output_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all scene directories
    dataset_path = Path(dataset_dir)
    scene_dirs = [d for d in dataset_path.iterdir() if d.is_dir()]
    
    print(f"\nFound {len(scene_dirs)} scenes to process\n")
    
    # Process each scene
    successful = 0
    failed = 0
    total_nodes = 0
    total_edges = 0
    
    for scene_dir in tqdm(scene_dirs, desc="Processing scenes"):
        scene_id = scene_dir.name
        
        # Find PLY file (try multiple patterns)
        ply_path = None
        ply_patterns = [
            "mesh.refined.v2.obj.ply",
            "mesh.refined.obj.ply",
            "mesh.refined.0.010000.obj.ply",
            "*.obj.ply"
        ]
        
        for pattern in ply_patterns:
            if "*" in pattern:
                matches = list(scene_dir.glob(pattern))
                if matches:
                    ply_path = matches[0]
                    break
            else:
                candidate = scene_dir / pattern
                if candidate.exists():
                    ply_path = candidate
                    break
        
        # Find semseg file (try multiple patterns)
        semseg_path = None
        semseg_patterns = [
            "semseg.v2.json",
            "mesh.refined.0.010000.segs.v2.json",
            "*.segs.v2.json",
            "semseg.json"
        ]
        
        for pattern in semseg_patterns:
            if "*" in pattern:
                matches = list(scene_dir.glob(pattern))
                if matches:
                    semseg_path = matches[0]
                    break
            else:
                candidate = scene_dir / pattern
                if candidate.exists():
                    semseg_path = candidate
                    break
        
    for scene_dir in tqdm(scene_dirs, desc="Processing scenes"):
        scene_id = scene_dir.name
        
        # Find PLY and semseg files with correct 3RScan naming
        ply_path = scene_dir / "labels.instances.annotated.v2.ply"
        semseg_path = scene_dir / "semseg.v2.json"
        
        # Fallback patterns
        if not ply_path.exists():
            ply_path = scene_dir / "mesh.refined.v2.obj.ply"
        if not ply_path.exists():
            ply_path = scene_dir / "mesh.refined.obj.ply"
        
        if not ply_path.exists() or not semseg_path.exists():
            print(f"  ⚠️  Skipping {scene_id}: Missing files")
            failed += 1
            continue
        
        # Output path
        output_path = Path(output_dir) / f"{scene_id}.json"
        
        # Process scene
        debug = (successful == 0)  # Debug first scene only
        success, n_nodes, n_edges, description = build_scene_graph(
            str(ply_path),
            str(semseg_path),
            str(output_path),
            scene_id,
            debug=debug
        )
        
        if success:
            successful += 1
            total_nodes += n_nodes
            total_edges += n_edges
        else:
            failed += 1
    
    # Summary
    print(f"\n{'='*70}")
    print("PROCESSING COMPLETE")
    print(f"{'='*70}")
    print(f"✓ Successful: {successful}")
    print(f"✗ Failed: {failed}")
    print(f"📊 Total nodes: {total_nodes}")
    print(f"📊 Total edges: {total_edges}")
    print(f"📊 Avg nodes per scene: {total_nodes/successful:.1f}")
    print(f"📊 Avg edges per scene: {total_edges/successful:.1f}")
    print(f"\n✓ Scene graphs saved to: {output_dir}")
    print(f"{'='*70}\n")


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate scene graphs with scene-level CLIP for all scenes")
    parser.add_argument("--dataset_dir", required=True, 
                       help="Directory containing scene folders (e.g., /path/to/3RScan)")
    parser.add_argument("--output_dir", required=True,
                       help="Output directory for scene graph JSONs")
    
    args = parser.parse_args()
    
    process_all_scenes(args.dataset_dir, args.output_dir)