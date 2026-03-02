"""
Modified Scene Graph Builder - Creates unique, spatially-aware labels

KEY CHANGE: Instead of "wall", "wall", "wall", creates:
  - "wall_north_upper"
  - "wall_south_lower"
  - "wall_east_middle"

This makes CLIP embeddings actually useful!
"""

import json
import numpy as np
from plyfile import PlyData
import argparse
import clip
import torch

device = "mps" if torch.backends.mps.is_available() else "cpu"
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)

# Thresholds (same as before)
DIR_THRESHOLD = 0.20
HEIGHT_THRESHOLD = 0.15
NEAR_THRESHOLD = 0.80


# ============================================================
# NEW: Create spatially-aware labels
# ============================================================

def get_spatial_descriptor(centroid, bbox_size, scene_bounds):
    """
    Create a spatial descriptor based on position in the room.
    
    Returns: e.g., "north_upper", "south_lower", "east_middle"
    """
    x, y, z = centroid
    min_x, max_x, min_y, max_y, min_z, max_z = scene_bounds
    
    # Horizontal position (north/south/east/west/center)
    x_range = max_x - min_x
    z_range = max_z - min_z
    
    x_norm = (x - min_x) / x_range if x_range > 0 else 0.5
    z_norm = (z - min_z) / z_range if z_range > 0 else 0.5
    
    # Determine horizontal position
    if abs(x_norm - 0.5) > 0.3:  # Far from center
        horizontal = "east" if x_norm > 0.5 else "west"
    elif abs(z_norm - 0.5) > 0.3:
        horizontal = "north" if z_norm > 0.5 else "south"
    else:
        horizontal = "center"
    
    # Vertical position (upper/middle/lower)
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
    """Add node_id to make labels unique"""
    
    if base_label.startswith("obj") or base_label == "object" or base_label == "unknown":
        size = get_size_descriptor(bbox_size)
        color = get_color_descriptor(mean_color)
        spatial = get_spatial_descriptor(centroid, bbox_size, scene_bounds)
        
        # ADD NODE ID to make it unique!
        unique_label = f"{size}_{color}_object_{node_id}_{spatial}"
    else:
        # Named objects
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
# Load PLY and group points by instanceId
# ============================================================

def load_ply_instances(ply_path):
    print("\n[PLY] Reading:", ply_path)
    ply = PlyData.read(ply_path)
    v = ply["vertex"].data

    xyz = np.vstack([v["x"], v["y"], v["z"]]).T
    rgb = np.vstack([v["red"], v["green"], v["blue"]]).T
    inst = v["objectId"]

    instance_ids = np.unique(inst)
    print(f"[PLY] Found {len(instance_ids)} unique object IDs")

    instance_points = {}
    for iid in instance_ids:
        mask = inst == iid
        pts = xyz[mask]
        colors = rgb[mask]
        instance_points[str(iid)] = (pts, colors)

    return instance_points, xyz  # Also return all points for scene bounds


def load_semseg(semseg_path):
    print("\n[SEMSEG] Loading:", semseg_path)
    with open(semseg_path, "r") as f:
        data = json.load(f)

    id_to_label = {str(o["objectId"]): o["label"].lower()
                   for o in data["segGroups"]}

    print(f"[SEMSEG] Loaded {len(id_to_label)} labels.")
    return id_to_label


def compute_node_attributes(points, colors):
    centroid = points.mean(axis=0)
    d = np.linalg.norm(points - centroid, axis=1)
    radius = float(min(d.max(), 0.40))
    mean_color = colors.mean(axis=0).tolist()

    return centroid.tolist(), mean_color, radius


def compute_geometric_features(points):
    """Compute geometric features for the object."""
    # Covariance-based features
    centered = points - points.mean(axis=0)
    cov = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    eigenvalues = np.maximum(eigenvalues, 1e-8)
    eigenvalues = np.sort(eigenvalues)[::-1]  # Descending
    
    # Shape descriptors
    l1, l2, l3 = eigenvalues
    linearity = (l1 - l2) / l1 if l1 > 0 else 0
    planarity = (l2 - l3) / l1 if l1 > 0 else 0
    sphericity = l3 / l1 if l1 > 0 else 0
    
    # Standard deviation
    std_dev = np.sqrt(eigenvalues).tolist()
    
    # Extent (eigenvalues are variances, so sqrt for std dev)
    extent = (2 * np.sqrt(eigenvalues)).tolist()
    
    # Bounding box
    bbox_min = points.min(axis=0)
    bbox_max = points.max(axis=0)
    bbox_size = (bbox_max - bbox_min).tolist()
    
    return {
        "std_dev": std_dev,
        "std_color": [0, 0, 0],  # Placeholder
        "extent": extent,
        "linearity": float(linearity),
        "planarity": float(planarity),
        "sphericity": float(sphericity),
        "bbox_size": bbox_size,
        "num_points": int(len(points))
    }


# ============================================================
# Relation helpers (same as before)
# ============================================================

def directional_rel(ci, cj):
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
    d = np.linalg.norm(cj - ci)
    if d <= (ri + rj) * 0.65:
        return ["touching"]
    if d < NEAR_THRESHOLD:
        return ["near"]
    return []


def symmetric_rel(sub, obj, rel):
    if rel == "left_of": return obj, sub, "right_of"
    if rel == "right_of": return obj, sub, "left_of"
    if rel == "in_front_of": return obj, sub, "behind"
    if rel == "behind": return obj, sub, "in_front_of"
    if rel == "above": return obj, sub, "below"
    if rel == "below": return obj, sub, "above"
    if rel == "near": return obj, sub, "near"
    return None


# ============================================================
# Main Scene Graph Builder (MODIFIED)
# ============================================================

def build_scene_graph(ply_path, semseg_path, output_path):
    inst_points, all_points = load_ply_instances(ply_path)
    id_to_label = load_semseg(semseg_path)

    # Compute scene bounds for spatial descriptors
    scene_bounds = [
        all_points[:, 0].min(), all_points[:, 0].max(),  # x
        all_points[:, 1].min(), all_points[:, 1].max(),  # y
        all_points[:, 2].min(), all_points[:, 2].max(),  # z
    ]
    
    print(f"\n[SCENE] Bounds: x=[{scene_bounds[0]:.2f}, {scene_bounds[1]:.2f}], "
          f"y=[{scene_bounds[2]:.2f}, {scene_bounds[3]:.2f}], "
          f"z=[{scene_bounds[4]:.2f}, {scene_bounds[5]:.2f}]")

    # Build nodes with unique labels
    nodes = {}
    clip_cache = {}
    label_counts = {}  # Track label occurrences
    
    print("\n[NODES] Creating unique labels and CLIP embeddings...")

    for iid, (pts, colors) in inst_points.items():
        # Get base label from semseg
        base_label = id_to_label.get(iid, f"obj_{iid}")
        
        # Compute attributes
        centroid, mean_color, radius = compute_node_attributes(pts, colors)
        geom_features = compute_geometric_features(pts)
        
        # Create unique label with spatial info
        unique_label = create_unique_label(
            base_label, 
            centroid, 
            geom_features["bbox_size"],
            mean_color,
            scene_bounds,
            label_counts,
            iid
        )
        
        # Generate CLIP embedding for unique label
        if unique_label not in clip_cache:
            tokens = clip.tokenize(unique_label).to(device)
            with torch.no_grad():
                clip_emb = clip_model.encode_text(tokens)[0]
                clip_emb = clip_emb / clip_emb.norm()
            clip_cache[unique_label] = clip_emb.cpu().tolist()

        nodes[iid] = {
            "label": unique_label,
            "base_label": base_label,  # Keep original for reference
            "centroid": centroid,
            "mean_color": mean_color,
            "radius": radius,
            "clip_text_emb": clip_cache[unique_label],
            "geometric_features": geom_features
        }

        print(f"  {iid}: {base_label:20s} → {unique_label}")

    # Build relations (same K-NN approach as before)
    print(f"\n[RELATIONS] Building K-nearest neighbor relations (K=5)...")
    
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

    print(f"[RELATIONS] Created {len(edges)} relations")

    # Save
    out = {
        "scene_id": ply_path.split("/")[-2],
        "nodes": nodes,
        "edges_text": edges
    }

    with open(output_path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\n[OK] Scene graph saved: {output_path}")
    print(f"     {len(nodes)} nodes with unique labels")
    print(f"     {len(clip_cache)} unique CLIP embeddings generated")


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ply", required=True)
    parser.add_argument("--semseg", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    print("\n" + "="*70)
    print("SCENE GRAPH GENERATION WITH UNIQUE LABELS")
    print("="*70)
    
    build_scene_graph(args.ply, args.semseg, args.out)
    
    print("\n" + "="*70)
    print("✓ Complete!")
    print("  Now CLIP embeddings will be diverse and meaningful!")
    print("="*70)