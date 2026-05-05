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
import argparse
import clip
import torch

from scripts.retrieval.utils import (
    DIR_THRESHOLD, HEIGHT_THRESHOLD, NEAR_THRESHOLD,
    get_spatial_descriptor, get_size_descriptor, get_color_descriptor,
    create_unique_label, load_ply_instances, load_semseg,
    compute_node_attributes, compute_geometric_features,
    directional_rel, distance_rel, symmetric_rel,
)

device = (
    "cuda"
    if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)
clip_model = None
clip_preprocess = None


def _ensure_clip(model_name: str = "ViT-B/32"):
    global clip_model, clip_preprocess
    if clip_model is None:
        print(f"Loading CLIP model ({model_name}) on {device}...")
        clip_model, clip_preprocess = clip.load(model_name, device=device)


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
    parser.add_argument("--clip_model", type=str, default="ViT-B/32",
                       help="CLIP model variant to load")
    args = parser.parse_args()

    _ensure_clip(args.clip_model)

    print("\n" + "="*70)
    print("SCENE GRAPH GENERATION WITH UNIQUE LABELS")
    print("="*70)

    build_scene_graph(args.ply, args.semseg, args.out)
    
    print("\n" + "="*70)
    print("✓ Complete!")
    print("  Now CLIP embeddings will be diverse and meaningful!")
    print("="*70)