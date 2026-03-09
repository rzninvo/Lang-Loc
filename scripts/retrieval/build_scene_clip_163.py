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
import argparse
import clip
import torch
import os
from pathlib import Path
from tqdm import tqdm

from scripts.retrieval.utils import (
    DIR_THRESHOLD, HEIGHT_THRESHOLD, NEAR_THRESHOLD,
    get_spatial_descriptor, get_size_descriptor, get_color_descriptor,
    create_unique_label, load_ply_instances, load_semseg,
    compute_node_attributes, compute_geometric_features,
    directional_rel, distance_rel, symmetric_rel,
    create_scene_description, get_scene_clip_embedding,
)

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Loading CLIP model on {device}...")
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
print("✓ CLIP loaded\n")


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

    data = torch.load('/Users/shirley/Downloads/scanscribe_graphs_train_final_no_graph_min.pt', 
                  weights_only=False, map_location='cpu')
    
    train_scene_ids = list(data.keys())
    print(f"Number of training scenes: {len(train_scene_ids)}")
    print("First 5:", train_scene_ids[:5])

    # Save to text file
    with open('scanscribe_train_scene_ids.txt', 'w') as f:
        for sid in train_scene_ids:
            f.write(sid + '\n')
    print("Saved to scanscribe_train_scene_ids.txt")
    
    # Find all scene directories
    dataset_path = Path(dataset_dir)
    scene_dirs = [d for d in dataset_path.iterdir() 
              if d.is_dir() and d.name in set(train_scene_ids)]
    
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