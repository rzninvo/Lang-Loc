"""
Create combined training dataset:
- ScanScribe text graphs with CLIP (no geometry)
- 3DSSG 3D graphs with CLIP + geometry

This gives you the best of both worlds!
"""

import os
import json
import torch
import numpy as np
import clip
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading CLIP...")
clip_model, _ = clip.load("ViT-B/32", device=device)
print("✓ CLIP loaded\n")


def convert_scanscribe_to_clip_format(scanscribe_graph, scene_id, text_id):
    """
    Convert ScanScribe graph (Word2Vec) to your format (CLIP).
    No geometry info - just structure and semantics.
    """
    nodes_dict = {}
    
    for node in scanscribe_graph['nodes']:
        label = node['label']
        
        # Get CLIP embedding
        with torch.no_grad():
            tokens = clip.tokenize([label]).to(device)
            clip_emb = clip_model.encode_text(tokens)
            clip_emb = clip_emb / clip_emb.norm(dim=-1, keepdim=True)
        
        nodes_dict[node['id']] = {
            "label": label,
            "base_label": label,
            "centroid": [0.0, 0.0, 0.0],  # No spatial info from text
            "mean_color": [128.0, 128.0, 128.0],  # Default gray
            "radius": 0.4,
            "clip_text_emb": clip_emb[0].cpu().numpy().tolist()
        }
    
    # Convert edges
    edges_text = []
    for edge in scanscribe_graph['edges']:
        edges_text.append({
            "subject": edge['source'],
            "object": edge['target'],
            "relation": edge['relationship']
        })
    
    return {
        "scene_id": scene_id,
        "text_id": text_id,
        "nodes": nodes_dict,
        "edges_text": edges_text,
        "source": "scanscribe"  # Mark as text-based
    }


def load_3dssg_with_clip(scene_graphs_dir):
    """
    Load your existing 3DSSG graphs (already have CLIP + geometry).
    These are your scene_graphs_unique/ files.
    """
    graphs = {}
    
    for filename in tqdm(os.listdir(scene_graphs_dir), desc="Loading 3DSSG"):
        if not filename.endswith('.json'):
            continue
        
        scene_id = filename.replace('.json', '')
        filepath = os.path.join(scene_graphs_dir, filename)
        
        with open(filepath, 'r') as f:
            graph_data = json.load(f)
        
        # Add source marker
        graph_data['source'] = '3dssg'
        graphs[scene_id] = graph_data
    
    return graphs


def load_scan_to_reference_mapping(metadata_path):
    """
    Load 3RScan.json and create scan_id → reference_id mapping.
    
    Example:
      scan_id: "4238491e-60a7-271e-9fe8-eb04b4209883"
      reference_id: "4731976a-f9f7-2a1a-9737-305b709ca37f"
    """
    print("Loading 3RScan.json metadata...")
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    scan_to_reference = {}
    reference_to_scans = {}
    
    for entry in metadata:
        reference_id = entry['reference']
        
        # Reference itself maps to itself
        scan_to_reference[reference_id] = reference_id
        reference_to_scans[reference_id] = [reference_id]
        
        # Each scan maps to the reference
        for scan in entry.get('scans', []):
            scan_id = scan['reference']
            scan_to_reference[scan_id] = reference_id
            reference_to_scans[reference_id].append(scan_id)
    
    print(f"✓ Loaded {len(scan_to_reference)} scan IDs")
    print(f"✓ Mapped to {len(reference_to_scans)} reference rooms\n")
    
    return scan_to_reference, reference_to_scans


def create_combined_dataset(scanscribe_path, scene_graphs_dir, metadata_path, output_dir='combined_dataset'):
    """
    Create combined training dataset with both text and 3D graphs.
    """
    
    print(f"{'='*70}")
    print("Creating Combined Dataset: ScanScribe + 3DSSG with CLIP")
    print(f"{'='*70}\n")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load scan → reference mapping
    scan_to_ref, ref_to_scans = load_scan_to_reference_mapping(metadata_path)
    
    # Load ScanScribe
    print("Loading ScanScribe data...")
    scanscribe_scenes = torch.load(scanscribe_path, map_location='cpu', weights_only=False)
    print(f"✓ Loaded {len(scanscribe_scenes)} ScanScribe scenes (scan IDs)\n")
    
    # Convert ScanScribe to CLIP format
    print("Converting ScanScribe to CLIP format...")
    print("DEBUG: Mapping scan IDs → reference IDs\n")
    
    scanscribe_graphs = {}
    total_texts = 0
    mapped_count = 0
    unmapped_scans = []
    
    for scan_id in tqdm(scanscribe_scenes.keys(), desc="Converting"):
        # Map scan_id → reference_id
        reference_id = scan_to_ref.get(scan_id)
        
        if reference_id is None:
            unmapped_scans.append(scan_id)
            continue
        
        # DEBUG: Show first 3 mappings
        if mapped_count < 3:
            print(f"  DEBUG: scan {scan_id[:20]}... → reference {reference_id[:20]}...")
        
        mapped_count += 1
        
        for text_id, graph in scanscribe_scenes[scan_id].items():
            graph_id = f"{reference_id}_text_{text_id}"
            scanscribe_graphs[graph_id] = convert_scanscribe_to_clip_format(
                graph, reference_id, text_id  # Use reference_id, not scan_id!
            )
            total_texts += 1
    
    print(f"\n✓ Converted {total_texts} text descriptions")
    print(f"✓ Mapped {mapped_count} scan IDs → reference IDs")
    
    if unmapped_scans:
        print(f"⚠️  {len(unmapped_scans)} scans not found in 3RScan.json:")
        for scan in unmapped_scans[:5]:
            print(f"    - {scan}")
        if len(unmapped_scans) > 5:
            print(f"    ... and {len(unmapped_scans) - 5} more")
    
    print()
    
    # Load 3DSSG
    print("Loading 3DSSG graphs...")
    dssg_graphs = load_3dssg_with_clip(scene_graphs_dir)
    print(f"✓ Loaded {len(dssg_graphs)} 3DSSG scenes\n")
    
    # Find overlapping scenes (by reference ID)
    scanscribe_reference_ids = set(g['scene_id'] for g in scanscribe_graphs.values())
    dssg_reference_ids = set(dssg_graphs.keys())
    overlapping = scanscribe_reference_ids & dssg_reference_ids
    
    print(f"{'='*70}")
    print("OVERLAP ANALYSIS (by reference room ID)")
    print(f"{'='*70}")
    print(f"ScanScribe references: {len(scanscribe_reference_ids)}")
    print(f"3DSSG references: {len(dssg_reference_ids)}")
    print(f"Overlapping references: {len(overlapping)}")
    
    if len(overlapping) > 0:
        print(f"\nFirst 5 overlapping reference IDs:")
        for i, ref_id in enumerate(list(overlapping)[:5]):
            # Count how many texts for this reference
            num_texts = sum(1 for g in scanscribe_graphs.values() if g['scene_id'] == ref_id)
            print(f"  {i+1}. {ref_id}")
            print(f"     → {num_texts} text descriptions")
            if ref_id in ref_to_scans:
                print(f"     → {len(ref_to_scans[ref_id])} scans total")
    
    print(f"\nNon-overlapping:")
    print(f"  ScanScribe only: {len(scanscribe_reference_ids - dssg_reference_ids)}")
    print(f"  3DSSG only: {len(dssg_reference_ids - scanscribe_reference_ids)}")
    print()
    
    # Save ScanScribe graphs
    print("Saving ScanScribe graphs...")
    for graph_id, graph_data in tqdm(scanscribe_graphs.items(), desc="Saving"):
        filepath = os.path.join(output_dir, f"{graph_id}.json")
        with open(filepath, 'w') as f:
            json.dump(graph_data, f, indent=2)
    
    # Save 3DSSG graphs
    print("\nSaving 3DSSG graphs...")
    for scene_id, graph_data in tqdm(dssg_graphs.items(), desc="Saving"):
        filepath = os.path.join(output_dir, f"{scene_id}.json")
        with open(filepath, 'w') as f:
            json.dump(graph_data, f, indent=2)
    
    # Save metadata
    metadata = {
        "total_graphs": len(scanscribe_graphs) + len(dssg_graphs),
        "scanscribe_text_graphs": len(scanscribe_graphs),
        "3dssg_3d_graphs": len(dssg_graphs),
        "overlapping_references": len(overlapping),
        "overlapping_reference_ids": sorted(list(overlapping)),
        "scanscribe_unique_references": len(scanscribe_reference_ids - dssg_reference_ids),
        "3dssg_unique_references": len(dssg_reference_ids - scanscribe_reference_ids),
        "format": {
            "node_features": "centroid(3) + color(3) + CLIP(512) = 518 dims",
            "clip_model": "ViT-B/32",
            "scanscribe_geometry": "dummy [0,0,0] centroids (text has no spatial info)",
            "3dssg_geometry": "real centroids + colors from 3D scans",
            "id_mapping": "ScanScribe scan IDs mapped to 3RScan reference IDs"
        }
    }
    
    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n{'='*70}")
    print("DATASET CREATED!")
    print(f"{'='*70}")
    print(f"Total graphs: {metadata['total_graphs']}")
    print(f"  Text graphs: {metadata['scanscribe_text_graphs']}")
    print(f"  3D graphs: {metadata['3dssg_3d_graphs']}")
    print(f"\nOverlap (by reference room):")
    print(f"  Both datasets: {metadata['overlapping_references']} references")
    print(f"  ScanScribe only: {metadata['scanscribe_unique_references']} references")
    print(f"  3DSSG only: {metadata['3dssg_unique_references']} references")
    print(f"\nSaved to: {output_dir}/")
    
    # Debug: Show example filenames
    print(f"\nExample files created:")
    files = sorted(os.listdir(output_dir))[:5]
    for f in files:
        if f.endswith('.json') and f != 'metadata.json':
            print(f"  - {f}")
    
    print(f"\nYou can now train with:")
    print(f"  - Text→3D matching (ScanScribe text → 3DSSG database)")
    print(f"  - 3D→3D matching (3DSSG → 3DSSG)")
    print(f"  - Mixed batches for robust learning!")
    
    return metadata


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Create combined ScanScribe+3DSSG CLIP dataset')
    parser.add_argument('--scanscribe_path', type=str, required=True, help='Path to ScanScribe train graphs .pt')
    parser.add_argument('--scene_graphs_dir', type=str, required=True, help='Directory with unique-label scene graphs')
    parser.add_argument('--metadata_path', type=str, required=True, help='Path to 3RScan.json metadata')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for combined dataset')
    ds_args = parser.parse_args()

    # Create dataset
    metadata = create_combined_dataset(
        scanscribe_path=ds_args.scanscribe_path,
        scene_graphs_dir=ds_args.scene_graphs_dir,
        metadata_path=ds_args.metadata_path,
        output_dir=ds_args.output_dir
    )
    
    print(f"\n{'='*70}")
    print("NEXT STEPS")
    print(f"{'='*70}")
