"""
Add relationship CLIP embeddings to scene graphs.

Updates your existing scene graphs to include CLIP embeddings for relationships.
This will help the model learn edge semantics, not just node semantics.
"""

import json
import os
import clip
import torch
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
clip_model = None


def _ensure_clip(model_name: str = "ViT-B/32"):
    global clip_model
    if clip_model is None:
        print(f"Loading CLIP model ({model_name}) on {device}...")
        clip_model, _ = clip.load(model_name, device=device)


def add_relation_clip_to_scene_graph(scene_path, output_path):
    """Add CLIP embeddings to relationships in existing scene graph."""
    
    with open(scene_path, 'r') as f:
        data = json.load(f)
    
    # Process edges
    edges_with_clip = []
    
    for edge in data.get('edges_text', []):
        relation = edge['relation']
        
        # Get CLIP embedding for relationship
        with torch.no_grad():
            tokens = clip.tokenize([relation]).to(device)
            clip_emb = clip_model.encode_text(tokens)
            clip_emb = clip_emb / clip_emb.norm(dim=-1, keepdim=True)
        
        edge_with_clip = edge.copy()
        edge_with_clip['relation_clip_emb'] = clip_emb[0].cpu().tolist()
        edges_with_clip.append(edge_with_clip)
    
    # Update data
    data['edges_text'] = edges_with_clip
    
    # Save
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)


def process_all_scenes(input_dir, output_dir):
    """Add relationship CLIP to all scene graphs."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    scene_files = [f for f in os.listdir(input_dir) if f.endswith('.json')]
    
    print(f"Processing {len(scene_files)} scene graphs...")
    
    for filename in tqdm(scene_files):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        
        add_relation_clip_to_scene_graph(input_path, output_path)
    
    print(f"✓ Done! Updated scenes saved to {output_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--clip_model", type=str, default="ViT-B/32",
                       help="CLIP model variant to load")
    args = parser.parse_args()

    _ensure_clip(args.clip_model)
    process_all_scenes(args.input_dir, args.output_dir)