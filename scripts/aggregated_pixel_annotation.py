import os
import json
import numpy as np
from plyfile import PlyData
from collections import defaultdict

def load_mesh_vertices(ply_file):
    """Load vertices (x,y,z) from ScanNet PLY."""
    plydata = PlyData.read(ply_file)
    vertices = np.vstack([
        plydata['vertex'].data['x'],
        plydata['vertex'].data['y'],
        plydata['vertex'].data['z']
    ]).T
    return vertices

def aggregate_instances(scene_path, output_json):
    # Find PLY file (prefer labels.ply if available)
    ply_file = os.path.join(scene_path, "scene0000_00_vh_clean_2.labels.ply")
    if not os.path.exists(ply_file):
        ply_file = os.path.join(scene_path, "scene0000_00_vh_clean_2.ply")

    seg_file = os.path.join(scene_path, "scene0000_00_vh_clean_2.0.010000.segs.json")
    agg_file = os.path.join(scene_path, "scene0000_00.aggregation.json")

    print("Using PLY:", ply_file)
    print("Using SEG:", seg_file)
    print("Using AGG:", agg_file)

    # Load vertices
    vertices = load_mesh_vertices(ply_file)
    print("Loaded vertices:", vertices.shape)

    # Load segIndices
    with open(seg_file, "r") as f:
        seg_data = json.load(f)
    seg_indices = np.array(seg_data["segIndices"])
    print("Loaded segIndices:", seg_indices.shape)

    if vertices.shape[0] != len(seg_indices):
        print("⚠️ Vertex count and segIndices length mismatch!")
        return

    # Load aggregation
    with open(agg_file, "r") as f:
        agg_data = json.load(f)

    # Track duplicate class counts
    class_counter = defaultdict(int)

    annotations = []
    for group in agg_data["segGroups"]:
        inst_id = group["id"]
        label = group["label"]
        segments = set(group["segments"])

        # Find vertices belonging to this instance
        mask = np.isin(seg_indices, list(segments))
        pts = vertices[mask]

        if pts.shape[0] == 0:
            continue

        # Increment counter for this class
        class_counter[label] += 1
        class_name = f"{label}_{class_counter[label]}"

        centroid = pts.mean(axis=0).tolist()
        bbox_min = pts.min(axis=0).tolist()
        bbox_max = pts.max(axis=0).tolist()

        annotations.append({
            "instance_id": inst_id,
            "class_name": class_name,
            "point_count": int(pts.shape[0]),
            "centroid": {"x": centroid[0], "y": centroid[1], "z": centroid[2]},
            "bounding_box": {
                "min_x": bbox_min[0], "min_y": bbox_min[1], "min_z": bbox_min[2],
                "max_x": bbox_max[0], "max_y": bbox_max[1], "max_z": bbox_max[2]
            }
        })
        print(f"Processed instance {inst_id} → {class_name}, {pts.shape[0]} vertices")

    scene_summary = {
        "scene": os.path.basename(scene_path),
        "annotations": annotations
    }

    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(scene_summary, f, indent=4)

    print(f"\n✅ Saved {len(annotations)} per-instance annotations to {output_json}")


if __name__ == "__main__":
    # Set your paths
    scene_path = "/Users/shirley/Documents/SCHOOL/SPRING25/masterproject/datasets/onescenedownload/Master-Project-Dataset-Creation/data/scans/scene0000_00"
    output_json = "/Users/shirley/Documents/SCHOOL/SPRING25/masterproject/datasets/onescenedownload/Master-Project-Dataset-Creation/data/aggregations/scene0000_00_instances.json"

    aggregate_instances(scene_path, output_json)
