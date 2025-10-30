import json
import numpy as np
from itertools import combinations
import networkx as nx
import pathlib,os

# scene = "e3004a81-9f2a-2778-874e-fa76b0e67096"
scene = "fcf66d82-622d-291c-87be-78d421381146"
dataset_path = "/Users/shirley/Documents/SCHOOL/FALL2025/MASTER-PROJECT/3RScan"


def build_global_clusters_bbox(json_path="semseg.v2.json", overlap_thresh=0.05, save_path="scene_clusters_bbox.json", summary_json_path="scene_clusters_summary.json"):
    """
    Build clusters (Set A) by checking bounding box overlap instead of centroid distance.
    overlap_thresh: fractional overlap ratio (0.0 = any contact, 0.1 = significant overlap)
    """

    with open(json_path, "r") as f:
        data = json.load(f)

    objects = data["segGroups"]

    # Extract AABBs from OBBs
    boxes = {}
    labels = {}
    for obj in objects:
        oid = str(obj["objectId"])
        label = obj["label"]
        c = np.array(obj["obb"]["centroid"])
        half_l = np.array(obj["obb"]["axesLengths"]) / 2.0

        # axis-aligned approximation
        min_pt = c - half_l
        max_pt = c + half_l

        boxes[oid] = (min_pt, max_pt)
        labels[oid] = label

    # Overlap function
    def overlap_ratio(box1, box2):
        min1, max1 = box1
        min2, max2 = box2

        inter_min = np.maximum(min1, min2)
        inter_max = np.minimum(max1, max2)
        inter_dim = np.maximum(0, inter_max - inter_min)
        inter_vol = np.prod(inter_dim)

        vol1 = np.prod(max1 - min1)
        vol2 = np.prod(max2 - min2)
        union_vol = vol1 + vol2 - inter_vol

        if union_vol == 0:
            return 0
        return inter_vol / union_vol

    # Build graph of touching/overlapping objects
    G = nx.Graph()
    for oid in boxes:
        G.add_node(oid, label=labels[oid])

    for o1, o2 in combinations(boxes.keys(), 2):
        ov = overlap_ratio(boxes[o1], boxes[o2])
        if ov > overlap_thresh:
            G.add_edge(o1, o2, overlap=ov)

    # Connected components → clusters
    clusters = []
    clusters_summary = []

    for comp in nx.connected_components(G):
        cluster_objs = list(comp)
        cluster_labels = [labels[o] for o in cluster_objs]

        # Compute aggregate box
        all_mins = np.array([boxes[o][0] for o in cluster_objs])
        all_maxs = np.array([boxes[o][1] for o in cluster_objs])
        bbox_min = np.min(all_mins, axis=0)
        bbox_max = np.max(all_maxs, axis=0)
        centroid = (bbox_min + bbox_max) / 2.0
        size = bbox_max - bbox_min

        geometric_embedding = np.concatenate([centroid, size])

        clusters.append({
            "cluster_id": f"A_{len(clusters)}",
            "object_ids": cluster_objs,
            "labels": cluster_labels,
            "bbox_min": bbox_min.tolist(),
            "bbox_max": bbox_max.tolist(),
            "centroid": centroid.tolist(),
            "size": size.tolist(),
            "embedding": geometric_embedding.tolist()
        })

        clusters_summary.append(f"A_{len(clusters)}" + ": " + ", ".join(cluster_labels))


    with open(save_path, "w") as f:
        json.dump(clusters, f, indent=2)
    print(f"Built {len(clusters)} clusters based on bounding box overlap → saved to {save_path}")

    with open(summary_json_path, "w") as f:
        json.dump(clusters_summary, f, indent=2)
    print(f"Saved clusters summary to {summary_json_path}")



    return clusters



def main():
    dataset_path = "/Users/shirley/Documents/SCHOOL/FALL2025/MASTER-PROJECT/3RScan"
    object_json_path = f"{dataset_path}/{scene}/semseg.v2.json"
    # object_json_path = "/Users/shirley/Documents/SCHOOL/FALL2025/MASTER-PROJECT/3RScan/e3004a81-9f2a-2778-874e-fa76b0e67096/semseg.v2.json"
       # ensure output dir exists (make path relative to this script so tempCodeRunner won't break it)
    base_dir = pathlib.Path(__file__).resolve().parent  # src/motif_creation_v2
    json_output_dir = base_dir / scene
    json_output_dir.mkdir(parents=True, exist_ok=True)
    json_output_path = f"src/motif_creation_v2/{scene}/scene_clusters.json"
    summary_json_path = f"src/motif_creation_v2/{scene}/summary.json"

    build_global_clusters_bbox(object_json_path, overlap_thresh=0.03, save_path=json_output_path, summary_json_path=summary_json_path)


if __name__ == "__main__":
    main()