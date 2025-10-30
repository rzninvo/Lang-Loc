import json
import numpy as np
from typing import List, Dict

# scene = "e3004a81-9f2a-2778-874e-fa76b0e67096"
scene = "fcf66d82-622d-291c-87be-78d421381146"

dataset_path = "/Users/shirley/Documents/SCHOOL/FALL2025/MASTER-PROJECT/3RScan"


def map_frames_from_clusters(
    clusters_path,
    descriptions_path,
    output_path,
    distance_thresh=0.2
    
) -> List[Dict]:
    """
    Build Set B (frame-level motifs) by mapping visible objects in each frame
    to global clusters (Set A) using geometric embedding information.

    Parameters
    ----------
    clusters_path : str
        Path to the Set A JSON file (clusters with centroid + size embeddings).
    descriptions_path : str
        Path to the per-frame JSON with visible objects and their centroids.
    distance_thresh : float
        Maximum centroid distance (in meters) to associate an object to a cluster
        if it is not strictly inside the cluster bounding box.
    output_path : str
        Path to save the resulting Set B JSON.

    Returns
    -------
    List[Dict]
        List of per-frame motif mappings.
    """
    # ------------------------------------------------------------
    # Load Set A clusters
    # ------------------------------------------------------------
    with open(clusters_path, "r") as f:
        clusters = json.load(f)
    for c in clusters:
        c["centroid"] = np.array(c["centroid"], dtype=float)
        c["bbox_min"] = np.array(c["bbox_min"], dtype=float)
        c["bbox_max"] = np.array(c["bbox_max"], dtype=float)

    def is_in_cluster(point, cluster):
        return np.all(point >= cluster["bbox_min"]) and np.all(point <= cluster["bbox_max"])

    # ------------------------------------------------------------
    # Load all frame data
    # ------------------------------------------------------------
    with open(descriptions_path, "r") as f:
        frames = json.load(f)

    all_motifs = []

    for frame in frames:
        fid = frame["image_index"]
        sid = frame["scene_index"]
        visible = frame["visible_objects"]
        relations = frame.get("spatial_relations", [])

        motifs = []

        # Iterate through all subject–object pairs
        for rel in relations:
            subj_label = rel["subject"]
            obj_label = rel["object"]
            predicate = rel["relation"]

            # Get centroid for subject & object if available
            subj = next((v for v in visible.values() if v["label"] == subj_label), None)
            obj = next((v for v in visible.values() if v["label"] == obj_label), None)
            if subj is None or obj is None:
                continue

            subj_centroid = np.array(subj["centroid_world"], dtype=float)
            obj_centroid = np.array(obj["centroid_world"], dtype=float)

            # Find cluster for subject/object
            subj_cluster, obj_cluster = None, None
            for c in clusters:
                if is_in_cluster(subj_centroid, c):
                    subj_cluster = c
                if is_in_cluster(obj_centroid, c):
                    obj_cluster = c

            # Fallback: use nearest cluster by centroid distance
            if subj_cluster is None or obj_cluster is None:
                for c in clusters:
                    dist_sub = np.linalg.norm(subj_centroid - c["centroid"])
                    dist_obj = np.linalg.norm(obj_centroid - c["centroid"])
                    if subj_cluster is None and dist_sub < distance_thresh:
                        subj_cluster = c
                    if obj_cluster is None and dist_obj < distance_thresh:
                        obj_cluster = c

            # Only keep motif if both are in same cluster
            if subj_cluster and obj_cluster and subj_cluster["cluster_id"] == obj_cluster["cluster_id"]:
                motifs.append({
                    "subject": subj_label,
                    "object": obj_label,
                    "relation": predicate,
                    "cluster_id": subj_cluster["cluster_id"],
                    "subject_id": [k for k, v in visible.items() if v == subj][0],
                    "object_id": [k for k, v in visible.items() if v == obj][0],
                    "centroid_subject": subj_centroid.tolist(),
                    "centroid_object": obj_centroid.tolist(),
                    "distance": round(float(np.linalg.norm(subj_centroid - obj_centroid)), 3)
                })

        all_motifs.append({
            "scene_index": sid,
            "image_index": fid,
            "num_motifs": len(motifs),
            "motifs": motifs
        })

    with open(output_path, "w") as f:
        json.dump(all_motifs, f, indent=2)

    print(f"Built Set B with relational motifs → saved to {output_path}")
    return all_motifs

def main():
    # desc_json = "/Users/shirley/Documents/SCHOOL/FALL2025/MASTER-PROJECT/3RScan/e3004a81-9f2a-2778-874e-fa76b0e67096/output/descriptions/all_descriptions.json"
    desc_json = f"{dataset_path}/{scene}/output/descriptions/all_descriptions.json"
    cluster_json = f"src/motif_creation_v2/{scene}/scene_clusters.json"
    save_path = f"src/motif_creation_v2/{scene}/frame_cluster_map.json"
    map_frames_from_clusters(clusters_path=cluster_json,
                             descriptions_path=desc_json,
                             output_path=save_path,
                             distance_thresh=0.2)
if __name__ == "__main__":
    main()