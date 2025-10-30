import json
from pathlib import Path
import numpy as np

def load_descriptions(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} frame entries.")
    return data

def build_motif_index(descriptions):
    """
    Builds a motif index from the scene descriptions file.
    Each motif corresponds to a (subject, predicate, object) triplet enriched with geometry.
    
    """

    motif_index = []

    for desc in descriptions:
        scene_id = desc["scene_index"]
        frame_id = desc["image_index"]
        visible_objects = desc.get("visible_objects", {})
        spatial_relations = desc.get("spatial_relations", [])

        # Build dictionary of centroids for quick access
        centroids_world = {}
        centroids_cam = {}

        for obj_data in visible_objects.values():
            label = obj_data.get("label")
            if not label:
                continue
            centroids_world[label] = np.array(obj_data.get("centroid_world", [0, 0, 0]), dtype=float)
            centroids_cam[label] = np.array(obj_data.get("centroid_cam", [0, 0, 0]), dtype=float)

        # Loop through all spatial relations
        for rel in spatial_relations:
            subj = rel.get("subject")
            obj = rel.get("object")
            predicate = rel.get("relation")

            # Skip if one of the objects not found
            if subj not in centroids_world or obj not in centroids_world:
                continue

            # Compute geometry
            c1_world = centroids_world[subj]
            c2_world = centroids_world[obj]
            c1_cam = centroids_cam.get(subj, np.zeros(3))
            c2_cam = centroids_cam.get(obj, np.zeros(3))

            direction_world = c2_world - c1_world
            distance_world = float(np.linalg.norm(direction_world))
            direction_unit_world = (
                direction_world / distance_world if distance_world > 0 else np.zeros(3)
            )
            midpoint_world = ((c1_world + c2_world) / 2).tolist()

            motif = {
                "scene": scene_id,
                "frame": frame_id,
                "relation": {
                    "subject": subj,
                    "object": obj,
                    "predicate": predicate,
                    "geometry": {
                        # World-space metrics (stable)
                        "centroid_world_subject": c1_world.tolist(),
                        "centroid_world_object": c2_world.tolist(),
                        "distance_world": distance_world,
                        "direction_vector_world": direction_unit_world.tolist(),
                        "midpoint_world": midpoint_world,

                        # Camera-space context (view dependent)
                        "centroid_cam_subject": c1_cam.tolist(),
                        "centroid_cam_object": c2_cam.tolist(),
                    },
                },
            }

            motif_index.append(motif)


    return motif_index

def save_motif_index(motif_index, output_path="src/motif_creation/sample_motif_index.json"):
    with open(output_path, "w") as f:
        json.dump(motif_index, f, indent=2)
    print(f"Saved motif index to {output_path}")

def main():
    data_path = Path("/Users/shirley/Documents/SCHOOL/FALL2025/MASTER-PROJECT/3RScan/e3004a81-9f2a-2778-874e-fa76b0e67096/output/descriptions/all_descriptions.json")  # adjust if needed
    data = load_descriptions(data_path)
    motif_index = build_motif_index(data)
    save_motif_index(motif_index)

if __name__ == "__main__":
    main()