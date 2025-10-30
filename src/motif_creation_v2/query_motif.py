import json
import re
import numpy as np

scene = "fcf66d82-622d-291c-87be-78d421381146"
dataset_path = "/Users/shirley/Documents/SCHOOL/FALL2025/MASTER-PROJECT/3RScan"

cluster_path = f"src/motif_creation_v2/{scene}/scene_clusters.json"
motif_path = f"src/motif_creation_v2/{scene}/frame_cluster_map.json"

def load_data(clusters_path, motifs_path):
    with open(clusters_path, "r") as f:
        clusters = json.load(f)
    with open(motifs_path, "r") as f:
        motifs = json.load(f)
    return clusters, motifs


def parse_user_input(text):
    """
    Very rough NLP: extract object pairs and directional keywords. Needs further refinement using LLM.
    """
    #cut text using commas and "and"
    parts = re.split(r",| and ", text.lower())
    objects = [p.strip() for p in parts if p.strip() and not any(dir in p for dir in ["left", "right", "front", "behind", "above", "below", "under", "on top", "on floor"])]    
    relations = []
    if "left" in text:
        relations.append("left_of")
    if "right" in text:
        relations.append("right_of")
    if "front" in text or "ahead" in text:
        relations.append("in_front_of")
    if "behind" in text:
        relations.append("behind")
    if "above" in text or "on top" in text:
        relations.append("above")
    if "below" in text or "under" in text or "on floor" in text:
        relations.append("below")
    return objects, relations


def find_matching_motifs(motifs, objects, relations):
    matches = []
    for frame in motifs:
        for m in frame["motifs"]:
            if (
                (m["subject"] in objects and m["object"] in objects)
                and (not relations or m["relation"] in relations)
            ):
                matches.append((frame["image_index"], m))
    return matches


def find_cluster_neighbors(clusters, target_cluster, radius=1.5):
    """Find clusters near a given cluster centroid."""
    c0 = np.array(target_cluster["centroid"])
    neighbors = []
    for c in clusters:
        dist = np.linalg.norm(np.array(c["centroid"]) - c0)
        if 0 < dist < radius:
            neighbors.append((dist, c))
    neighbors.sort(key=lambda x: x[0])
    return [c for _, c in neighbors]
# ---------------------------------------------------------------------
# Main chatbot
# ---------------------------------------------------------------------

def query_chatbot(clusters_path="scene_clusters_bbox.json",
                  motifs_path="frame_motifs_with_relations.json",
                  target_frame_count=3):
    clusters, motifs = load_data(clusters_path, motifs_path)
    current_candidates = None  # Start with no frame restriction

    print("What group of objects can you clearly see that are close to each other?")
    while True:
        user_text = input("You: ")
        objects, relations = parse_user_input(user_text)
        if not objects:
            print("No recognizable objects found in your input. Try again.")
            continue

        print(f"Parsed → objects={objects}, relations={relations}")

        # ------------------------------------------------------------
        # Step 1. Try direct motif match
        # ------------------------------------------------------------
        matches = find_matching_motifs(motifs, objects, relations)
        if matches:
            frames = {fid for fid, _ in matches}
            if current_candidates is None:
                current_candidates = frames
            else:
                # Intersect with existing candidates to narrow down
                current_candidates = current_candidates & frames

            print(f"Found {len(frames)} motif matches in {len(frames)} frames.")
            print(f"After narrowing, {len(current_candidates)} candidate frame(s) remain.")

        else:
            # --------------------------------------------------------
            # Step 2. Check clusters containing these objects
            # --------------------------------------------------------
            matched_clusters = [c for c in clusters if any(o in c["labels"] for o in objects)]
            if not matched_clusters:
                print("Could not find any clusters containing those objects.")
                continue

            cluster_ids = [c["cluster_id"] for c in matched_clusters]
            print(f"objects map to clusters: {cluster_ids}")

            # --------------------------------------------------------
            # If objects are from different clusters, find nearby ones
            # --------------------------------------------------------
            if len(set(cluster_ids)) > 1:
                print("🔎 The mentioned objects belong to different clusters.")
                expanded_clusters = set(cluster_ids)
                for c in matched_clusters:
                    neighbors = find_cluster_neighbors(clusters, c)
                    expanded_clusters.update([n["cluster_id"] for n in neighbors])
                print(f"Considering nearby clusters: {list(expanded_clusters)[:5]}")
            else:
                expanded_clusters = {cluster_ids[0]}

            # --------------------------------------------------------
            # Step 3. Find frames that contain motifs from these clusters
            # --------------------------------------------------------
            frames = set()
            for frame in motifs:
                frame_clusters = {m["cluster_id"] for m in frame["motifs"]}
                if frame_clusters & expanded_clusters:
                    objs_in_frame = {m["subject"] for m in frame["motifs"]} | {m["object"] for m in frame["motifs"]}
                    if any(o in objs_in_frame for o in objects):
                        frames.add(frame["image_index"])

            if frames:
                if current_candidates is None:
                    current_candidates = frames
                else:
                    current_candidates = current_candidates & frames
                print(f"Found {len(frames)} candidate frames from related clusters.")
                print(f"After narrowing, {len(current_candidates)} candidate frame(s) remain.")
            else:
                print("No frames contain motifs from these clusters.")

        # ------------------------------------------------------------
        # Step 4. Evaluate current candidate set
        # ------------------------------------------------------------
        if not current_candidates or len(current_candidates) == 0:
            print("No candidate frames found yet. Try describing more objects.")
        elif len(current_candidates) <= target_frame_count:
            print(f"Narrowed down to {len(current_candidates)} unique frame(s): {list(current_candidates)}")
            break
        else:
            print(f"Still {len(current_candidates)} possible frames.")
            print("Please describe another nearby object or relation to narrow further.")
            continue

       
if __name__ == "__main__":
    query_chatbot(clusters_path=cluster_path, motifs_path=motif_path)