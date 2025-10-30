# archive: dont use this file


import json
import numpy as np

class SceneChatbot:
    def __init__(self, motif_index_path: str):
        with open(motif_index_path, "r") as f:
            self.motifs = json.load(f)

        # Index motifs by (scene, frame)
        self.frames = {}
        for m in self.motifs:
            scene, frame = m["scene"], m["frame"]
            subj, obj = m["relation"]["subject"], m["relation"]["object"]
            key = (scene, frame)
            if key not in self.frames:
                self.frames[key] = []
            self.frames[key].append(m)
        
        self.current_candidates = list(self.frames.keys())

    # --------------------------------------------------------
    # Step 1: Ask for visible objects
    # --------------------------------------------------------
    def ask_objects_seen(self):
        print("\n🤖 What objects do you see? (e.g., 'sofa, floor, curtain')")
        user_input = input("You: ").strip().lower()
        objects = [obj.strip() for obj in user_input.split(",") if obj.strip()]
        self.filter_by_objects_seen(objects)
        self.objects_seen = objects

    def filter_by_objects_seen(self, objects):
        filtered = []
        for (scene, frame), motifs in self.frames.items():
            frame_objs = {m["relation"]["subject"] for m in motifs} | {m["relation"]["object"] for m in motifs}
            if all(o in frame_objs for o in objects):
                filtered.append((scene, frame))

        self.current_candidates = filtered
        print(f"\n✅ Found {len(filtered)} candidate frames containing {objects}.")
        for s, f in filtered[:5]:
            print(f" - Scene: {s}, Frame: {f}")
        if len(filtered) > 5:
            print("   ...and more.")

    # --------------------------------------------------------
    # Step 2: Filter by nearest object
    # --------------------------------------------------------
    def ask_nearest_object(self):
        if not self.current_candidates:
            print("⚠️ No candidate frames. Try again with different objects.")
            return

        print("\n🤖 What is the nearest object to you?")
        nearest_obj = input("You: ").strip().lower()
        self.filter_by_nearest_object(nearest_obj)
        self.nearest_obj = nearest_obj

    def filter_by_nearest_object(self, nearest_obj):
        new_candidates = []
        for m in self.motifs:
            scene, frame = m["scene"], m["frame"]
            if (scene, frame) not in self.current_candidates:
                continue

            rel = m["relation"]
            subj, obj = rel["subject"], rel["object"]
            geom = rel["geometry"]

            subj_z = geom["centroid_cam_subject"][2]
            obj_z = geom["centroid_cam_object"][2]
            closest_label = subj if subj_z < obj_z else obj

            if closest_label == nearest_obj:
                new_candidates.append((scene, frame))

        self.current_candidates = list(set(new_candidates))
        print(f"\n🎯 After filtering by nearest object '{nearest_obj}', "
              f"{len(self.current_candidates)} candidate frames remain.")
        for s, f in self.current_candidates[:5]:
            print(f" - Scene: {s}, Frame: {f}")

    # --------------------------------------------------------
    # Step 3: Iteratively ask about nearby motifs
    # --------------------------------------------------------
    def find_unique_neighbor_motif(self, radius=1.5):
        """
        Returns the most unique nearby motif among remaining frames.
        """
        candidate_neighbors = []

        for scene, frame in self.current_candidates:
            motifs = self.frames[(scene, frame)]

            # Find motifs involving objects the user mentioned
            main_motifs = [
                m for m in motifs
                if any(obj in [m["relation"]["subject"], m["relation"]["object"]] for obj in self.objects_seen)
            ]

            # Find nearby motifs
            for m in main_motifs:
                m_mid = np.array(m["relation"]["geometry"]["midpoint_world"])
                for n in motifs:
                    if n == m:
                        continue
                    n_mid = np.array(n["relation"]["geometry"]["midpoint_world"])
                    dist = np.linalg.norm(m_mid - n_mid)
                    if dist < radius:
                        candidate_neighbors.append((scene, frame, m, n, dist))

        if not candidate_neighbors:
            return None

        # Compute frequency of each motif across all scenes
        freq = {}
        for _, _, _, n, _ in candidate_neighbors:
            key = (n["relation"]["subject"], n["relation"]["predicate"], n["relation"]["object"])
            freq[key] = freq.get(key, 0) + 1

        # Pick motif with lowest frequency (most unique)
        return min(candidate_neighbors, key=lambda x: freq[
            (x[3]["relation"]["subject"], x[3]["relation"]["predicate"], x[3]["relation"]["object"])
        ])

    def iterative_motif_filter(self, radius=1.5, target_count=3):
        """
        Iteratively asks user about nearby motifs until only a few candidates remain.
        User can answer 'yes', 'no', or 'unsure'.
        """
        while len(self.current_candidates) > target_count:
            unique = self.find_unique_neighbor_motif(radius=radius)
            if not unique:
                print("⚠️ No more discriminative motifs found.")
                break

            scene, frame, main_m, neighbor, _ = unique
            subj, pred, obj = neighbor["relation"]["subject"], neighbor["relation"]["predicate"], neighbor["relation"]["object"]

            print(f"\n🤖 Do you also see '{subj} {pred} {obj}' nearby? (yes/no/unsure)")
            answer = input("You: ").strip().lower()

            if answer.startswith("y"):
                # Keep only this frame
                self.current_candidates = [(scene, frame)]
                print(f"✅ Great! That uniquely identifies frame {frame} in scene {scene}.")
            elif answer.startswith("n"):
                # Remove frames that contain that motif
                bad_key = (subj, pred, obj)
                new_candidates = []
                for s, f in self.current_candidates:
                    motifs = self.frames[(s, f)]
                    if not any(
                        (m["relation"]["subject"], m["relation"]["predicate"], m["relation"]["object"]) == bad_key
                        for m in motifs
                    ):
                        new_candidates.append((s, f))
                self.current_candidates = new_candidates
                print(f"❌ Excluded frames containing {bad_key}. Remaining: {len(self.current_candidates)} \nFrames: {self.current_candidates}")
            else:
                # Unsure — skip this question, no filtering
                print("🤷 Skipping this motif... no changes made.")

            if len(self.current_candidates) <= target_count:
                break

    # --------------------------------------------------------
    # Run the full pipeline
    # --------------------------------------------------------
    def run(self):
        print("🤖 Hello! I’ll help you locate your camera pose in the scene.")
        self.ask_objects_seen()
        self.ask_nearest_object()
        self.iterative_motif_filter()
        print("\n✨ Done! Remaining candidate frames:")
        for s, f in self.current_candidates:
            print(f" - Scene: {s}, Frame: {f}")

if __name__ == "__main__":
    bot = SceneChatbot("src/motif_creation/sample_motif_index.json")
    bot.run()