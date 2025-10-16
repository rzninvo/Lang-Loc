import os, random
import streamlit as st
from src.utils.io_utils import load_json_file

def initialize_session_state(
    key: str,
    dataset_path: str,
    output_folder: str,
    annotations_file: str,
    uninterpretable_file: str,
):
    """
    Initializes per-dataset state under st.session_state[key].

    Fields set:
      annotations, uninterpretable_images
      scene_list, scene_to_files, total_images
      current_scene_index, current_image_index
      history
    """
    if key not in st.session_state:
        st.session_state[key] = {}

    bucket = st.session_state[key]

    # persistent jsons (per dataset)
    if "annotations" not in bucket:
        bucket["annotations"] = load_json_file(annotations_file)
    if "uninterpretable_images" not in bucket:
        bucket["uninterpretable_images"] = load_json_file(uninterpretable_file)

    # dataset structure
    if "scene_list" not in bucket or "scene_to_files" not in bucket or "total_images" not in bucket:
        scenes, scene_to_files, total = load_dataset_structure(dataset_path, output_folder)
        bucket["scene_list"] = scenes
        bucket["scene_to_files"] = scene_to_files
        bucket["total_images"] = total

    # current indices
    if "current_scene_index" not in bucket:
        bucket["current_scene_index"] = random.randrange(len(bucket["scene_list"])) if bucket["scene_list"] else 0

    if "current_image_index" not in bucket:
        if bucket["scene_list"]:
            cur_scene = bucket["scene_list"][bucket["current_scene_index"]]
            files = bucket["scene_to_files"].get(cur_scene, [])
            bucket["current_image_index"] = random.randrange(len(files)) if files else 0
        else:
            bucket["current_image_index"] = 0

    if "history" not in bucket:
        bucket["history"] = []

def load_dataset_structure(dataset_path, output_folder):
    """
    Builds a mapping: scene -> sorted list of valid color frames (jpg/png)
    A frame is valid only if it has matching depth(.png) and pose(.txt).

    Returns:
        scenes (list[str])
        scene_to_files (dict[str, list[str]])
        total_images (int)
    """
    if not os.path.exists(dataset_path):
        return [], {}, 0

    scene_to_files = {}
    for scene_name in os.listdir(dataset_path):
        scene_out = os.path.join(dataset_path, scene_name, output_folder)
        color_dir = os.path.join(scene_out, "color")
        depth_dir = os.path.join(scene_out, "depth")
        pose_dir  = os.path.join(scene_out, "pose")
        if not (os.path.isdir(color_dir) and os.path.isdir(depth_dir) and os.path.isdir(pose_dir)):
            continue

        color_files = sorted(f for f in os.listdir(color_dir) if f.lower().endswith((".jpg", ".png")))
        valid = [
            f for f in color_files
            if (os.path.exists(os.path.join(depth_dir, f"{os.path.splitext(f)[0]}.png")) 
                or os.path.exists(os.path.join(depth_dir, f"{os.path.splitext(f)[0]}.pgm")))
            and os.path.exists(os.path.join(pose_dir,  f"{os.path.splitext(f)[0]}.txt"))
        ]
        if valid:
            scene_to_files[scene_name] = valid

    scenes = sorted(scene_to_files.keys())
    total_images = sum(len(v) for v in scene_to_files.values())
    return scenes, scene_to_files, total_images
