import os
from src.utils.io_utils import load_json_file
import streamlit as st

def initialize_session_state(config):
    if 'current_scene_index' not in st.session_state:
        st.session_state.current_scene_index = 0
    if 'current_image_index' not in st.session_state:
        st.session_state.current_image_index = 0
    if 'annotations' not in st.session_state:
        st.session_state.annotations = load_json_file(config['paths']['annotations_file'])
    if 'uninterpretable_images' not in st.session_state:
        st.session_state.uninterpretable_images = load_json_file(config['paths']['uninterpretable_file'])
    if 'scene_list' not in st.session_state:
        st.session_state.scene_list, st.session_state.num_view = load_dataset_structure(config['paths']['dataset_path'], config['render']['output_folder'])

def load_dataset_structure(dataset_path, output_folder):
    """
    Loads the list of scenes that have the expected keyframe output structure:
    - color/   (with at least one JPG or PNG)
    - depth/   (with PNG files matching color filenames)
    - pose/    (with TXT files matching color filenames)
    - label/   (optional, may be empty)

    Args:
        dataset_path (str): Base dataset path (e.g. "data/scans")
        output_folder (str): Name of the folder inside each scene (e.g. "output")

    Returns:
        list[str]: Sorted list of scene IDs that match the structure.
        num_views (int): Number of views per scene, or 0 if no valid scenes found.
    """
    if not os.path.exists(dataset_path):
        return [], 0

    scenes = []
    view_counts = []

    for scene_name in os.listdir(dataset_path):
        scene_path = os.path.join(dataset_path, scene_name, output_folder)
        color_dir = os.path.join(scene_path, "color")
        depth_dir = os.path.join(scene_path, "depth")
        pose_dir = os.path.join(scene_path, "pose")

        if not (os.path.isdir(color_dir) and os.path.isdir(depth_dir) and os.path.isdir(pose_dir)):
            continue

        color_files = sorted([
            f for f in os.listdir(color_dir)
            if f.lower().endswith((".jpg", ".png"))
        ])
        if not color_files:
            continue

        # Only count frames that have matching depth & pose
        valid_files = [
            f for f in color_files
            if os.path.exists(os.path.join(depth_dir, f"{os.path.splitext(f)[0]}.png"))
            and os.path.exists(os.path.join(pose_dir, f"{os.path.splitext(f)[0]}.txt"))
        ]

        if valid_files:
            scenes.append(scene_name)
            view_counts.append(len(valid_files))

    # If scenes have variable view counts, use min/max logic
    num_views = min(view_counts) if view_counts else 0
    return sorted(scenes), num_views

