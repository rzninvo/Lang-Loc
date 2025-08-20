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
    if 'scene_list' not in st.session_state or 'scene_to_files' not in st.session_state:
        scenes, scene_to_files, total = load_dataset_structure(
            config['paths']['dataset_path'],
            config['render']['output_folder']
        )
        st.session_state.scene_list = scenes
        st.session_state.scene_to_files = scene_to_files
        st.session_state.total_images = total

def load_dataset_structure(dataset_path, output_folder):
    """
    Builds a mapping: scene -> sorted list of valid frame filenames (jpg/png)
    A frame is valid only if it has matching depth(.png) and pose(.txt).

    Returns:
        scenes (list[str]): sorted scene IDs
        scene_to_files (dict[str, list[str]]): filenames per scene
        total_images (int): sum of all frames across all scenes
    """
    if not os.path.exists(dataset_path):
        return [], {}, 0

    scene_to_files = {}
    for scene_name in os.listdir(dataset_path):
        scene_path = os.path.join(dataset_path, scene_name, output_folder)
        color_dir = os.path.join(scene_path, "color")
        depth_dir = os.path.join(scene_path, "depth")
        pose_dir  = os.path.join(scene_path, "pose")

        if not (os.path.isdir(color_dir) and os.path.isdir(depth_dir) and os.path.isdir(pose_dir)):
            continue

        color_files = sorted(
            f for f in os.listdir(color_dir)
            if f.lower().endswith((".jpg", ".png"))
        )
        valid_files = [
            f for f in color_files
            if os.path.exists(os.path.join(depth_dir, f"{os.path.splitext(f)[0]}.png"))
            and os.path.exists(os.path.join(pose_dir,  f"{os.path.splitext(f)[0]}.txt"))
        ]
        if valid_files:
            scene_to_files[scene_name] = valid_files

    scenes = sorted(scene_to_files.keys())
    total_images = sum(len(v) for v in scene_to_files.values())
    return scenes, scene_to_files, total_images
