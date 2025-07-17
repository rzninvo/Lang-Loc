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
        st.session_state.scene_list = load_dataset_structure(config['paths']['dataset_path'], config['render']['num_views'])

def load_dataset_structure(dataset_path, num_views):
    if not os.path.exists(dataset_path):
        return []
    scenes = []
    for item in os.listdir(dataset_path):
        scene_path = os.path.join(dataset_path, item, 'output_images')
        if os.path.isdir(scene_path):
            views = [f"view_{i}.png" for i in range(1, num_views + 1)]
            if all(os.path.exists(os.path.join(scene_path, v)) for v in views):
                scenes.append(item)
    return sorted(scenes)