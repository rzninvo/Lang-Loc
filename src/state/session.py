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
        st.session_state.scene_list = load_dataset_structure(config['paths']['dataset_path'], config['render']['output_folder'], config['render']['num_views'])

def load_dataset_structure(dataset_path, output_folder, num_views):
    if not os.path.exists(dataset_path):
        return []
    scenes = []
    for item in os.listdir(dataset_path):
        scene_path = os.path.join(dataset_path, item, output_folder)
        if os.path.isdir(scene_path):
            # Check for different possible image structures
            possible_image_dirs = [
                os.path.join(scene_path, "color"),  # New structure with color subdirectory
                scene_path  # Old structure with images directly in output folder
            ]
            
            for img_dir in possible_image_dirs:
                if os.path.isdir(img_dir):
                    # Check for both .jpg and .png extensions
                    views_jpg = [f"view_{i}.jpg" for i in range(1, num_views + 1)]
                    views_png = [f"view_{i}.png" for i in range(1, num_views + 1)]
                    
                    # Check if all jpg files exist
                    if all(os.path.exists(os.path.join(img_dir, v)) for v in views_jpg):
                        scenes.append(item)
                        break
                    # Check if all png files exist
                    elif all(os.path.exists(os.path.join(img_dir, v)) for v in views_png):
                        scenes.append(item)
                        break
    return sorted(scenes)
