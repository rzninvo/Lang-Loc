import streamlit as st
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils.config_loader import load_config
from src.utils.io_utils import (
    save_json_file,
    save_annotation,
    mark_uninterpretable,
)
from src.state.session import initialize_session_state
from src.ui.annotation_panel import render_annotation_panel
from src.ui.sample_reference import render_sample_reference
from src.ui.instructions import render_instructions
from src.ui.admin_panels import render_admin_tables
from src.ui.sidebar import render_sidebar
from src.navigation.navigator import next_image, previous_image

# --- Load config ---
CONFIG = load_config()
# Ensure results directory exists
results_dir = os.path.dirname(CONFIG["paths"]["annotations_file"])
os.makedirs(results_dir, exist_ok=True)

DATASET_PATH = CONFIG["paths"]["dataset_path"]
ANNOTATIONS_FILE = CONFIG["paths"]["annotations_file"]
UNINTERPRETABLE_FILE = CONFIG["paths"]["uninterpretable_file"]
SAMPLE_SCENE = CONFIG["ui"]["sample_scene"]
SAMPLE_VIEW = CONFIG["ui"]["sample_view"]
SAMPLE_DESCRIPTION = CONFIG["ui"]["sample_description"]
IMAGES_PER_SCENE = CONFIG["render"]["num_views"]
OUTPUT_FOLDER = CONFIG["render"]["output_folder"]

# --- Helpers ---
def handle_save_annotation(description):
    current_scene = st.session_state.scene_list[st.session_state.current_scene_index]
    image_index = st.session_state.current_image_index + 1
    st.session_state.annotations = save_annotation(
        description,
        st.session_state.annotations,
        current_scene,
        image_index
    )
    return save_json_file(st.session_state.annotations, ANNOTATIONS_FILE)

def handle_mark_uninterpretable():
    current_scene = st.session_state.scene_list[st.session_state.current_scene_index]
    image_index = st.session_state.current_image_index + 1
    st.session_state.uninterpretable_images = mark_uninterpretable(
        current_scene,
        image_index,
        st.session_state.uninterpretable_images
    )
    return save_json_file(st.session_state.uninterpretable_images, UNINTERPRETABLE_FILE)

# --- Streamlit page setup ---
st.set_page_config(
    page_title="ScanNet Image Annotation Tool",
    page_icon="🖼️",
    layout="wide"
)

# --- Session state ---
initialize_session_state(CONFIG)

# --- Sidebar ---
render_sidebar(
    dataset_path=DATASET_PATH,
    total_scenes=len(st.session_state.scene_list),
    total_images=len(st.session_state.scene_list) * IMAGES_PER_SCENE,
    annotation_count=len(st.session_state.annotations),
    uninterpretable_count=len(st.session_state.uninterpretable_images)
)

# --- Sample & Guidelines ---
render_sample_reference(
    dataset_path=DATASET_PATH,
    output_folder=OUTPUT_FOLDER,
    sample_scene=SAMPLE_SCENE,
    sample_view=SAMPLE_VIEW,
    sample_description=SAMPLE_DESCRIPTION
)

# --- Annotation Panel ---
render_annotation_panel(
    dataset_path=DATASET_PATH,
    output_folder=OUTPUT_FOLDER,
    images_per_scene=IMAGES_PER_SCENE,
    save_annotation_fn=lambda description: handle_save_annotation(description),
    mark_uninterpretable_fn = lambda: handle_mark_uninterpretable(),
    next_image_fn=lambda: next_image(IMAGES_PER_SCENE),
    previous_image_fn=lambda: previous_image(IMAGES_PER_SCENE)
)

# --- Admin Tables ---
render_admin_tables(
    annotations=st.session_state.annotations,
    uninterpretable_images=st.session_state.uninterpretable_images
)

# --- Instructions ---
render_instructions()
