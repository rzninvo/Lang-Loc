"""
Streamlit annotation tool for 3D scene dataset creation.

This application provides a web interface for annotating images from
ScanNet and 3RScan datasets with natural language descriptions.

Usage:
    streamlit run app/app.py

Note:
    Run from the repository root directory to ensure proper imports.
"""
import os

import streamlit as st

from src.utils.config_loader import load_config
from src.utils.io_utils import save_json_file, save_annotation, mark_uninterpretable
from src.state.session import initialize_session_state, load_dataset_structure
from src.ui.annotation_panel import render_annotation_panel
from src.ui.sample_reference import render_sample_reference
from src.ui.instructions import render_instructions
from src.ui.admin_panels import render_admin_tables
from src.ui.sidebar import render_sidebar
from src.navigation.navigator import next_image, previous_image

# ---------------------------------------------------------------------
# Helpers to read config with graceful fallbacks (backward compatible)
# ---------------------------------------------------------------------
def get_path(cfg, *keys, default=None):
    cur = cfg
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur

def ensure_parent_dir(path):
    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)

# --- Load config ---
CONFIG = load_config()

# Dataset roots
SCANNET_PATH = get_path(CONFIG, "paths", "scannet_dataset_path")
RSCAN_PATH   = get_path(CONFIG, "paths", "3rscan_dataset_path")
OUTPUT_FOLDER = get_path(CONFIG, "render", "output_folder", default="output")

# Per-dataset annotation files (fall back to single file or derive a suffixed path)
ann_single = get_path(CONFIG, "paths", "annotations_file")
unint_single = get_path(CONFIG, "paths", "uninterpretable_file")

ANN_SCANNET = get_path(CONFIG, "paths", "annotations_file_scannet", default=ann_single)
UNINT_SCANNET = get_path(CONFIG, "paths", "uninterpretable_file_scannet", default=unint_single)

ANN_RSCAN = get_path(CONFIG, "paths", "annotations_file_3rscan",
                     default=(ann_single.replace(".json", "_3rscan.json") if ann_single else None))
UNINT_RSCAN = get_path(CONFIG, "paths", "uninterpretable_file_3rscan",
                       default=(unint_single.replace(".json", "_3rscan.json") if unint_single else None))

# Sample references (optional per-dataset overrides)
S_SCENE_SCAN   = get_path(CONFIG, "ui", "sample_scene", default="")
S_VIEW_SCAN    = get_path(CONFIG, "ui", "sample_view", default="")
S_DESC_SCAN    = get_path(CONFIG, "ui", "sample_description", default="")

S_SCENE_RSCAN  = get_path(CONFIG, "ui", "sample_scene_3rscan", default=S_SCENE_SCAN)
S_VIEW_RSCAN   = get_path(CONFIG, "ui", "sample_view_3rscan", default=S_VIEW_SCAN)
S_DESC_RSCAN   = get_path(CONFIG, "ui", "sample_description_3rscan",
                          default=(S_DESC_SCAN or "Describe the scene with objects, relations, materials, context."))

# Make sure result folders exist
for p in [ANN_SCANNET, UNINT_SCANNET, ANN_RSCAN, UNINT_RSCAN]:
    ensure_parent_dir(p)

# --- Streamlit page setup ---
st.set_page_config(page_title="Image Annotation Tool", page_icon="🖼️", layout="wide")
st.title("📷 Image Annotation Tool")

# ---------------------------------------------------------------------
# Small helpers to route actions into the right dataset "bucket"
# ---------------------------------------------------------------------
def handle_save_annotation(description, key, annotations_file):
    data = st.session_state[key]
    scene = data["scene_list"][data["current_scene_index"]]
    files = data["scene_to_files"][scene]
    file_id = os.path.splitext(files[data["current_image_index"]])[0]
    data["annotations"] = save_annotation(description, data["annotations"], scene, file_id)
    return save_json_file(data["annotations"], annotations_file)

def handle_mark_uninterpretable(key, uninterpretable_file):
    data = st.session_state[key]
    scene = data["scene_list"][data["current_scene_index"]]
    files = data["scene_to_files"][scene]
    file_id = os.path.splitext(files[data["current_image_index"]])[0]
    data["uninterpretable_images"] = mark_uninterpretable(scene, file_id, data["uninterpretable_images"])
    return save_json_file(data["uninterpretable_images"], uninterpretable_file)


# ---------------------------------------------------------------------
# Unified Dataset Tab Renderer
# ---------------------------------------------------------------------
def render_dataset_tab(
    session_key: str,
    dataset_name: str,
    dataset_path: str,
    annotations_file: str,
    uninterpretable_file: str,
    sample_scene: str,
    sample_view: str,
    sample_description: str,
):
    """
    Render the annotation interface for a dataset tab.

    This function consolidates the common UI pattern used for both ScanNet
    and 3RScan tabs, reducing code duplication.

    Args:
        session_key: Session state key (e.g., 'scannet', 'rscan').
        dataset_name: Display name for error messages (e.g., 'ScanNet', '3RScan').
        dataset_path: Path to the dataset root directory.
        annotations_file: Path to save annotations JSON.
        uninterpretable_file: Path to save uninterpretable images JSON.
        sample_scene: Sample scene ID for reference.
        sample_view: Sample view/frame ID for reference.
        sample_description: Sample description text for reference.
    """
    if not dataset_path or not os.path.isdir(dataset_path):
        st.error(f"No {dataset_name} data found at: {dataset_path or '(unset)'}")
        return

    # Initialize session state for this dataset
    initialize_session_state(
        key=session_key,
        dataset_path=dataset_path,
        output_folder=OUTPUT_FOLDER,
        annotations_file=annotations_file,
        uninterpretable_file=uninterpretable_file,
    )

    # Render sidebar with dataset stats
    render_sidebar(
        key=session_key,
        dataset_path=dataset_path,
        total_scenes=len(st.session_state[session_key]["scene_list"]),
        total_images=st.session_state[session_key]["total_images"],
        annotation_count=len(st.session_state[session_key]["annotations"]),
        uninterpretable_count=len(st.session_state[session_key]["uninterpretable_images"]),
    )

    # Render sample reference
    render_sample_reference(
        dataset_path=dataset_path,
        output_folder=OUTPUT_FOLDER,
        sample_scene=sample_scene,
        sample_view=sample_view,
        sample_description=sample_description,
    )

    # Render annotation panel with action handlers
    render_annotation_panel(
        dataset_key=session_key,
        dataset_path=dataset_path,
        output_folder=OUTPUT_FOLDER,
        save_annotation_fn=lambda desc: handle_save_annotation(desc, session_key, annotations_file),
        mark_uninterpretable_fn=lambda: handle_mark_uninterpretable(session_key, uninterpretable_file),
        next_image_fn=lambda: next_image(session_key),
        previous_image_fn=lambda: previous_image(session_key),
    )

    # Render admin tables and instructions
    render_admin_tables(
        annotations=st.session_state[session_key]["annotations"],
        uninterpretable_images=st.session_state[session_key]["uninterpretable_images"],
    )
    render_instructions()


# ---------------------------------------------------------------------
# Tabs: ScanNet and 3RScan
# ---------------------------------------------------------------------
tab_scannet, tab_rscan = st.tabs(["ScanNet", "3RScan"])

with tab_scannet:
    render_dataset_tab(
        session_key="scannet",
        dataset_name="ScanNet",
        dataset_path=SCANNET_PATH,
        annotations_file=ANN_SCANNET,
        uninterpretable_file=UNINT_SCANNET,
        sample_scene=S_SCENE_SCAN,
        sample_view=S_VIEW_SCAN,
        sample_description=S_DESC_SCAN,
    )

with tab_rscan:
    render_dataset_tab(
        session_key="rscan",
        dataset_name="3RScan",
        dataset_path=RSCAN_PATH,
        annotations_file=ANN_RSCAN,
        uninterpretable_file=UNINT_RSCAN,
        sample_scene=S_SCENE_RSCAN,
        sample_view=S_VIEW_RSCAN,
        sample_description=S_DESC_RSCAN,
    )
