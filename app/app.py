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
# Tabs: ScanNet and 3RScan
# ---------------------------------------------------------------------
tab_scannet, tab_rscan = st.tabs(["ScanNet", "3RScan"])

with tab_scannet:
    if not SCANNET_PATH or not os.path.isdir(SCANNET_PATH):
        st.error(f"No ScanNet data found at: {SCANNET_PATH or '(unset)'}")
    else:
        initialize_session_state(
            key="scannet",
            dataset_path=SCANNET_PATH,
            output_folder=OUTPUT_FOLDER,
            annotations_file=ANN_SCANNET,
            uninterpretable_file=UNINT_SCANNET,
        )

        render_sidebar(
            key="scannet",
            dataset_path=SCANNET_PATH,
            total_scenes=len(st.session_state["scannet"]["scene_list"]),
            total_images=st.session_state["scannet"]["total_images"],
            annotation_count=len(st.session_state["scannet"]["annotations"]),
            uninterpretable_count=len(st.session_state["scannet"]["uninterpretable_images"]),
        )

        render_sample_reference(
            dataset_path=SCANNET_PATH,
            output_folder=OUTPUT_FOLDER,
            sample_scene=S_SCENE_SCAN,
            sample_view=S_VIEW_SCAN,
            sample_description=S_DESC_SCAN,
        )

        render_annotation_panel(
            dataset_key="scannet",
            dataset_path=SCANNET_PATH,
            output_folder=OUTPUT_FOLDER,
            save_annotation_fn=lambda desc: handle_save_annotation(desc, "scannet", ANN_SCANNET),
            mark_uninterpretable_fn=lambda: handle_mark_uninterpretable("scannet", UNINT_SCANNET),
            next_image_fn=lambda: next_image("scannet"),
            previous_image_fn=lambda: previous_image("scannet"),
        )

        render_admin_tables(
            annotations=st.session_state["scannet"]["annotations"],
            uninterpretable_images=st.session_state["scannet"]["uninterpretable_images"],
        )
        render_instructions()

with tab_rscan:
    if not RSCAN_PATH or not os.path.isdir(RSCAN_PATH):
        st.error(f"No 3RScan data found at: {RSCAN_PATH or '(unset)'}")
    else:
        initialize_session_state(
            key="rscan",
            dataset_path=RSCAN_PATH,
            output_folder=OUTPUT_FOLDER,
            annotations_file=ANN_RSCAN,
            uninterpretable_file=UNINT_RSCAN,
        )

        render_sidebar(
            key="rscan",
            dataset_path=RSCAN_PATH,
            total_scenes=len(st.session_state["rscan"]["scene_list"]),
            total_images=st.session_state["rscan"]["total_images"],
            annotation_count=len(st.session_state["rscan"]["annotations"]),
            uninterpretable_count=len(st.session_state["rscan"]["uninterpretable_images"]),
        )

        render_sample_reference(
            dataset_path=RSCAN_PATH,
            output_folder=OUTPUT_FOLDER,
            sample_scene=S_SCENE_RSCAN,
            sample_view=S_VIEW_RSCAN,
            sample_description=S_DESC_RSCAN,
        )

        render_annotation_panel(
            dataset_key="rscan",
            dataset_path=RSCAN_PATH,
            output_folder=OUTPUT_FOLDER,
            save_annotation_fn=lambda desc: handle_save_annotation(desc, "rscan", ANN_RSCAN),
            mark_uninterpretable_fn=lambda: handle_mark_uninterpretable("rscan", UNINT_RSCAN),
            next_image_fn=lambda: next_image("rscan"),
            previous_image_fn=lambda: previous_image("rscan"),
        )

        render_admin_tables(
            annotations=st.session_state["rscan"]["annotations"],
            uninterpretable_images=st.session_state["rscan"]["uninterpretable_images"],
        )
        render_instructions()
