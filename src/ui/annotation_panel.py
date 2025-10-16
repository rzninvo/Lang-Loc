import streamlit as st
from PIL import Image
import os
from src.utils.io_utils import is_image_uninterpretable

def render_annotation_panel(
    dataset_key: str,
    dataset_path: str,
    output_folder: str,
    save_annotation_fn,
    mark_uninterpretable_fn,
    next_image_fn,
    previous_image_fn,
):
    """
    Renders the annotation panel for a given dataset (ScanNet or 3RScan).

    Args:
        dataset_key (str): session_state key ("scannet" or "rscan")
        dataset_path (str): dataset root
        output_folder (str): output subfolder name
        save_annotation_fn (callable): function to save annotation
        mark_uninterpretable_fn (callable): function to mark uninterpretable
        next_image_fn (callable): move to next image
        previous_image_fn (callable): move to previous image
    """

    if dataset_key not in st.session_state:
        st.error(f"❌ No session state for dataset: {dataset_key}")
        return

    data = st.session_state[dataset_key]

    if not data.get("scene_list"):
        st.error("❌ No scenes found!")
        st.info("Check dataset path and output structure.")
        return

    scenes = data["scene_list"]
    scene_to_files = data["scene_to_files"]
    current_scene = scenes[data["current_scene_index"]]
    files = scene_to_files.get(current_scene, [])

    if not files:
        st.error(f"❌ No valid images found for scene: {current_scene}")
        return

    # Clamp index within this scene
    data["current_image_index"] = min(
        data["current_image_index"],
        max(0, len(files) - 1)
    )

    current_filename = files[data["current_image_index"]]
    color_dir = os.path.join(dataset_path, current_scene, output_folder, "color")
    current_image_path = os.path.join(color_dir, current_filename)
    file_id = os.path.splitext(current_filename)[0]

    # --- Progress ---
    prior = sum(len(scene_to_files.get(s, [])) for s in scenes[:data["current_scene_index"]])
    current_global_index = prior + data["current_image_index"] + 1
    total_images = data["total_images"] if data["total_images"] else 1

    st.progress(current_global_index / total_images)
    st.write(f"**Progress:** {current_global_index} / {total_images} images")

    col1, col2, col3 = st.columns(3)
    with col1: st.metric("Scene", current_scene)
    with col2: st.metric("File", current_filename)
    with col3:
        if is_image_uninterpretable(current_scene, file_id, data["uninterpretable_images"]):
            st.error("⚠️ Uninterpretable")
        else:
            st.info("✅ Interpretable")

    if os.path.exists(current_image_path):
        st.image(Image.open(current_image_path),
                 caption=f"{current_scene} - {current_filename}",
                 use_container_width=True)
    else:
        st.error(f"Image not found: {current_image_path}")
        return

    # --- Annotation box ---
    is_unint = is_image_uninterpretable(current_scene, file_id, data["uninterpretable_images"])
    description = st.text_area(
        "📝 Describe this image:",
        height=100,
        key=f"description_{dataset_key}_{current_scene}_{current_filename}"
    )

    if is_unint:
        st.warning("⚠️ This image has been marked as uninterpretable. "
                   "If you believe it *is* interpretable, you can still annotate it.")

    # --- Buttons ---
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("⬅️ Previous", use_container_width=True, key=f"prev_{dataset_key}"):
            previous_image_fn()
            st.rerun()

    with col2:
        if st.button("💾 Save & Next", use_container_width=True, type="primary", key=f"save_{dataset_key}"):
            if description.strip():
                if save_annotation_fn(description):
                    st.success("Annotation saved!")
                    next_image_fn()
                    st.rerun()
                else:
                    st.error("Failed to save annotation")
            else:
                st.error("Please enter a description before saving")

    with col3:
        if st.button("❌ Uninterpretable", use_container_width=True, key=f"unint_{dataset_key}"):
            if mark_uninterpretable_fn():
                st.success("Marked as uninterpretable!")
                next_image_fn()
                st.rerun()
            else:
                st.error("Failed to mark image")

    with col4:
        if st.button("⏭️ Skip", use_container_width=True, key=f"skip_{dataset_key}"):
            next_image_fn()
            st.rerun()
