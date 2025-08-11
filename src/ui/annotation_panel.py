import streamlit as st
from PIL import Image
import os
from src.utils.io_utils import is_image_uninterpretable

def render_annotation_panel(
    dataset_path,
    output_folder,
    save_annotation_fn,
    mark_uninterpretable_fn,
    next_image_fn,
    previous_image_fn,
):
    if not st.session_state.scene_list:
        st.error("❌ No scenes found!")
        st.info("Check dataset path and output structure.")
        return

    scenes = st.session_state.scene_list
    scene_to_files = st.session_state.scene_to_files
    current_scene = scenes[st.session_state.current_scene_index]
    files = scene_to_files.get(current_scene, [])

    if not files:
        st.error(f"❌ No valid images found for scene: {current_scene}")
        return

    # Clamp index within this scene
    st.session_state.current_image_index = min(
        st.session_state.current_image_index,
        max(0, len(files) - 1)
    )

    current_filename = files[st.session_state.current_image_index]
    color_dir = os.path.join(dataset_path, current_scene, output_folder, "color")
    current_image_path = os.path.join(color_dir, current_filename)
    file_id = os.path.splitext(current_filename)[0]

    # --- Progress over all scenes/images ---
    # cumulative images before current scene
    prior = 0
    for s in scenes[:st.session_state.current_scene_index]:
        prior += len(scene_to_files.get(s, []))
    current_global_index = prior + st.session_state.current_image_index + 1
    total_images = st.session_state.total_images if st.session_state.total_images else 1

    st.progress(current_global_index / total_images)
    st.write(f"**Progress:** {current_global_index} / {total_images} images")

    col1, col2, col3 = st.columns(3)
    with col1: st.metric("Scene", current_scene)
    with col2: st.metric("File", current_filename)
    with col3:
        if is_image_uninterpretable(current_scene, file_id, st.session_state.uninterpretable_images):
            st.error("⚠️ Uninterpretable")
        else:
            st.info("✅ Interpretable")

    st.image(Image.open(current_image_path), caption=f"{current_scene} - {current_filename}", use_container_width=True)

    is_unint = is_image_uninterpretable(current_scene, file_id, st.session_state.uninterpretable_images)
    description = st.text_area(
        "📝 Describe this image:",
        height=100,
        disabled=is_unint,
        key=f"description_{current_scene}_{current_filename}"
    )

    if is_unint:
        st.warning("This image has been marked as uninterpretable. Use navigation to move on.")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("⬅️ Previous", use_container_width=True):
            previous_image_fn()
            st.rerun()

    with col2:
        if st.button("💾 Save & Next", use_container_width=True, type="primary", disabled=is_unint):
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
        if st.button("❌ Uninterpretable", use_container_width=True):
            if mark_uninterpretable_fn():
                st.success("Marked as uninterpretable!")
                next_image_fn()
                st.rerun()
            else:
                st.error("Failed to mark image")

    with col4:
        if st.button("⏭️ Skip", use_container_width=True):
            next_image_fn()
            st.rerun()
