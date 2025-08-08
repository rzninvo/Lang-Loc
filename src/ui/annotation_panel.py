import streamlit as st
from PIL import Image
import os
from src.utils.io_utils import is_image_uninterpretable

def render_annotation_panel(
    dataset_path,
    output_folder,
    images_per_scene,
    save_annotation_fn,
    mark_uninterpretable_fn,
    next_image_fn,
    previous_image_fn,
):
    # Check if we have any scenes available
    if not st.session_state.scene_list:
        st.error("❌ No scenes found!")
        st.info(f"Please ensure that:")
        st.info(f"1. The dataset path exists: `{dataset_path}`")
        st.info(f"2. Scenes are downloaded and rendered in the expected structure:")
        st.info(f"   - Each scene should have an `{output_folder}` directory")
        st.info(f"   - Each scene should have {images_per_scene} rendered images (any filenames)")
        st.info(f"3. You can run the sampling script to create a manageable dataset.")
        return

    current_scene = st.session_state.scene_list[st.session_state.current_scene_index]
    scene_color_dir = os.path.join(dataset_path, current_scene, output_folder, "color")

    # Get sorted list of available image files for this scene
    if not os.path.exists(scene_color_dir):
        st.error(f"❌ No color folder found: {scene_color_dir}")
        return

    scene_images = sorted([
        f for f in os.listdir(scene_color_dir)
        if f.lower().endswith((".jpg", ".png"))
    ])

    if not scene_images:
        st.error(f"❌ No images found in {scene_color_dir}")
        return

    # Ensure we don't go out of bounds
    total_images_for_scene = len(scene_images)
    if st.session_state.current_image_index >= total_images_for_scene:
        st.session_state.current_image_index = 0

    current_filename = scene_images[st.session_state.current_image_index]
    current_image_path = os.path.join(scene_color_dir, current_filename)
    file_id = os.path.splitext(current_filename)[0]  # without .jpg/.png

    # Progress bar (based on total number of scenes * images_per_scene assumption)
    total_images = len(st.session_state.scene_list) * images_per_scene
    current_image_number = st.session_state.current_scene_index * images_per_scene + st.session_state.current_image_index + 1

    st.progress(current_image_number / total_images)
    st.write(f"**Progress:** {current_image_number} / {total_images} images")

    col1, col2, col3 = st.columns(3)
    with col1: st.metric("Scene", current_scene)
    with col2: st.metric("File", current_filename)
    with col3:
        if is_image_uninterpretable(current_scene, file_id, st.session_state.uninterpretable_images):
            st.error("⚠️ Uninterpretable")
        else:
            st.info("✅ Interpretable")

    # Display image
    st.image(Image.open(current_image_path), caption=f"{current_scene} - {current_filename}", use_container_width=True)

    # Annotation box
    is_unint = is_image_uninterpretable(current_scene, file_id, st.session_state.uninterpretable_images)
    description = st.text_area(
        "📝 Describe this image:",
        height=100,
        disabled=is_unint,
        key=f"description_{current_scene}_{current_filename}"
    )

    if is_unint:
        st.warning("This image has been marked as uninterpretable. Use navigation buttons to move to another image.")

    # Navigation buttons
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("⬅️ Previous", use_container_width=True):
            previous_image_fn()
            st.rerun()

    with col2:
        if st.button("💾 Save & Next", use_container_width=True, type="primary", disabled=is_unint):
            if description.strip():
                if save_annotation_fn(description):
                    st.success("Annotation saved successfully!")
                    next_image_fn()
                    st.rerun()
                else:
                    st.error("Failed to save annotation")
            else:
                st.error("Please enter a description before saving")

    with col3:
        if st.button("❌ Uninterpretable", use_container_width=True):
            if mark_uninterpretable_fn():
                st.success("Image marked as uninterpretable!")
                next_image_fn()
                st.rerun()
            else:
                st.error("Failed to mark image")

    with col4:
        if st.button("⏭️ Skip", use_container_width=True):
            next_image_fn()
            st.rerun()
