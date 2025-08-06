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
        st.info(f"   - Each scene should have `{images_per_scene}` rendered images (view_1.jpg, view_2.jpg, etc.)")
        st.info(f"3. You can run the sampling script to create a manageable dataset:")
        st.info(f"   ```bash")
        st.info(f"   python scripts/sample_scannet_images.py scene0000_00 --sample_rate 150")
        st.info(f"   ```")
        return

    current_scene = st.session_state.scene_list[st.session_state.current_scene_index]
    image_index = st.session_state.current_image_index + 1
    current_view = f"view_{image_index}"
    total_images = len(st.session_state.scene_list) * images_per_scene
    current_image_number = st.session_state.current_scene_index * images_per_scene + image_index

    st.progress(current_image_number / total_images)
    st.write(f"**Progress:** {current_image_number} / {total_images} images")

    col1, col2, col3 = st.columns(3)
    with col1: st.metric("Scene", current_scene)
    with col2: st.metric("View", current_view)
    with col3:
        if is_image_uninterpretable(current_scene, image_index, st.session_state.uninterpretable_images):
            st.error("⚠️ Uninterpretable")
        else:
            st.info("✅ Interpretable")

    # Try different possible image paths
    possible_paths = [
        os.path.join(dataset_path, current_scene, output_folder, "color", f"{current_view}.jpg"),
        os.path.join(dataset_path, current_scene, output_folder, f"{current_view}.jpg"),
        os.path.join(dataset_path, current_scene, output_folder, f"{current_view}.png"),
    ]
    
    current_image_path = None
    for path in possible_paths:
        if os.path.exists(path):
            current_image_path = path
            break
    
    if current_image_path:
        st.image(Image.open(current_image_path), caption=f"{current_scene} - {current_view}", use_container_width=True)
    else:
        st.error(f"Image not found. Tried paths: {possible_paths}")

    st.subheader("📝 Describe this image:")
    is_unint = is_image_uninterpretable(current_scene, image_index, st.session_state.uninterpretable_images)
    description = st.text_area(
        "Enter your description here:",
        height=100,
        disabled=is_unint,
        key=f"description_{current_scene}_{current_view}"
    )

    if is_unint:
        st.warning("This image has been marked as uninterpretable. Use navigation buttons to move to another image.")

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
