import streamlit as st
from PIL import Image
import os

def render_sample_reference(dataset_path, output_folder, sample_scene, sample_view, sample_description):
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Example Image")
        # Try different possible paths for the sample image
        possible_paths = [
            os.path.join(dataset_path, sample_scene, output_folder, "color", f"{sample_view}.jpg"),
            os.path.join(dataset_path, sample_scene, output_folder, f"{sample_view}.jpg"),
            os.path.join(dataset_path, sample_scene, output_folder, f"{sample_view}.png"),
        ]
        
        sample_image_path = None
        for path in possible_paths:
            if os.path.exists(path):
                sample_image_path = path
                break
        
        if sample_image_path:
            st.image(Image.open(sample_image_path), use_container_width=True)
        else:
            st.error(f"Example image not found. Tried paths: {possible_paths}")

    with col2:
        st.subheader("Example Description")
        st.info(sample_description)
        st.subheader("Annotation Guidelines")
        st.markdown("""
        A good description should include:
        - Objects visible in the image
        - Spatial relationships between objects
        - Colors and materials
        - Overall image context
        - Any notable features or details
        
        **Note:** If an image is too blurry, dark, corrupted, or otherwise impossible to interpret, use the "Uninterpretable" button instead of trying to describe it.
        """)
