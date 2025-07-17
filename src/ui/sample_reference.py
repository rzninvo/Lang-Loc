import streamlit as st
from PIL import Image
import os

def render_sample_reference(dataset_path, output_folder, sample_scene, sample_view, sample_description):
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Sample Image")
        path = os.path.join(dataset_path, sample_scene, output_folder, f"{sample_view}.png")
        if os.path.exists(path):
            st.image(Image.open(path), use_container_width=True)
        else:
            st.error("Sample image not found")

    with col2:
        st.subheader("Sample Description")
        st.info(sample_description)
        st.subheader("Annotation Guidelines")
        st.markdown("""
        A good description should include:
        - Objects visible in the scene
        - Spatial relationships between objects
        - Colors and materials
        - Overall scene context
        - Any notable features or details
        
        **Note:** If an image is too blurry, dark, corrupted, or otherwise impossible to interpret, use the "Uninterpretable" button instead of trying to describe it.
        """)
