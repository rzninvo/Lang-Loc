import streamlit as st
from PIL import Image
import os

def render_sample_reference(dataset_path, output_folder, sample_scene, sample_view, sample_description):
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Example Image")

        # Directory where sample images are stored
        scene_color_dir = os.path.join(dataset_path, sample_scene, output_folder, "color")

        sample_image_path = None

        # If sample_view is given as a filename directly (e.g., '001945.jpg')
        direct_path = os.path.join(scene_color_dir, sample_view)
        if os.path.exists(direct_path):
            sample_image_path = direct_path
        else:
            # If sample_view is something like 'view_1', try different possible paths
            possible_paths = [
                os.path.join(scene_color_dir, f"{sample_view}.jpg"),
                os.path.join(scene_color_dir, f"{sample_view}.png"),
                os.path.join(dataset_path, sample_scene, output_folder, f"{sample_view}.jpg"),
                os.path.join(dataset_path, sample_scene, output_folder, f"{sample_view}.png"),
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    sample_image_path = path
                    break

        # If no sample_view was found, just pick the first image in the folder
        if not sample_image_path and os.path.exists(scene_color_dir):
            image_files = sorted(
                [f for f in os.listdir(scene_color_dir) if f.lower().endswith((".jpg", ".png"))]
            )
            if image_files:
                sample_image_path = os.path.join(scene_color_dir, image_files[0])

        # Show the image or an error
        if sample_image_path:
            st.image(Image.open(sample_image_path), use_container_width=True)
        else:
            st.error(f"Example image not found in {scene_color_dir}")

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
        
        **Note:** If an image is too blurry, dark, corrupted, or otherwise impossible to interpret,
        use the "Uninterpretable" button instead of trying to describe it.
        """)
