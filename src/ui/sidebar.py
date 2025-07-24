import streamlit as st

def render_sidebar(dataset_path, total_scenes, total_images, annotation_count, uninterpretable_count):
    st.sidebar.header("Dataset Information")
    if total_scenes > 0:
        st.sidebar.success("Dataset loaded successfully!")
        st.sidebar.info(f"Total scenes: {total_scenes}")
        st.sidebar.info(f"Total images: {total_images}")
        st.sidebar.info(f"Annotations completed: {annotation_count}")
        st.sidebar.info(f"Uninterpretable images: {uninterpretable_count}")
    else:
        st.sidebar.error("No valid scenes found.")
        st.sidebar.info(f"Expected path: {dataset_path}")