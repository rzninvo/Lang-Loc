import streamlit as st

def render_sidebar(key, dataset_path, total_scenes, total_images, annotation_count, uninterpretable_count):
    """
    Renders the sidebar with dataset information.
    Args:
        key (str): session_state key ("scannet" or "rscan")
        dataset_path (str): dataset root path
        total_scenes (int): total number of scenes
        total_images (int): total number of images
        annotation_count (int): number of completed annotations
        uninterpretable_count (int): number of uninterpretable images
    Returns:
        None
    """
    dataset_name = "Scannet" if key == "scannet" else "3RScan"
    st.sidebar.header(f"{dataset_name} Dataset Information")
    if total_scenes > 0:
        st.sidebar.success("Dataset loaded successfully!")
        st.sidebar.info(f"Total scenes: {total_scenes}")
        st.sidebar.info(f"Total images: {total_images}")
        st.sidebar.info(f"Annotations completed: {annotation_count}")
        st.sidebar.info(f"Uninterpretable images: {uninterpretable_count}")
    else:
        st.sidebar.error("No valid scenes found.")
        st.sidebar.info(f"Expected path: {dataset_path}")