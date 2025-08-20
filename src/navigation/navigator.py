import streamlit as st
import random

def next_image():
    if not st.session_state.scene_list:
        return

    scene_to_files = st.session_state.scene_to_files
    scenes = st.session_state.scene_list
    current = st.session_state.current_scene_index

    # candidate scenes = all others with at least one file
    candidates = [
        i for i, s in enumerate(scenes)
        if i != current and len(scene_to_files.get(s, [])) > 0
    ]

    if not candidates:
        # fallback: if no other scenes with images, stay on current scene and reset to first image
        st.session_state.current_image_index = 0
        return

    st.session_state.current_scene_index = random.choice(candidates)
    st.session_state.current_image_index = 0

def previous_image():
    if not st.session_state.scene_list:
        return
    scene = st.session_state.scene_list[st.session_state.current_scene_index]
    n = len(st.session_state.scene_to_files.get(scene, []))
    if n == 0:
        return
    st.session_state.current_image_index -= 1
    if st.session_state.current_image_index < 0:
        st.session_state.current_scene_index = (st.session_state.current_scene_index - 1) % len(st.session_state.scene_list)
        prev_scene = st.session_state.scene_list[st.session_state.current_scene_index]
        st.session_state.current_image_index = max(0, len(st.session_state.scene_to_files.get(prev_scene, [])) - 1)

