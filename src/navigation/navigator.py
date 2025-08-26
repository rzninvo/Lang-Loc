import streamlit as st
import random

def next_image():
    if not st.session_state.scene_list:
        return

    scene_to_files = st.session_state.scene_to_files
    scenes = st.session_state.scene_list
    current_scene = st.session_state.scene_list[st.session_state.current_scene_index]
    current_file = st.session_state.scene_to_files[current_scene][st.session_state.current_image_index]

    # push current state to history before moving
    st.session_state.history.append((st.session_state.current_scene_index,
                                     st.session_state.current_image_index))

    # pick a random new scene (can include current one too if you want)
    candidates = [
        (si, fi)
        for si, s in enumerate(scenes)
        for fi in range(len(scene_to_files.get(s, [])))
        if not (si == st.session_state.current_scene_index and fi == st.session_state.current_image_index)
    ]

    if not candidates:
        return

    new_scene_index, new_image_index = random.choice(candidates)
    st.session_state.current_scene_index = new_scene_index
    st.session_state.current_image_index = new_image_index


def previous_image():
    if not st.session_state.history:
        return  # nothing to go back to

    # pop last visited image
    prev_scene_index, prev_image_index = st.session_state.history.pop()
    st.session_state.current_scene_index = prev_scene_index
    st.session_state.current_image_index = prev_image_index

