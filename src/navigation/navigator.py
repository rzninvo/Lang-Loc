import streamlit as st

def next_image():
    if not st.session_state.scene_list:
        return
    scene = st.session_state.scene_list[st.session_state.current_scene_index]
    n = len(st.session_state.scene_to_files.get(scene, []))
    if n == 0:
        return
    st.session_state.current_image_index += 1
    if st.session_state.current_image_index >= n:
        st.session_state.current_image_index = 0
        st.session_state.current_scene_index = (st.session_state.current_scene_index + 1) % len(st.session_state.scene_list)

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

