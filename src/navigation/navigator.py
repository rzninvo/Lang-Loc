import streamlit as st

def next_image(IMAGES_PER_SCENE):
    if not st.session_state.scene_list:
        return
    st.session_state.current_image_index += 1
    if st.session_state.current_image_index >= IMAGES_PER_SCENE:
        st.session_state.current_image_index = 0
        st.session_state.current_scene_index = (st.session_state.current_scene_index + 1) % len(st.session_state.scene_list)

def previous_image(IMAGES_PER_SCENE):
    if not st.session_state.scene_list:
        return
    st.session_state.current_image_index -= 1
    if st.session_state.current_image_index < 0:
        st.session_state.current_scene_index = (st.session_state.current_scene_index - 1) % len(st.session_state.scene_list)
        st.session_state.current_image_index = IMAGES_PER_SCENE - 1
