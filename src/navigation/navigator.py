import streamlit as st
import random

def next_image(key: str):
    if key not in st.session_state:
        return
    bucket = st.session_state[key]
    if not bucket.get("scene_list"):
        return

    scenes = bucket["scene_list"]
    scene_to_files = bucket["scene_to_files"]
    cur_si = bucket["current_scene_index"]
    cur_scene = scenes[cur_si]
    cur_fi = bucket["current_image_index"]

    # push current to history
    bucket["history"].append((cur_si, cur_fi))

    # build candidate pool across all scenes
    candidates = [
        (si, fi)
        for si, s in enumerate(scenes)
        for fi in range(len(scene_to_files.get(s, [])))
        if not (si == cur_si and fi == cur_fi)
    ]
    if not candidates:
        return

    si, fi = random.choice(candidates)
    bucket["current_scene_index"] = si
    bucket["current_image_index"] = fi

def previous_image(key: str):
    if key not in st.session_state:
        return
    bucket = st.session_state[key]
    if not bucket.get("history"):
        return
    si, fi = bucket["history"].pop()
    bucket["current_scene_index"] = si
    bucket["current_image_index"] = fi
