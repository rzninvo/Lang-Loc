import os
import json
import streamlit as st
from datetime import datetime

def load_json_file(path):
    if os.path.exists(path):
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except Exception:  
            return []
    return []

def save_json_file(data, path):
    try:
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        return True
    except Exception as e:
        st.error(f"Error saving to {path}: {str(e)}")
        return False

def save_annotation(description, annotations, scene_index, image_index):
    scene_pose = (image_index - 1) * 60
    annotation = {
        'scene_index': scene_index,
        'image_index': f"view_{image_index}",
        'scene_pose': scene_pose,
        'description': description,
        'timestamp': datetime.now().isoformat()
    }
    annotations.append(annotation)
    return annotations

def mark_uninterpretable(scene_index, image_index, uninterpretable_images):
    scene_pose = (image_index - 1) * 60

    for item in uninterpretable_images:
        if item['scene_index'] == scene_index and item['image_index'] == f"view_{image_index}":
            return uninterpretable_images  # Already marked

    uninterpretable_entry = {
        'scene_index': scene_index,
        'image_index': f"view_{image_index}",
        'scene_pose': scene_pose,
        'timestamp': datetime.now().isoformat(),
        'reason': 'marked_as_uninterpretable'
    }
    uninterpretable_images.append(uninterpretable_entry)
    return uninterpretable_images

def is_image_uninterpretable(scene_index, image_index, uninterpretable_images):
    for item in uninterpretable_images:
        if item['scene_index'] == scene_index and item['image_index'] == f"view_{image_index}":
            return True
    return False
