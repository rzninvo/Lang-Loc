import os
import json
import streamlit as st
from datetime import datetime
from pathlib import Path

from src.utils.camera_utils import load_camera_poses_json


def load_json_file(path):
    if os.path.exists(path):
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except Exception:  
            return []
    return []


def load_camera_poses(scene_index, output_folder, base_dir="data/scans", pose_filename="camera_pose.json"):
    """
    Load camera poses for a scene from the output JSON file.

    This is a wrapper around load_camera_poses_json for backward compatibility
    with the existing Streamlit UI code.

    Args:
        scene_index: Scene identifier (e.g., 'scene0000_00').
        output_folder: Output subdirectory name (e.g., 'output').
        base_dir: Base directory containing scene folders.
        pose_filename: Name of the pose JSON file.

    Returns:
        Dictionary mapping frame IDs to 4x4 pose matrices (as nested lists).
    """
    scene_path = Path(base_dir) / scene_index
    return load_camera_poses_json(scene_path, output_folder, pose_filename)

def save_json_file(data, path):
    try:
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        return True
    except Exception as e:
        st.error(f"Error saving to {path}: {str(e)}")
        return False

def save_annotation(description, annotations, scene_index, image_index, output_folder="output"):
    pose_dict = load_camera_poses(scene_index, output_folder)
    pose_matrix = pose_dict.get(f"{image_index}", None)
    annotation = {
        'scene_index': scene_index,
        'image_index': f"{image_index}",
        'scene_pose': pose_matrix,
        'description': description,
        'timestamp': datetime.now().isoformat()
    }
    annotations.append(annotation)
    return annotations

def mark_uninterpretable(scene_index, image_index, uninterpretable_images, output_folder="output"):
    pose_dict = load_camera_poses(scene_index, output_folder)
    pose_matrix = pose_dict.get(f"{image_index}", None)

    for item in uninterpretable_images:
        if item['scene_index'] == scene_index and item['image_index'] == f"{image_index}":
            return uninterpretable_images  # Already marked

    uninterpretable_entry = {
        'scene_index': scene_index,
        'image_index': f"{image_index}",
        'scene_pose': pose_matrix,
        'timestamp': datetime.now().isoformat(),
        'reason': 'marked_as_uninterpretable'
    }
    uninterpretable_images.append(uninterpretable_entry)
    return uninterpretable_images

def is_image_uninterpretable(scene_index, image_index, uninterpretable_images):
    for item in uninterpretable_images:
        if item['scene_index'] == scene_index and item['image_index'] == f"{image_index}":
            return True
    return False
