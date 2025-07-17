"""
Image Generator Script for ScanNet Scenes

This script loads a ScanNet scene mesh and generates multiple rendered views
from different camera angles using Open3D's offscreen rendering. Camera and
output settings are configurable through a YAML file.

Usage:
    python src/image_generator.py <scene_id> [--config path/to/config.yaml]

Arguments:
    scene_id        The ID of the scene to render (e.g., scene0000_00)
    --config        Optional path to a YAML config file (default: config/default.yaml)

Expected directory structure:
    data/scans/<scene_id>/<scene_id>_vh_clean_2.ply
    data/scans/<scene_id>/output_images/view_*.png
"""

import open3d as o3d
import numpy as np
import os
import sys
import subprocess
from tqdm import tqdm
from src.utils.config_loader import load_config


def render_scene(scene_id, config):
    """
    Render multiple views of a ScanNet scene mesh using offscreen Open3D rendering.

    Args:
        scene_id (str): ID of the scene (e.g., 'scene0000_00')
        config (dict): Parsed configuration dictionary
    """
    render_cfg = config['render']
    path_cfg = config['paths']

    width = render_cfg['resolution']['width']
    height = render_cfg['resolution']['height']
    fov = render_cfg['fov']
    fov_type_str = render_cfg['fov_type'].lower()
    radius = render_cfg['radius']
    cam_height = render_cfg['height']
    num_views = render_cfg['num_views']
    up_vector = np.array(render_cfg['up_vector'])
    output_folder = render_cfg['output_folder']

    base_dir = os.path.join(path_cfg['base_data_dir'], "scans")
    base_dir = os.path.join(base_dir, scene_id)
    mesh_path = os.path.join(base_dir, f"{scene_id}_vh_clean_2.ply")

    if not os.path.exists(mesh_path):
        print(f"[INFO] Scene {scene_id} not found locally. Downloading...")
        subprocess.run([path_cfg['download_script'], scene_id], check=True)
        if not os.path.exists(mesh_path):
            print(f"[ERROR] Failed to download scene {scene_id}")
            sys.exit(1)

    print(f"[INFO] Rendering views for scene {scene_id}...")

    mesh = o3d.io.read_triangle_mesh(mesh_path)
    mesh.compute_vertex_normals()

    renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
    scene = renderer.scene

    if fov_type_str == "vertical":
        fov_type = o3d.visualization.rendering.Camera.FovType.Vertical
    elif fov_type_str == "horizontal":
        fov_type = o3d.visualization.rendering.Camera.FovType.Horizontal
    else:
        raise ValueError("Invalid fov_type in config: use 'vertical' or 'horizontal'")

    aspect_ratio = width / height
    scene.camera.set_projection(fov, aspect_ratio, 0.1, 100.0, fov_type)

    material = o3d.visualization.rendering.MaterialRecord()
    material.shader = "defaultLit"
    scene.add_geometry("mesh", mesh, material)

    center = mesh.get_center()
    output_dir = os.path.join(base_dir, output_folder)
    os.makedirs(output_dir, exist_ok=True)

    for i in tqdm(range(num_views), desc=f"Rendering {scene_id}"):
        angle = i * 2 * np.pi / num_views
        cam_pos = np.array([
            center[0] + radius * np.cos(angle),
            center[1] + radius * np.sin(angle),
            center[2] + cam_height
        ])

        scene.camera.look_at(center, cam_pos, up_vector)
        img = renderer.render_to_image()
        o3d.io.write_image(os.path.join(output_dir, f"view_{i+1}.png"), img)

    print(f"[INFO] Done. Rendered images saved to {output_dir}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate rendered images from ScanNet scene mesh")
    parser.add_argument("scene_id", type=str, help="Scene ID (e.g., scene0000_00)")
    parser.add_argument("--config", type=str, default="config/default.yaml", help="Path to config YAML file")

    args = parser.parse_args()

    config = load_config(args.config)
    render_scene(args.scene_id, config)
