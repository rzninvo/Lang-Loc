import open3d as o3d
import numpy as np
import os
import sys
import subprocess

def render_scene(scene_id):
    base_dir = os.path.join("data", "scans", scene_id)
    mesh_path = os.path.join(base_dir, f"{scene_id}_vh_clean_2.ply")

    # Check if scene mesh exists
    if not os.path.exists(mesh_path):
        print(f"[INFO] Scene {scene_id} not found locally. Downloading...")
        subprocess.run(["./download_subset.sh", scene_id], check=True)
        if not os.path.exists(mesh_path):
            print(f"[ERROR] Failed to download scene {scene_id}")
            sys.exit(1)

    print(f"[INFO] Rendering views for scene {scene_id}...")
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    mesh.compute_vertex_normals()

    # Setup renderer
    renderer = o3d.visualization.rendering.OffscreenRenderer(1920, 1080)
    scene = renderer.scene
    # Set custom FoV
    fov = 75
    aspect_ratio = 1920 / 1080
    scene.camera.set_projection(fov, aspect_ratio, 0.1, 100.0,
                            o3d.visualization.rendering.Camera.FovType.Vertical)


    material = o3d.visualization.rendering.MaterialRecord()
    material.shader = "defaultLit"
    scene.add_geometry("mesh", mesh, material)

    center = mesh.get_center()
    radius = 1.5
    height = 1.0

    output_dir = os.path.join(base_dir, "output_images")
    os.makedirs(output_dir, exist_ok=True)

    for i in range(6):
        angle = i * 60.0 * np.pi / 180.0
        cam_pos = np.array([
            center[0] + radius * np.cos(angle),
            center[1] + radius * np.sin(angle),
            center[2] + height
        ])

        scene.camera.look_at(center, cam_pos, np.array([0, 0, 1]))
        img = renderer.render_to_image()
        o3d.io.write_image(os.path.join(output_dir, f"view_{i+1}.png"), img)

    print(f"[INFO] Done. Rendered images saved to {output_dir}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python render_scene.py <scene_id> (e.g., scene0000_00)")
        sys.exit(1)

    scene_id = sys.argv[1]
    render_scene(scene_id)
