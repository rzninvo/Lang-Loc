#!/usr/bin/env python3
import torch
import numpy as np
import open3d as o3d
from PIL import Image
import matplotlib.pyplot as plt
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    FoVPerspectiveCameras, MeshRenderer, MeshRasterizer,
    RasterizationSettings, HardPhongShader, PointLights,
    TexturesUV
)
from pytorch3d.renderer.cameras import look_at_view_transform


# ---------------- Load mesh from 3RScan -----------------
scene_path = "/home/rohamzn/UZH Uni/Master Project/Master-Project-Dataset-Creation/data/3RScan/7272e161-a01b-20f6-8b5a-0b97efeb6545"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

obj_path = f"{scene_path}/mesh.refined.v2.obj"
tex_path = f"{scene_path}/mesh.refined_0.png"

mesh = o3d.io.read_triangle_mesh(obj_path, True)

# Flip UV vertically (OpenGL→OpenCV)
uvs = np.asarray(mesh.triangle_uvs)
mesh.triangle_uvs = o3d.utility.Vector2dVector(uvs)

verts = np.asarray(mesh.vertices)
faces = np.asarray(mesh.triangles)
verts_uvs = np.asarray(mesh.triangle_uvs)
faces_uvs = np.arange(len(verts_uvs)).reshape(-1, 3)

# Load texture image
tex_img = np.array(Image.open(tex_path).convert("RGB")).astype(np.float32) / 255.0
tex_img = tex_img ** 2.2  # gamma correction

# Optional: flip axis to match OpenCV
verts[:, 1] *= -1  # Y flip; try Z instead if mirrored

# Build Meshes object
meshes = Meshes(
    verts=[torch.tensor(verts, dtype=torch.float32, device=device)],
    faces=[torch.tensor(faces, dtype=torch.int64, device=device)],
    textures=TexturesUV(
        maps=torch.tensor(tex_img, dtype=torch.float32, device=device).unsqueeze(0),
        faces_uvs=[torch.tensor(faces_uvs, dtype=torch.int64, device=device)],
        verts_uvs=[torch.tensor(verts_uvs, dtype=torch.float32, device=device)],
    )
)

# ---------------- Setup PyTorch3D renderer -----------------
R, T = look_at_view_transform(dist=2.0, elev=30, azim=40)
cameras = FoVPerspectiveCameras(R=R, T=T, device=device)

lights = PointLights(
    device=device,
    location=[[2.0, 2.0, 2.0]],
    ambient_color=((1.0, 1.0, 1.0),),
    diffuse_color=((0.0, 0.0, 0.0),),
    specular_color=((0.0, 0.0, 0.0),),
)

raster_settings = RasterizationSettings(
    image_size=(720, 1080),
    faces_per_pixel=1,
    blur_radius=0.0,
    perspective_correct=True,
    cull_backfaces=False,  # disable for full visibility
)

renderer = MeshRenderer(
    rasterizer=MeshRasterizer(raster_settings=raster_settings),
    shader=HardPhongShader(device=device, cameras=cameras, lights=lights),
)

# ---------------- Render and visualize -----------------
img = renderer(meshes, cameras=cameras)[0, ..., :3].cpu().numpy()
plt.figure(figsize=(10, 10))
plt.imshow(img)
plt.title("PyTorch3D Texture Validation")
plt.axis("off")
plt.show()
