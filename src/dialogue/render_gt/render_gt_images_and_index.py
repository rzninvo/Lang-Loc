#!/usr/bin/env python3
"""
render_gt_images_and_index.py

Render GT-view RGB images for each entry in a candidates JSON and write index.parquet.

Run this locally where your 3RScan dataset folder exists.

Expected 3RScan layout per scene:
  <dataset_root>/<scene_id>/
    mesh.refined.v2.obj
    mesh.refined.mtl
    mesh.refined_0.png
    ...

Inputs:
  - candidates_json: abu_eval_pose_candidates*.json (or subset) with:
      scenes: [
        { scene_id, gt_pose: { scene_pose (4x4), position, direction }, ...},
        ...
      ]

Outputs:
  - PNG renders: <out_dir>/<scene_id>/<entry_id>.png
  - index.parquet: one row per entry (paths + GT pose + render settings)

Install (recommended):
  pip install open3d pandas pyarrow numpy tqdm pillow

Fallback renderer:
  pip install trimesh pyrender pyglet PyOpenGL
"""

from __future__ import annotations
import argparse, json, math, re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm


# ------------------------- JSON loading (tolerate trailing commas) -------------------------
def relaxed_json_load(path: Path) -> Dict[str, Any]:
    s = path.read_text(encoding="utf-8")
    s = re.sub(r"//.*?$", "", s, flags=re.M)
    s = re.sub(r"/\*.*?\*/", "", s, flags=re.S)
    s = re.sub(r",\s*(\}|\])", r"\1", s)
    return json.loads(s)


# ------------------------- Pose conversions -------------------------
def as_mat4(m: Any) -> np.ndarray:
    T = np.asarray(m, dtype=np.float64)
    if T.shape != (4, 4):
        raise ValueError(f"Expected 4x4 matrix, got {T.shape}")
    return T

def opencv_to_opengl_cam() -> np.ndarray:
    """
    Convert OpenCV camera coords (x right, y down, z forward)
    to OpenGL camera coords (x right, y up, z backward).
    Applied in camera coordinates.
    """
    C = np.eye(4, dtype=np.float64)
    C[1, 1] = -1.0
    C[2, 2] = -1.0
    return C

def to_camera_pose_c2w(T_scene_pose: np.ndarray, pose_convention: str, camera_coords: str) -> np.ndarray:
    """
    Returns camera-to-world pose suitable for OpenGL-style renderers.

    pose_convention:
      - c2w: input matrix is camera-to-world
      - w2c: input matrix is world-to-camera (we invert it)

    camera_coords:
      - opencv: apply OpenCV->OpenGL axis conversion in camera coordinates
      - opengl: no conversion
    """
    if pose_convention == "c2w":
        T_c2w = T_scene_pose
    elif pose_convention == "w2c":
        T_c2w = np.linalg.inv(T_scene_pose)
    else:
        raise ValueError("pose_convention must be 'c2w' or 'w2c'")

    if camera_coords == "opencv":
        T_c2w = T_c2w @ opencv_to_opengl_cam()
    elif camera_coords != "opengl":
        raise ValueError("camera_coords must be 'opencv' or 'opengl'")

    return T_c2w


# ------------------------- Renderer backends -------------------------
@dataclass
class RenderConfig:
    width: int = 640
    height: int = 480
    fov_deg: float = 60.0
    near: float = 0.05
    far: float = 30.0
    bg_rgb: Tuple[float, float, float] = (1.0, 1.0, 1.0)

def intrinsics_from_fov(cfg: RenderConfig) -> Tuple[float, float, float, float]:
    fov = math.radians(cfg.fov_deg)
    fy = 0.5 * cfg.height / math.tan(0.5 * fov)
    fx = fy
    cx = cfg.width / 2.0
    cy = cfg.height / 2.0
    return fx, fy, cx, cy

class BaseRenderer:
    def render(self, obj_path: Path, T_c2w: np.ndarray, cfg: RenderConfig) -> np.ndarray:
        raise NotImplementedError

class Open3DRenderer(BaseRenderer):
    def __init__(self):
        import open3d as o3d  # noqa
        from open3d.visualization import rendering  # noqa
        self.o3d = o3d
        self.rendering = rendering
        self._cache = {}  # key -> (renderer, scene)

    def _get(self, obj_path: Path, cfg: RenderConfig):
        key = f"{obj_path.resolve()}|{cfg.width}x{cfg.height}"
        if key in self._cache:
            return self._cache[key]

        o3d = self.o3d
        rendering = self.rendering

        mesh = o3d.io.read_triangle_mesh(str(obj_path), enable_post_processing=True)
        if mesh.is_empty():
            raise RuntimeError(f"Open3D failed to load mesh: {obj_path}")
        mesh.compute_vertex_normals()

        r = rendering.OffscreenRenderer(cfg.width, cfg.height)
        scene = r.scene
        scene.set_background([*cfg.bg_rgb, 1.0])

        mat = rendering.MaterialRecord()
        mat.shader = "defaultLit"
        scene.add_geometry("mesh", mesh, mat)

        # lights (avoid black render)
        scene.scene.set_lighting(rendering.Open3DScene.LightingProfile.SOFT_SHADOWS, (0.0, 0.0, 0.0))
        scene.scene.enable_sun_light(True)
        scene.scene.set_sun_light(
            direction=np.array([0.3, -1.0, -1.0], dtype=np.float32),
            color=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            intensity=75000.0,
        )

        self._cache[key] = (r, scene)
        return r, scene

    def render(self, obj_path: Path, T_c2w: np.ndarray, cfg: RenderConfig) -> np.ndarray:
        import open3d as o3d  # noqa
        r, _scene = self._get(obj_path, cfg)

        fx, fy, cx, cy = intrinsics_from_fov(cfg)
        intrinsic = o3d.camera.PinholeCameraIntrinsic(cfg.width, cfg.height, fx, fy, cx, cy)

        # Open3D expects extrinsic = world-to-camera
        T_w2c = np.linalg.inv(T_c2w)
        r.setup_camera(intrinsic, T_w2c)

        img = r.render_to_image()
        arr = np.asarray(img)
        if arr.ndim == 2:
            arr = np.repeat(arr[:, :, None], 3, axis=2)
        if arr.shape[2] == 4:
            arr = arr[:, :, :3]
        return arr.astype(np.uint8)

class PyrenderRenderer(BaseRenderer):
    def __init__(self):
        import trimesh  # noqa
        import pyrender  # noqa
        self.trimesh = trimesh
        self.pyrender = pyrender
        self._mesh_cache = {}
        self._offscreen_renderer = None
        self._offscreen_cfg_key = None

    def _load_mesh(self, obj_path: Path):
        key = str(obj_path.resolve())
        if key in self._mesh_cache:
            return self._mesh_cache[key]

        tm = self.trimesh.load(str(obj_path), force="scene", process=False)

        if isinstance(tm, self.trimesh.Scene):
            geoms = list(tm.geometry.values())
            if not geoms:
                raise RuntimeError(f"No geometry in OBJ scene: {obj_path}")
            # Merge into one mesh (textures may not fully survive merge; OK for debug renders)
            tm_mesh = self.trimesh.util.concatenate(geoms) if len(geoms) > 1 else geoms[0]
        else:
            tm_mesh = tm

        pr_mesh = self.pyrender.Mesh.from_trimesh(tm_mesh, smooth=True)
        self._mesh_cache[key] = pr_mesh
        return pr_mesh

    def _get_offscreen_renderer(self, cfg: RenderConfig):
        key = (cfg.width, cfg.height)
        if self._offscreen_renderer is None or self._offscreen_cfg_key != key:
            if self._offscreen_renderer is not None:
                self._offscreen_renderer.delete()
            self._offscreen_renderer = self.pyrender.OffscreenRenderer(
                viewport_width=cfg.width, viewport_height=cfg.height
            )
            self._offscreen_cfg_key = key
        return self._offscreen_renderer

    def render(self, obj_path: Path, T_c2w: np.ndarray, cfg: RenderConfig) -> np.ndarray:
        pyrender = self.pyrender

        pr_mesh = self._load_mesh(obj_path)
        scene = pyrender.Scene(bg_color=np.array([*cfg.bg_rgb, 1.0]), ambient_light=np.array([0.25, 0.25, 0.25]))
        scene.add(pr_mesh)

        fx, fy, cx, cy = intrinsics_from_fov(cfg)
        cam = pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy, znear=cfg.near, zfar=cfg.far)
        scene.add(cam, pose=T_c2w)

        light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
        scene.add(light, pose=T_c2w)

        r = self._get_offscreen_renderer(cfg)
        color, _depth = r.render(scene)
        return color.astype(np.uint8)


# ------------------------- Helpers -------------------------
def find_obj(dataset_root: Path, scene_id: str) -> Path:
    obj = dataset_root / scene_id / "mesh.refined.v2.obj"
    if not obj.exists():
        raise FileNotFoundError(f"Missing OBJ mesh for scene {scene_id}: {obj}")
    return obj

def safe_entry_id(entry: Dict[str, Any], idx: int) -> str:
    for k in ("frame_id", "query_id", "entry_id"):
        v = entry.get(k)
        if v:
            return str(v)
    return f"entry-{idx:06d}"

def flatten_T(T: np.ndarray) -> Dict[str, float]:
    return {f"T{r}{c}": float(T[r, c]) for r in range(4) for c in range(4)}


# ------------------------- Main -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates_json", required=True)
    ap.add_argument("--dataset_root", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--index_out", required=True)

    ap.add_argument("--only_scene_id", default="")
    ap.add_argument("--limit", type=int, default=0)

    ap.add_argument("--renderer", choices=["open3d", "pyrender"], default="open3d")
    ap.add_argument("--pose_convention", choices=["c2w", "w2c"], default="c2w")
    ap.add_argument("--camera_coords", choices=["opencv", "opengl"], default="opencv")

    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fov_deg", type=float, default=60.0)
    ap.add_argument("--near", type=float, default=0.05)
    ap.add_argument("--far", type=float, default=30.0)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    candidates_json = Path(args.candidates_json)
    dataset_root = Path(args.dataset_root)
    out_dir = Path(args.out_dir)
    index_out = Path(args.index_out)

    out_dir.mkdir(parents=True, exist_ok=True)
    index_out.parent.mkdir(parents=True, exist_ok=True)

    data = relaxed_json_load(candidates_json)
    entries = data.get("scenes", data.get("entries", []))
    if not isinstance(entries, list):
        raise ValueError("Expected candidates JSON to contain a list under key 'scenes' (or 'entries').")

    if args.only_scene_id:
        entries = [e for e in entries if e.get("scene_id") == args.only_scene_id]
    if args.limit and args.limit > 0:
        entries = entries[: args.limit]

    cfg = RenderConfig(width=args.width, height=args.height, fov_deg=args.fov_deg, near=args.near, far=args.far)

    if args.renderer == "open3d":
        renderer: BaseRenderer = Open3DRenderer()
    else:
        renderer = PyrenderRenderer()

    from PIL import Image  # pillow

    rows: List[Dict[str, Any]] = []

    for i, entry in enumerate(tqdm(entries, desc="Rendering GT images")):
        scene_id = str(entry.get("scene_id", "")).strip()
        if not scene_id:
            continue

        gt_pose = entry.get("gt_pose", {})
        if "scene_pose" not in gt_pose:
            continue

        T_raw = as_mat4(gt_pose["scene_pose"])
        T_c2w = to_camera_pose_c2w(T_raw, args.pose_convention, args.camera_coords)

        obj_path = find_obj(dataset_root, scene_id)

        entry_id = safe_entry_id(entry, i)
        out_scene = out_dir / scene_id
        out_scene.mkdir(parents=True, exist_ok=True)
        img_path = out_scene / f"{entry_id}.png"

        if (not img_path.exists()) or args.overwrite:
            img = renderer.render(obj_path, T_c2w, cfg)
            Image.fromarray(img).save(img_path)

        pos = gt_pose.get("position", [float(T_raw[0, 3]), float(T_raw[1, 3]), float(T_raw[2, 3])])
        direc = gt_pose.get("direction", [float(T_raw[0, 2]), float(T_raw[1, 2]), float(T_raw[2, 2])])

        rows.append({
            "scene_id": scene_id,
            "entry_id": entry_id,
            "render_path": str(img_path.resolve()),
            "width": cfg.width,
            "height": cfg.height,
            "fov_deg": cfg.fov_deg,
            "near": cfg.near,
            "far": cfg.far,
            "renderer": args.renderer,
            "pose_convention": args.pose_convention,
            "camera_coords": args.camera_coords,
            "gt_x": float(pos[0]), "gt_y": float(pos[1]), "gt_z": float(pos[2]),
            "gt_dx": float(direc[0]), "gt_dy": float(direc[1]), "gt_dz": float(direc[2]),
            **flatten_T(T_raw),
        })

    df = pd.DataFrame(rows)
    if len(df) == 0:
        raise RuntimeError("No rows written. Check candidates JSON has gt_pose.scene_pose entries.")

    df.to_parquet(index_out, index=False)
    print(f"\nWrote {len(df)} rows: {index_out}")
    print(f"Rendered images under: {out_dir}")

    print("\nIf renders look wrong, try:")
    print("  --pose_convention w2c")
    print("  --camera_coords opengl")

if __name__ == "__main__":
    main()
