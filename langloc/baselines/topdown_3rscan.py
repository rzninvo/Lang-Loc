#!/usr/bin/env python3
"""
Generate top-down (2D) views from 3RScan meshes.

Renders projected mesh triangles (not scatter points) for a solid floor-plan
appearance with a white background.

Examples
--------
Single scan:
    python -m langloc.baselines.topdown_3rscan \
        --root /path/to/3RScan_processed \
        --scan-id 0ad2d3a1-79e2-2212-9b99-a96495d9f7fe \
        --visualize \
        --output ./out_dir

Batch for every scan under --root:
    python -m langloc.baselines.topdown_3rscan \
        --root /path/to/3RScan_processed \
        --all-scans \
        --output ./topdown_maps


"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np
import open3d as o3d
from PIL import Image
from tqdm import tqdm

from langloc.eval.view_iou import PREFERRED_MESH_FILES, discover_mesh

log = logging.getLogger(__name__)


def discover_scan_dirs(dataset_root: Path) -> List[Path]:
    """Return scan directories that contain at least one known mesh filename."""
    scan_dirs: List[Path] = []
    for child in sorted(dataset_root.iterdir()):
        if not child.is_dir():
            continue
        if any((child / name).exists() for name in PREFERRED_MESH_FILES):
            scan_dirs.append(child)
    return scan_dirs


def _default_colors(num_points: int) -> np.ndarray:
    return np.tile(np.array([[0.45, 0.45, 0.45]], dtype=np.float32), (num_points, 1))


def _has_valid_vertex_colors(colors: np.ndarray, num_verts: int) -> bool:
    """Check if vertex colors array is present, valid, and non-trivial."""
    if colors.shape != (num_verts, 3):
        return False
    if not np.isfinite(colors).all():
        return False
    if np.allclose(colors, 0.0):
        return False
    return True


def _find_texture_image(mesh_path: Path,
                        mesh: o3d.geometry.TriangleMesh) -> np.ndarray | None:
    """Locate and return the texture image as an (H, W, 3) uint8 array, or None."""
    if mesh.textures:
        return np.asarray(mesh.textures[0])

    mesh_dir = mesh_path.parent
    stem = mesh_path.stem
    base = stem.split(".")[0] + "." + stem.split(".")[1] if "." in stem else stem
    for candidate in [
        mesh_dir / f"{base}_0.png",
        mesh_dir / f"{stem}_0.png",
        mesh_dir / f"{stem}.png",
        mesh_dir / "texture.png",
    ]:
        if candidate.exists():
            return np.array(Image.open(candidate).convert("RGB"))
    # Try reading from MTL file
    mtl_path = mesh_path.with_suffix(".mtl")
    if not mtl_path.exists():
        mtl_path = mesh_dir / (base + ".mtl")
    if mtl_path.exists():
        for line in mtl_path.read_text().splitlines():
            line = line.strip()
            if line.startswith("map_Kd"):
                tex_file = mesh_dir / line.split(None, 1)[1]
                if tex_file.exists():
                    return np.array(Image.open(tex_file).convert("RGB"))
    return None


def _bake_texture_to_vertex_colors(mesh_path: Path,
                                    mesh: o3d.geometry.TriangleMesh) -> bool:
    """
    Bake texture into vertex colors on the mesh *in-place* so Open3D can
    display them in the interactive viewer.  Returns True on success.

    Strategy: for each vertex, collect the UV of every triangle corner that
    references it, average those UVs, and sample the texture there.
    """
    if not mesh.has_triangle_uvs():
        return False

    triangle_uvs = np.asarray(mesh.triangle_uvs, dtype=np.float32)
    faces = np.asarray(mesh.triangles, dtype=np.int32)
    num_verts = len(mesh.vertices)

    if triangle_uvs.size == 0 or faces.size == 0:
        return False

    tex_img = _find_texture_image(mesh_path, mesh)
    if tex_img is None:
        return False

    h, w = tex_img.shape[:2]
    uvs_per_face = triangle_uvs.reshape(-1, 3, 2)  # (F, 3, 2)

    # Accumulate UVs per vertex using vectorized bincount (faster than np.add.at)
    all_vert_ids = faces.reshape(-1)                          # (F*3,)
    all_uvs = uvs_per_face.reshape(-1, 2).astype(np.float64)  # (F*3, 2)

    uv_sum_u = np.bincount(all_vert_ids, weights=all_uvs[:, 0], minlength=num_verts)
    uv_sum_v = np.bincount(all_vert_ids, weights=all_uvs[:, 1], minlength=num_verts)
    uv_count = np.bincount(all_vert_ids, minlength=num_verts).astype(np.float64)

    valid = uv_count > 0
    uv_avg = np.zeros((num_verts, 2), dtype=np.float32)
    uv_avg[valid, 0] = (uv_sum_u[valid] / uv_count[valid]).astype(np.float32)
    uv_avg[valid, 1] = (uv_sum_v[valid] / uv_count[valid]).astype(np.float32)

    px = np.clip((uv_avg[:, 0] * w).astype(int), 0, w - 1)
    py = np.clip(((1.0 - uv_avg[:, 1]) * h).astype(int), 0, h - 1)

    vertex_colors = tex_img[py, px, :3].astype(np.float64) / 255.0
    mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
    return True


def load_mesh(mesh_path: Path) -> Tuple[o3d.geometry.TriangleMesh,
                                        np.ndarray,
                                        np.ndarray,
                                        np.ndarray]:
    """
    Load mesh geometry, faces, and vertex colors.
    If the mesh has no vertex colors but has a UV-mapped texture, the texture
    is baked into per-vertex colors so both the 3D viewer and 2D renderer
    use the same high-quality color source.
    Returns:
        mesh: open3d triangle mesh
        vertices: (N, 3) float32
        faces: (F, 3) int32  — triangle vertex indices
        vertex_colors: (N, 3) float32 in [0, 1]
    """
    mesh = o3d.io.read_triangle_mesh(str(mesh_path), enable_post_processing=True)
    mesh.compute_vertex_normals()

    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.triangles, dtype=np.int32)
    colors = np.asarray(mesh.vertex_colors, dtype=np.float32)

    if vertices.size == 0 or faces.size == 0:
        raise RuntimeError(f"Could not read mesh with faces from {mesh_path}")

    if not _has_valid_vertex_colors(colors, len(vertices)):
        # Optional fallback for workflows that depend on vertex colors.
        if _bake_texture_to_vertex_colors(mesh_path, mesh):
            colors = np.asarray(mesh.vertex_colors, dtype=np.float32)
        else:
            colors = _default_colors(len(vertices))

    return mesh, vertices, faces, colors


def plane_to_axes(plane: str) -> Tuple[int, int, int]:
    """Map plane name to plotted axes and filtered height axis."""
    if plane == "xy":
        return 0, 1, 2
    if plane == "xz":
        return 0, 2, 1
    if plane == "yz":
        return 1, 2, 0
    raise ValueError(f"Unsupported plane '{plane}'")


def filter_faces_by_height(vertices: np.ndarray,
                           faces: np.ndarray,
                           height_axis: int,
                           floor_percentile: float,
                           ceiling_percentile: float,
                           cutoff_above_ground_m: float | None = None) -> np.ndarray:
    """Return a face mask for triangles whose all three vertices fall in height range."""
    heights = vertices[:, height_axis]
    lo = float(np.percentile(heights, floor_percentile))
    hi = float(np.percentile(heights, ceiling_percentile))
    if cutoff_above_ground_m is not None:
        # Estimate ground from a lower-height band, then clamp at ground + offset meters.
        # This is more robust than using the global minimum height (which may be an outlier).
        ground_lo = float(np.percentile(heights, max(0.0, floor_percentile)))
        ground_hi = float(np.percentile(heights, min(100.0, floor_percentile + 5.0)))
        ground_band = heights[(heights >= ground_lo) & (heights <= ground_hi)]
        ground_height = float(np.median(ground_band)) if ground_band.size else ground_lo
        hi = min(hi, ground_height + cutoff_above_ground_m)

    # A face is kept if all its vertices are within range
    face_heights = heights[faces]  # (F, 3)
    return np.all((face_heights >= lo) & (face_heights <= hi), axis=1)


def build_filtered_mesh(mesh_full: o3d.geometry.TriangleMesh,
                        vertices_full: np.ndarray,
                        faces_full: np.ndarray,
                        face_mask: np.ndarray,
                        vertex_colors_full: np.ndarray) -> o3d.geometry.TriangleMesh:
    """Create filtered mesh while preserving UV textures if present."""
    faces = faces_full[face_mask]
    used = np.unique(faces)
    remap = np.full(len(vertices_full), -1, dtype=np.int32)
    remap[used] = np.arange(len(used), dtype=np.int32)

    vertices = vertices_full[used]
    faces = remap[faces]

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices.astype(np.float64))
    mesh.triangles = o3d.utility.Vector3iVector(faces.astype(np.int32))

    # Preserve original UV texture mapping for accurate filtered visualization.
    tri_uvs = np.asarray(mesh_full.triangle_uvs, dtype=np.float32)
    if mesh_full.has_triangle_uvs() and tri_uvs.shape == (len(faces_full) * 3, 2):
        mesh.textures = mesh_full.textures
        uvs = tri_uvs.reshape(-1, 3, 2)[face_mask].reshape(-1, 2)
        mesh.triangle_uvs = o3d.utility.Vector2dVector(uvs.astype(np.float64))

        tri_mats = np.asarray(mesh_full.triangle_material_ids, dtype=np.int32)
        if tri_mats.size == len(faces_full):
            mesh.triangle_material_ids = o3d.utility.IntVector(tri_mats[face_mask].tolist())
    else:
        mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors_full[used].astype(np.float64))

    mesh.compute_vertex_normals()
    return mesh


def visualize_mesh_3d(mesh: o3d.geometry.TriangleMesh,
                      title: str = "") -> None:
    """Open a mesh in Open3D's interactive 3D viewer."""
    print(f"Opening 3D viewer ({len(mesh.vertices)} verts, {len(mesh.triangles)} faces) ...")
    o3d.visualization.draw_geometries(
        [mesh],
        window_name=title or "Filtered mesh",
        width=1280,
        height=960,
        mesh_show_back_face=True,
    )


def _topdown_camera_vectors(plane: str) -> Tuple[np.ndarray, np.ndarray]:
    """Return (front, up) vectors for requested top-down projection plane."""
    if plane == "xy":
        return np.array([0.0, 0.0, 1.0]), np.array([0.0, 1.0, 0.0])
    if plane == "xz":
        return np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])
    if plane == "yz":
        return np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])
    raise ValueError(f"Unsupported plane '{plane}'")


def _camera_param_path(output: Path) -> Path:
    """Return output path for camera matrices saved together in a .npz file."""
    base = output.stem
    return output.with_name(f"{base}_camera.npz")


def render_topdown_mesh(mesh: o3d.geometry.TriangleMesh,
                        plane: str,
                        output: Path,
                        dpi: int) -> None:
    """
    Render top-down 2D image using the same Open3D textured mesh pipeline as 3D viewer.
    """
    size = max(1024, int(dpi * 4))
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="topdown", width=size, height=size, visible=False)
    vis.add_geometry(mesh)

    opt = vis.get_render_option()
    opt.background_color = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    opt.mesh_show_back_face = True

    bbox = mesh.get_axis_aligned_bounding_box()
    lookat = bbox.get_center()
    front, up = _topdown_camera_vectors(plane)

    ctr = vis.get_view_control()
    ctr.set_lookat(lookat)
    ctr.set_front(front)
    ctr.set_up(up)
    ctr.set_zoom(0.5)

    vis.poll_events()
    vis.update_renderer()

    output.parent.mkdir(parents=True, exist_ok=True)
    
    params = ctr.convert_to_pinhole_camera_parameters()
    intrinsic = np.asarray(params.intrinsic.intrinsic_matrix, dtype=np.float64)
    extrinsic = np.asarray(params.extrinsic, dtype=np.float64)
    camera_out = _camera_param_path(output)
    np.savez(camera_out, intrinsic=intrinsic, extrinsic=extrinsic)

    vis.capture_screen_image(str(output), do_render=True)
    vis.destroy_window()
    log.debug("Saved: %s", output)
    log.debug("Saved camera params: %s", camera_out)


def make_topdown(scan_dir: Path,
                 output: Path,
                 plane: str,
                 floor_percentile: float,
                 ceiling_percentile: float,
                 cutoff_above_ground_m: float | None,
                 dpi: int,
                 show: bool,
                 visualize: bool = False,
                 visualize_full: bool = False) -> None:
    """Create a top-down image for one scan directory."""
    mesh_path = discover_mesh(scan_dir)
    mesh, vertices, faces, vertex_colors = load_mesh(mesh_path)

    if visualize_full:
        visualize_mesh_3d(mesh, title=f"{scan_dir.name} (full mesh)")

    _, _, height_axis = plane_to_axes(plane)

    face_mask = filter_faces_by_height(
        vertices, faces, height_axis,
        floor_percentile=floor_percentile,
        ceiling_percentile=ceiling_percentile,
        cutoff_above_ground_m=cutoff_above_ground_m,
    )
    filtered_mesh = build_filtered_mesh(
        mesh_full=mesh,
        vertices_full=vertices,
        faces_full=faces,
        face_mask=face_mask,
        vertex_colors_full=vertex_colors,
    )

    if len(filtered_mesh.triangles) == 0:
        raise RuntimeError(
            f"All faces were removed by height filtering for {scan_dir.name}. "
            "Try a wider percentile range or increase/disable --cutoff_above_ground_m."
        )

    if visualize:
        visualize_mesh_3d(filtered_mesh, title=scan_dir.name)

    render_topdown_mesh(
        mesh=filtered_mesh,
        plane=plane,
        output=output,
        dpi=dpi,
    )

    if show:
        Image.open(output).show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create top-down views for 3RScan scenes.")
    parser.add_argument("--root", required=True, type=Path,
                        help="3RScan root directory with <scan_id>/ subfolders.")
    parser.add_argument("--scan-id", "--scan_id", dest="scan_id", type=str,
                        help="Single scan UUID to render.")
    parser.add_argument("--all-scans", "--all_scans", dest="all_scans", action="store_true",
                        help="Render every scan folder under --root.")
    parser.add_argument("--output", type=Path,
                        help="Output directory. Single scan saves as "
                             "<output>/<scene_id>_topdown.png + "
                             "<scene_id>_topdown_camera.npz; "
                             "--all-scans saves as <output>/<scene_id>/<output-name>.")
    parser.add_argument("--out-dir", "--out_dir", dest="out_dir", type=Path,
                        help="Output directory for --all-scans mode.")
    parser.add_argument("--output-name", "--output_name", dest="output_name", type=str,
                        default="topdown.png",
                        help="Filename to use per scene in --all-scans mode. "
                             "Saved as <output>/<scene-id>/<output-name>, with "
                             "camera parameters in *_camera.npz (intrinsic + extrinsic).")
    parser.add_argument("--plane", choices=("xy", "xz", "yz"), default="xy",
                        help="Projection plane. Top-down for 3RScan is usually 'xy'.")
    parser.add_argument("--floor_percentile", type=float, default=0.2,
                        help="Lower height percentile to keep (removes floor outliers).")
    parser.add_argument("--ceiling_percentile", type=float, default=95.0,
                        help="Upper height percentile to keep (removes ceilings/noise).")
    parser.add_argument("--cutoff_above_ground_m", type=float, default=2.1, # 2.3,
                        help="Additionally cap max height to this many meters above "
                             "estimated ground on the selected height axis.")
    parser.add_argument("--dpi", type=int, default=300,
                        help="Output image DPI.")
    parser.add_argument("--show", action="store_true",
                        help="Display the plot window after saving.")
    parser.add_argument("--visualize", action="store_true",
                        help="Open the filtered mesh in Open3D's interactive 3D viewer.")
    parser.add_argument("--visualize-full", "--visualize_full", dest="visualize_full",
                        action="store_true",
                        help="Open the complete unfiltered mesh in Open3D viewer.")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.scan_id and args.all_scans:
        raise ValueError("Choose either --scan-id or --all-scans, not both.")
    if not args.scan_id and not args.all_scans:
        raise ValueError("Provide either --scan-id <uuid> or --all-scans.")

    if not (0.0 <= args.floor_percentile < args.ceiling_percentile <= 100.0):
        raise ValueError("Percentiles must satisfy 0 <= floor < ceiling <= 100.")
    if args.cutoff_above_ground_m is not None and args.cutoff_above_ground_m <= 0.0:
        raise ValueError("--cutoff_above_ground_m must be > 0 when provided.")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    args = parse_args()
    validate_args(args)

    root = args.root.expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(root)

    if args.scan_id:
        scan_dir = root / args.scan_id
        if not scan_dir.exists():
            raise FileNotFoundError(scan_dir)
        out_dir = args.output or args.out_dir or Path.cwd()
        out_dir.mkdir(parents=True, exist_ok=True)
        output = out_dir / f"{args.scan_id}_topdown.png"
        make_topdown(
            scan_dir=scan_dir,
            output=output,
            plane=args.plane,
            floor_percentile=args.floor_percentile,
            ceiling_percentile=args.ceiling_percentile,
            cutoff_above_ground_m=args.cutoff_above_ground_m,
            dpi=args.dpi,
            show=args.show,
            visualize=args.visualize,
            visualize_full=args.visualize_full,
        )
        return

    scan_dirs = discover_scan_dirs(root)
    if not scan_dirs:
        raise RuntimeError(f"No scan folders with known mesh files found under {root}")

    out_dir = args.output or args.out_dir or root
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Found {len(scan_dirs)} scan(s). Saving maps as <scene-id>/{args.output_name} under {out_dir}")

    ok = 0
    failed = 0
    pbar = tqdm(scan_dirs, desc="Rendering topdown", unit="scene")
    for scan_dir in pbar:
        out_file = out_dir / scan_dir.name / args.output_name
        try:
            make_topdown(
                scan_dir=scan_dir,
                output=out_file,
                plane=args.plane,
                floor_percentile=args.floor_percentile,
                ceiling_percentile=args.ceiling_percentile,
                cutoff_above_ground_m=args.cutoff_above_ground_m,
                dpi=args.dpi,
                show=False,
                visualize=args.visualize,
                visualize_full=args.visualize_full,
            )
            ok += 1
            pbar.set_postfix(ok=ok, failed=failed, refresh=False)
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"[ERROR] {scan_dir.name}: {exc}")
            pbar.set_postfix(ok=ok, failed=failed, refresh=False)

    print(f"Done. Success={ok}, Failed={failed}")


if __name__ == "__main__":
    main()
