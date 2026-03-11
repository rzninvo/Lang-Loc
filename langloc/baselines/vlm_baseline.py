#!/usr/bin/env python3
"""
VLM baseline evaluator for topdown pose prediction on 3RScan_processed.

Pipeline per frame:
1) Read topdown image and textual frame description.
2) Query Qwen-VL for topdown pixel (u,v) and heading angle (w).
3) Project prediction to world coordinates using topdown camera intrinsics/extrinsics.
4) Compute Euclidean and angular errors against frame ground-truth pose.
5) Save per-frame metrics JSON and a summary log under eval/.

Visualization is debug-only and fully gated behind --visualize.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, GenerationConfig, Qwen3VLForConditionalGeneration


from langloc.eval.view_iou import (
    discover_mesh,
    normalize,
)
from langloc.localization.frame_io import format_args_section

DEFAULT_MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
DEFAULT_TOPDOWN_NAME = "topdown.png"


@dataclass(frozen=True)
class Pose3D:
    """3D camera pose with position and optional viewing direction."""

    position: np.ndarray
    direction: Optional[np.ndarray]


@dataclass(frozen=True)
class QwenModelBundle:
    """Container for loaded Qwen VLM model, processor, and vision config."""

    model: Qwen3VLForConditionalGeneration
    processor: AutoProcessor
    patch_size: int
    merge_size: int


@dataclass(frozen=True)
class FailureRecord:
    """Record of a failed frame inference for post-run reporting."""

    scene_id: str
    frame_id: str
    reason: str


def parse_args() -> argparse.Namespace:
    """Build and return the CLI argument parser for VLM baseline evaluation."""
    parser = argparse.ArgumentParser(
        description="Run Qwen VLM baseline on all scenes/frames and evaluate pose errors."
    )
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Root directory containing <scene_id>/ topdown image and frame JSONs.",
    )
    parser.add_argument(
        "--save_metrics",
        type=Path,
        default=Path("eval/baseline_eval_metric_qwen.json"),
        help="Path to write per-frame metrics JSON.",
    )
    parser.add_argument(
        "--log_file",
        type=Path,
        default=Path("eval/baseline_eval_metric_qwen.log"),
        help="Path to write aggregate summary log.",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default=DEFAULT_MODEL_ID,
        help="HF model id for Qwen VLM.",
    )
    parser.add_argument(
        "--device_map",
        type=str,
        default="cuda",
        help="device_map passed to model.from_pretrained.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=128,
        help="Maximum new tokens for VLM generation.",
    )
    parser.add_argument(
        "--retry_count",
        type=int,
        default=2,
        help="Retry attempts after a failed frame inference/parse.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing metrics JSON if present.",
    )
    parser.add_argument(
        "--scene_ids",
        nargs="+",
        help="Optional list of scene IDs to evaluate.",
    )
    parser.add_argument(
        "--max_scenes",
        type=int,
        help="Optional maximum number of scenes to process.",
    )
    parser.add_argument(
        "--max_frames_per_scene",
        type=int,
        help="Optional maximum frames per scene (for debug runs).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for deterministic behavior where relevant.",
    )
    parser.add_argument(
        "--eye_height_m",
        type=float,
        default=1.6,
        help="Eye height above floor plane for predicted camera position.",
    )
    parser.add_argument(
        "--direction_step_px",
        type=float,
        default=20.0,
        help="Pixel step used to convert heading angle to world direction.",
    )
    parser.add_argument(
        "--hit_radii",
        nargs="+",
        type=float,
        default=[0.75, 1.0, 1.5, 2.0, 2.5],
        help="Radii (m) for Hit@r metrics.",
    )
    parser.add_argument(
        "--mass_percentiles",
        nargs="+",
        type=float,
        default=[50.0, 90.0],
        help="Mass-radius percentiles (single-point baseline maps to distance).",
    )
    parser.add_argument(
        "--top_k_min_dist",
        type=int,
        default=10,
        help="Top-K min distance setting (single-point baseline maps to distance).",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Debug only: visualize mesh with GT/predicted pose for each processed frame.",
    )
    return parser.parse_args()


def build_prompt(scene_description: str, width: int, height: int) -> str:
    """Construct the Qwen VLM prompt for topdown localization and heading estimation.

    Args:
        scene_description: Natural-language description of the scene.
        width: Topdown image width in pixels.
        height: Topdown image height in pixels.

    Returns:
        Formatted prompt string instructing the model to output u, v, w JSON.
    """
    return f"""
You are a vision model helping with top-down localization and heading estimation.

Return ONLY a JSON object with keys u,v,w:
{{"u": <float>, "v": <float>, "w": <float>}}

Coordinate convention:
- Origin (0,0) is top-left of the image.
- u increases to the right, v increases downward.
- u must be in [0, {width}), v must be in [0, {height}).

Angle convention:
- w in degrees in [0,360)
- w = 0° points right (+u)
- w increases CCW (90° up, 180° left, 270° down)

IMPORTANT:
- u,v must refer to pixel coordinates of the ORIGINAL input image size {width}x{height}.

Scene description:
<<<
{scene_description}
>>>
""".strip()


def parse_prediction_json(raw_response: str) -> Dict[str, float]:
    """Extract the first JSON object from the model's raw text output.

    Args:
        raw_response: Raw text generated by the VLM.

    Returns:
        Dict with float values for keys ``u``, ``v``, ``w``.

    Raises:
        ValueError: If no valid JSON with the required keys is found.
    """
    start = raw_response.find("{")
    end = raw_response.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in model output.")
    parsed = json.loads(raw_response[start:end + 1])
    for key in ("u", "v", "w"):
        if key not in parsed or not isinstance(parsed[key], (int, float)):
            raise ValueError(f"Model output missing numeric '{key}'.")
    return {"u": float(parsed["u"]), "v": float(parsed["v"]), "w": float(parsed["w"])}


def clamp_prediction(u: float, v: float, w: float, width: int, height: int) -> Tuple[int, int, float]:
    """Clamp predicted pixel coordinates to image bounds and normalize heading angle.

    Args:
        u: Predicted horizontal pixel coordinate.
        v: Predicted vertical pixel coordinate.
        w: Predicted heading angle in degrees.
        width: Image width in pixels.
        height: Image height in pixels.

    Returns:
        Clamped ``(u_int, v_int, w_normalized)`` tuple.
    """
    u_i = int(round(u))
    v_i = int(round(v))
    if not (0 <= u_i < width):
        u_i = max(0, min(u_i, width - 1))
    if not (0 <= v_i < height):
        v_i = max(0, min(v_i, height - 1))
    w_norm = float(w % 360.0)
    return u_i, v_i, w_norm


def ensure_processor_multiple(images: Iterable[Image.Image], patch_size: int, merge_size: int) -> None:
    """Verify that all processed images have dimensions divisible by patch*merge size.

    Args:
        images: Iterable of PIL Images after vision preprocessing.
        patch_size: Vision encoder patch size.
        merge_size: Vision encoder merge size.

    Raises:
        ValueError: If any image dimension is not divisible by the required multiple.
    """
    required = patch_size * merge_size
    for idx, image in enumerate(images):
        w_seen, h_seen = image.size
        if (w_seen % required) != 0 or (h_seen % required) != 0:
            raise ValueError(
                "Vision preprocessing mismatch: image "
                f"index {idx} resized to {w_seen}x{h_seen}, not divisible by {required}."
            )


def load_qwen_bundle(args: argparse.Namespace) -> QwenModelBundle:
    """Load and configure the Qwen VLM model and processor.

    Args:
        args: Parsed CLI arguments with ``model_id`` and ``device_map``.

    Returns:
        Frozen bundle containing the model, processor, and vision config.
    """
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_id,
        torch_dtype="auto",
        device_map=args.device_map,
    ).eval()
    processor = AutoProcessor.from_pretrained(args.model_id, do_resize=False, use_fast=False)
    patch_size = int(processor.image_processor.patch_size)
    merge_size = int(processor.image_processor.merge_size)
    return QwenModelBundle(model=model, processor=processor, patch_size=patch_size, merge_size=merge_size)


def run_qwen_inference(
    bundle: QwenModelBundle,
    image_path: Path,
    description: str,
    max_new_tokens: int,
) -> Tuple[int, int, float, str]:
    """Run a single VLM inference to predict topdown pixel position and heading.

    Args:
        bundle: Loaded Qwen model bundle.
        image_path: Path to the topdown image.
        description: Scene description text.
        max_new_tokens: Maximum tokens for generation.

    Returns:
        Tuple of ``(u_pixel, v_pixel, heading_deg, raw_model_response)``.

    Raises:
        ValueError: If the model outputs a degenerate all-zero prediction.
    """
    with Image.open(image_path) as img:
        width, height = img.convert("RGB").size

    prompt = build_prompt(description, width, height)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    image_inputs, video_inputs = process_vision_info(messages, image_patch_size=bundle.patch_size)
    image_seq: List[Image.Image]
    if isinstance(image_inputs, (list, tuple)):
        image_seq = list(image_inputs)
    else:
        image_seq = [image_inputs]
    ensure_processor_multiple(image_seq, bundle.patch_size, bundle.merge_size)

    text = bundle.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = bundle.processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt",
    ).to(bundle.model.device)
    gen_cfg = GenerationConfig(max_new_tokens=max_new_tokens, do_sample=False)

    with torch.no_grad():
        out = bundle.model.generate(**inputs, generation_config=gen_cfg)
    generated = out[0][inputs["input_ids"].shape[-1]:]
    raw_response = bundle.processor.decode(generated, skip_special_tokens=True).strip()
    pred = parse_prediction_json(raw_response)
    # Exact all-zero output is usually a collapsed/invalid prediction.
    if pred["u"] == 0.0 and pred["v"] == 0.0 and pred["w"] == 0.0:
        raise ValueError("Degenerate model output (u=v=w=0).")

    # The prompt explicitly instructs the model to output pixel coordinates in
    # the ORIGINAL image space (width × height).  ensure_processor_multiple
    # only pads (never downscales), so seen_w >= width and the model is told
    # the original dimensions — its output is already in original coords.
    # clamp_prediction handles any out-of-range values safely.
    u_i, v_i, w = clamp_prediction(pred["u"], pred["v"], pred["w"], width, height)
    return u_i, v_i, w, raw_response


def find_scene_dirs(root: Path, scene_ids: Optional[Sequence[str]], max_scenes: Optional[int]) -> List[Path]:
    """Discover scene directories under root, optionally filtered by id or count.

    Args:
        root: Dataset root directory.
        scene_ids: Optional whitelist of scene IDs.
        max_scenes: Optional cap on number of scenes.

    Returns:
        Sorted list of scene directory paths.
    """
    candidates = [p for p in sorted(root.iterdir()) if p.is_dir()]
    if scene_ids:
        wanted = set(scene_ids)
        candidates = [p for p in candidates if p.name in wanted]
    if max_scenes is not None:
        candidates = candidates[:max_scenes]
    return candidates


def resolve_topdown_paths(scene_dir: Path) -> Tuple[Path, Path]:
    """Locate the topdown image and camera parameter files for a scene.

    Args:
        scene_dir: Scene directory.

    Returns:
        Tuple of ``(topdown_image_path, camera_npz_path)``.

    Raises:
        FileNotFoundError: If either file is missing.
    """
    image_path = scene_dir / DEFAULT_TOPDOWN_NAME
    camera_path = scene_dir / "topdown_camera.npz"
    if not image_path.exists():
        raise FileNotFoundError(f"Missing topdown image: {image_path}")
    if not camera_path.exists():
        raise FileNotFoundError(f"Missing topdown camera sidecar: {camera_path}")
    return image_path, camera_path


def frame_json_paths(scene_dir: Path, max_frames: Optional[int]) -> List[Path]:
    """List frame description JSON paths, excluding ``*_parsed.json`` files.

    Args:
        scene_dir: Scene directory.
        max_frames: Optional limit on number of frames returned.

    Returns:
        Sorted list of frame JSON paths.
    """
    desc_dir = scene_dir / "output" / "descriptions"
    frames = [
        p
        for p in sorted(desc_dir.glob("frame-*.json"))
        if not p.stem.endswith("_parsed")
    ]
    if max_frames is not None:
        frames = frames[:max_frames]
    return frames


def load_topdown_camera(camera_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load topdown camera intrinsic and extrinsic matrices from an npz file.

    Args:
        camera_path: Path to the ``.npz`` camera parameter file.

    Returns:
        Tuple of ``(intrinsic_3x3, extrinsic_4x4)`` numpy arrays.

    Raises:
        ValueError: If matrices have unexpected shapes.
    """
    data = np.load(camera_path)
    intrinsic = np.asarray(data["intrinsic"], dtype=np.float64)
    extrinsic = np.asarray(data["extrinsic"], dtype=np.float64)
    if intrinsic.shape != (3, 3):
        raise ValueError(f"Expected intrinsic 3x3 in {camera_path}, got {intrinsic.shape}")
    if extrinsic.shape != (4, 4):
        raise ValueError(f"Expected extrinsic 4x4 in {camera_path}, got {extrinsic.shape}")
    return intrinsic, extrinsic


def gt_pose_from_frame(frame: dict, frame_path: Path) -> Pose3D:
    """Extract ground-truth camera pose from a frame JSON entry.

    Args:
        frame: Parsed frame JSON dict containing ``scene_pose``.
        frame_path: Path to the frame JSON (for error messages).

    Returns:
        Ground-truth pose with position and forward direction.

    Raises:
        ValueError: If ``scene_pose`` is not a valid 4x4 matrix.
    """
    pose = np.asarray(frame.get("scene_pose"), dtype=np.float64)
    if pose.shape != (4, 4):
        raise ValueError(f"Expected scene_pose 4x4 in {frame_path}, got {pose.shape}")
    gt_pos = pose[:3, 3].astype(np.float64)
    gt_dir = pose[:3, :3] @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
    gt_dir = normalize(gt_dir)
    return Pose3D(position=gt_pos, direction=gt_dir)


def estimate_floor_z(frame: dict, gt_pose: Pose3D, eye_height_m: float) -> float:
    """Estimate the Z coordinate of the floor plane from visible object annotations.

    Falls back to ``gt_pose.position[2] - eye_height_m`` when no floor objects
    are found in the frame metadata.

    Args:
        frame: Parsed frame JSON dict with ``visible_objects``.
        gt_pose: Ground-truth camera pose for fallback estimation.
        eye_height_m: Assumed eye height above floor in meters.

    Returns:
        Estimated floor Z coordinate in world space.
    """
    visible = frame.get("visible_objects", {}) or {}
    floor_lows: List[float] = []
    for obj in visible.values():
        label = str(obj.get("label", "")).strip().lower()
        if "floor" not in label:
            continue
        bbox = obj.get("bbox_world")
        if not isinstance(bbox, list) or len(bbox) != 2:
            continue
        try:
            lo = float(bbox[0][2])
            hi = float(bbox[1][2])
        except (TypeError, ValueError, IndexError):
            continue
        floor_lows.append(min(lo, hi))

    if floor_lows:
        return float(np.median(np.asarray(floor_lows, dtype=np.float64)))

    # Fallback when floor object annotation is absent in the frame JSON.
    return float(gt_pose.position[2] - eye_height_m)


def camera_center_from_extrinsic(extrinsic_w2c: np.ndarray) -> np.ndarray:
    """Compute the world-space camera center from a world-to-camera extrinsic matrix.

    Args:
        extrinsic_w2c: 4x4 world-to-camera transformation matrix.

    Returns:
        Camera center as a 3-element float64 array.
    """
    r = extrinsic_w2c[:3, :3]
    t = extrinsic_w2c[:3, 3]
    return (-r.T @ t).astype(np.float64)


def pixel_to_world_ray(
    u: float,
    v: float,
    intrinsic: np.ndarray,
    extrinsic_w2c: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Unproject a pixel coordinate to a world-space ray origin and direction.

    Args:
        u: Horizontal pixel coordinate.
        v: Vertical pixel coordinate.
        intrinsic: 3x3 camera intrinsic matrix.
        extrinsic_w2c: 4x4 world-to-camera extrinsic matrix.

    Returns:
        Tuple of ``(ray_origin, ray_direction)`` in world space.

    Raises:
        ValueError: If the resulting ray is degenerate.
    """
    pixel_h = np.array([u, v, 1.0], dtype=np.float64)
    ray_cam = np.linalg.inv(intrinsic) @ pixel_h
    ray_cam = normalize(ray_cam)
    if ray_cam is None:
        raise ValueError("Invalid camera ray from intrinsic inversion.")

    ray_world = normalize(extrinsic_w2c[:3, :3].T @ ray_cam)
    if ray_world is None:
        raise ValueError("Invalid world-space ray direction.")
    center = camera_center_from_extrinsic(extrinsic_w2c)
    return center, ray_world


def intersect_ray_with_z_plane(ray_origin: np.ndarray, ray_dir: np.ndarray, z_plane: float) -> np.ndarray:
    """Intersect a 3D ray with a horizontal plane at a fixed Z height.

    Args:
        ray_origin: 3D ray origin.
        ray_dir: 3D ray direction (need not be unit length).
        z_plane: Z coordinate of the horizontal plane.

    Returns:
        3D intersection point.

    Raises:
        ValueError: If the ray is parallel to the plane.
    """
    dz = float(ray_dir[2])
    if abs(dz) < 1e-9:
        raise ValueError("Ray is parallel to fixed Z plane; cannot intersect.")
    lam = (float(z_plane) - float(ray_origin[2])) / dz
    return ray_origin + lam * ray_dir


def pixel_to_world_on_z(
    u: float,
    v: float,
    intrinsic: np.ndarray,
    extrinsic_w2c: np.ndarray,
    z_plane: float,
) -> np.ndarray:
    """Project a pixel coordinate to a 3D world point on a fixed-Z plane.

    Args:
        u: Horizontal pixel coordinate.
        v: Vertical pixel coordinate.
        intrinsic: 3x3 intrinsic matrix.
        extrinsic_w2c: 4x4 world-to-camera extrinsic.
        z_plane: Target Z height for intersection.

    Returns:
        3D world-space intersection point.
    """
    origin, ray_dir = pixel_to_world_ray(u, v, intrinsic, extrinsic_w2c)
    return intersect_ray_with_z_plane(origin, ray_dir, z_plane)


def omega_to_world_direction(
    u: float,
    v: float,
    omega_deg: float,
    intrinsic: np.ndarray,
    extrinsic_w2c: np.ndarray,
    z_plane: float,
    step_px: float,
) -> Optional[np.ndarray]:
    """Convert a pixel heading angle to a 3D world-space direction vector.

    Advances ``step_px`` pixels along the heading direction in image space,
    projects both the origin and advanced point to the Z plane, and returns
    the normalized horizontal direction.

    Args:
        u: Predicted pixel position (horizontal).
        v: Predicted pixel position (vertical).
        omega_deg: Heading angle in degrees (0 = right, CCW positive).
        intrinsic: 3x3 intrinsic matrix.
        extrinsic_w2c: 4x4 world-to-camera extrinsic.
        z_plane: Z height for projection.
        step_px: Pixel step size for finite-difference direction estimate.

    Returns:
        Normalized 3D direction vector (Z component zeroed), or ``None``
        if normalization fails.
    """
    rad = math.radians(float(omega_deg))
    du = math.cos(rad) * float(step_px)
    dv = -math.sin(rad) * float(step_px)
    p0 = pixel_to_world_on_z(u, v, intrinsic, extrinsic_w2c, z_plane)
    p1 = pixel_to_world_on_z(u + du, v + dv, intrinsic, extrinsic_w2c, z_plane)
    direction = p1 - p0
    direction[2] = 0.0
    return normalize(direction)


def compute_errors(pred: Pose3D, gt: Pose3D) -> Tuple[float, Optional[float]]:
    """Compute Euclidean distance and angular error between predicted and GT poses.

    Args:
        pred: Predicted pose.
        gt: Ground-truth pose.

    Returns:
        Tuple of ``(distance_meters, angular_error_degrees)``.
        Angular error is ``None`` if either pose lacks a direction.
    """
    distance_m = float(np.linalg.norm(pred.position - gt.position))
    if pred.direction is None or gt.direction is None:
        return distance_m, None
    dot = float(np.clip(np.dot(pred.direction, gt.direction), -1.0, 1.0))
    angular_deg = float(math.degrees(math.acos(dot)))
    return distance_m, angular_deg


def single_point_metrics(
    distance_m: float,
    hit_radii: Sequence[float],
    mass_percentiles: Sequence[float],
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Compute Hit@r and mass-radius metrics for a single-point prediction.

    Args:
        distance_m: Euclidean distance error in meters.
        hit_radii: Radius thresholds for Hit@r (meters).
        mass_percentiles: Percentile labels for mass-radius reporting.

    Returns:
        Tuple of ``(hit_masses_dict, mass_radii_dict)``.
    """
    hit_masses = {str(float(r)): float(1.0 if distance_m <= float(r) else 0.0) for r in hit_radii}
    mass_radii = {str(float(p)): float(distance_m) for p in mass_percentiles}
    return hit_masses, mass_radii


def visualize_debug_pose(
    scene_id: str,
    scene_dir: Path,
    gt_pose: Pose3D,
    pred_pose: Pose3D,
    marker_radius_m: float = 0.08,
    arrow_length_m: float = 0.8,
) -> None:
    """Open an interactive 3D viewer showing GT and predicted poses on the scene mesh.

    Args:
        scene_id: Scene identifier for the window title.
        scene_dir: Scene directory containing the mesh file.
        gt_pose: Ground-truth pose (green marker).
        pred_pose: Predicted pose (red marker).
        marker_radius_m: Sphere marker radius in meters.
        arrow_length_m: Direction arrow length in meters.
    """
    import open3d as o3d

    mesh_path = discover_mesh(scene_dir)
    mesh = o3d.io.read_triangle_mesh(str(mesh_path), enable_post_processing=True)
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()

    geoms: List[o3d.geometry.Geometry] = [mesh]

    def marker(center: np.ndarray, color: Tuple[float, float, float]) -> o3d.geometry.TriangleMesh:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=marker_radius_m)
        sphere.compute_vertex_normals()
        sphere.paint_uniform_color(np.asarray(color, dtype=np.float64))
        sphere.translate(np.asarray(center, dtype=np.float64))
        return sphere

    def direction_line(
        center: np.ndarray,
        direction: Optional[np.ndarray],
        color: Tuple[float, float, float],
    ) -> Optional[o3d.geometry.LineSet]:
        if direction is None:
            return None
        d = normalize(np.asarray(direction, dtype=np.float64))
        if d is None:
            return None
        p0 = np.asarray(center, dtype=np.float64)
        p1 = p0 + d * float(arrow_length_m)
        line = o3d.geometry.LineSet()
        line.points = o3d.utility.Vector3dVector(np.vstack([p0, p1]))
        line.lines = o3d.utility.Vector2iVector(np.array([[0, 1]], dtype=np.int32))
        line.colors = o3d.utility.Vector3dVector(np.asarray([color], dtype=np.float64))
        return line

    geoms.append(marker(gt_pose.position, (0.1, 0.8, 0.1)))
    geoms.append(marker(pred_pose.position, (0.9, 0.1, 0.1)))
    gt_line = direction_line(gt_pose.position, gt_pose.direction, (0.1, 0.8, 0.1))
    pred_line = direction_line(pred_pose.position, pred_pose.direction, (0.9, 0.1, 0.1))
    if gt_line is not None:
        geoms.append(gt_line)
    if pred_line is not None:
        geoms.append(pred_line)

    o3d.visualization.draw_geometries(
        geoms,
        window_name=f"VLM Baseline Debug - {scene_id}",
        width=1400,
        height=900,
        mesh_show_back_face=True,
    )


def load_existing_results(path: Path) -> Tuple[List[dict], set[Tuple[str, str]]]:
    """Load previously saved evaluation results for resume support.

    Args:
        path: Path to the metrics JSON file.

    Returns:
        Tuple of ``(results_list, completed_keys_set)`` where each key
        is a ``(scene_id, frame_id)`` pair.
    """
    if not path.exists():
        return [], set()
    payload = json.loads(path.read_text())
    if not isinstance(payload, list):
        return [], set()
    completed = set()
    for item in payload:
        if not isinstance(item, dict):
            continue
        scene_id = str(item.get("scene_id", "")).strip()
        frame_id = str(item.get("frame_id", "")).strip()
        if scene_id and frame_id:
            completed.add((scene_id, frame_id))
    return payload, completed


def aggregate_log(
    results: List[dict],
    failures: List[FailureRecord],
    args: argparse.Namespace,
    elapsed_sec: float,
) -> str:
    """Build a human-readable summary log from evaluation results and failures.

    Args:
        results: Per-frame metric dictionaries.
        failures: Records of frames that failed during inference.
        args: CLI arguments (included in log header).
        elapsed_sec: Total wall-clock time in seconds.

    Returns:
        Formatted multi-line log string.
    """
    scene_rows: List[str] = []
    per_scene_results: Dict[str, List[dict]] = defaultdict(list)
    per_scene_fail: Dict[str, int] = defaultdict(int)

    for item in results:
        per_scene_results[str(item["scene_id"])].append(item)
    for fail in failures:
        per_scene_fail[fail.scene_id] += 1

    scene_ids = sorted(set(per_scene_results.keys()) | set(per_scene_fail.keys()))
    header = (
        "Scene                                | Frames | Failed | "
        "Err mean (m) | Err median (m) | Ang mean (deg) | Ang median (deg)"
    )
    separator = "-" * len(header)
    scene_rows.extend([header, separator])

    all_err: List[float] = []
    all_ang: List[float] = []
    for scene_id in scene_ids:
        rows = per_scene_results.get(scene_id, [])
        errors = [float(r["distance_error"]) for r in rows]
        angles = [float(r["angular_error_deg"]) for r in rows if r.get("angular_error_deg") is not None]
        all_err.extend(errors)
        all_ang.extend(angles)
        err_mean = float(np.mean(errors)) if errors else float("nan")
        err_median = float(np.median(errors)) if errors else float("nan")
        ang_mean = float(np.mean(angles)) if angles else float("nan")
        ang_median = float(np.median(angles)) if angles else float("nan")
        scene_rows.append(
            f"{scene_id:<36} | {len(rows):>6d} | {per_scene_fail.get(scene_id, 0):>6d} | "
            f"{err_mean:>12.3f} | {err_median:>14.3f} | {ang_mean:>14.3f} | {ang_median:>16.3f}"
        )

    lines = [format_args_section(args), "", "Scene-level summary table", ""]
    if scene_ids:
        lines.extend(scene_rows)
    else:
        lines.append("No successful frames.")

    lines.extend(
        [
            "",
            "Aggregate metrics ---------------------------------------",
            f"  Processed frames        : {len(results)}",
            f"  Failed frames           : {len(failures)}",
            f"  Distance error (m)      : mean={np.mean(all_err):.3f} | median={np.median(all_err):.3f}"
            if all_err
            else "  Distance error (m)      : n/a",
            f"  Angular error (deg)     : mean={np.mean(all_ang):.3f} | median={np.median(all_ang):.3f}"
            if all_ang
            else "  Angular error (deg)     : n/a",
            f"  Wall time (s)           : {elapsed_sec:.2f}",
            "---------------------------------------------------------",
        ]
    )

    if failures:
        lines.extend(["", "Failure summary ----------------------------------------"])
        for fail in failures:
            lines.append(f"  {fail.scene_id} | {fail.frame_id} | {fail.reason}")
        lines.append("---------------------------------------------------------")
    return "\n".join(lines).rstrip() + "\n"


def save_json(path: Path, payload: List[dict]) -> None:
    """Write a list of dicts to a JSON file, creating parent directories as needed.

    Args:
        path: Output file path.
        payload: Data to serialize.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def main() -> None:
    """Run the VLM baseline evaluation over all scenes and frames."""
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    start_time = time.time()
    scene_dirs = find_scene_dirs(args.root, args.scene_ids, args.max_scenes)
    if not scene_dirs:
        print(f"No scene directories found under {args.root}")
        return

    existing_results: List[dict] = []
    completed: set[Tuple[str, str]] = set()
    if args.resume:
        existing_results, completed = load_existing_results(args.save_metrics)
        print(f"Resume enabled: loaded {len(existing_results)} existing records.")

    print(f"Loading model: {args.model_id}")
    bundle = load_qwen_bundle(args)

    results: List[dict] = list(existing_results)
    failures: List[FailureRecord] = []
    processed_new = 0

    for s_idx, scene_dir in enumerate(scene_dirs, start=1):
        scene_id = scene_dir.name
        print(f"[{s_idx:04d}/{len(scene_dirs):04d}] Scene {scene_id}")

        try:
            topdown_image, camera_npz = resolve_topdown_paths(scene_dir)
            intrinsic, extrinsic = load_topdown_camera(camera_npz)
        except Exception as exc:  # noqa: BLE001
            failures.append(FailureRecord(scene_id=scene_id, frame_id="*", reason=str(exc)))
            print(f"  [WARN] Scene skipped: {exc}")
            continue

        frame_paths = frame_json_paths(scene_dir, args.max_frames_per_scene)
        if not frame_paths:
            failures.append(FailureRecord(scene_id=scene_id, frame_id="*", reason="No frame-*.json files found"))
            print("  [WARN] No frame JSON files.")
            continue

        for f_idx, frame_path in enumerate(frame_paths, start=1):
            try:
                frame = json.loads(frame_path.read_text())
                frame_id = str(frame.get("image_index", frame_path.stem))
                key = (scene_id, frame_id)
                if key in completed:
                    continue

                description = str(frame.get("description", "")).strip()
                if not description:
                    raise ValueError("Missing non-empty 'description' in frame JSON.")

                gt_pose = gt_pose_from_frame(frame, frame_path)
                floor_z = estimate_floor_z(frame, gt_pose, args.eye_height_m)
                pred_z = floor_z + float(args.eye_height_m)

                last_err: Optional[Exception] = None
                for _attempt in range(args.retry_count + 1):
                    try:
                        pred_u, pred_v, pred_w, raw = run_qwen_inference(
                            bundle=bundle,
                            image_path=topdown_image,
                            description=description,
                            max_new_tokens=args.max_new_tokens,
                        )
                        pred_pos = pixel_to_world_on_z(pred_u, pred_v, intrinsic, extrinsic, pred_z)
                        pred_dir = omega_to_world_direction(
                            u=pred_u,
                            v=pred_v,
                            omega_deg=pred_w,
                            intrinsic=intrinsic,
                            extrinsic_w2c=extrinsic,
                            z_plane=pred_z,
                            step_px=args.direction_step_px,
                        )
                        pred_pose = Pose3D(position=pred_pos, direction=pred_dir)
                        dist_m, ang_deg = compute_errors(pred_pose, gt_pose)
                        hit_masses, mass_radii = single_point_metrics(
                            distance_m=dist_m,
                            hit_radii=args.hit_radii,
                            mass_percentiles=args.mass_percentiles,
                        )
                        entry = {
                            "scene_id": scene_id,
                            "frame_id": frame_id,
                            "hit_masses": hit_masses,
                            "mass_radii": mass_radii,
                            "topk_min_dist": float(dist_m),
                            "distance_error": float(dist_m),
                            "angular_error_deg": ang_deg,
                            "grid_points": 1,
                            "matched_objects": 0,
                            "iou_error": None,
                            "pred_u": int(pred_u),
                            "pred_v": int(pred_v),
                            "pred_w_deg": float(pred_w),
                            "pred_position_xyz": [float(x) for x in pred_pose.position.tolist()],
                            "pred_direction_xyz": None if pred_pose.direction is None else [float(x) for x in pred_pose.direction.tolist()],
                            "gt_position_xyz": [float(x) for x in gt_pose.position.tolist()],
                            "gt_direction_xyz": None if gt_pose.direction is None else [float(x) for x in gt_pose.direction.tolist()],
                            "raw_model_response": raw,
                        }
                        results.append(entry)
                        completed.add(key)
                        processed_new += 1
                        if args.visualize:
                            visualize_debug_pose(scene_id=scene_id, scene_dir=scene_dir, gt_pose=gt_pose, pred_pose=pred_pose)
                        break
                    except Exception as exc:  # noqa: BLE001
                        last_err = exc
                else:
                    raise RuntimeError(str(last_err) if last_err is not None else "Unknown inference error.")

                if f_idx % 10 == 0:
                    print(f"  processed {f_idx}/{len(frame_paths)} frames")
            except Exception as exc:  # noqa: BLE001
                failures.append(
                    FailureRecord(
                        scene_id=scene_id,
                        frame_id=str(frame_path.stem),
                        reason=str(exc),
                    )
                )

        # Save incrementally for long runs.
        save_json(args.save_metrics, results)

    elapsed = time.time() - start_time
    log_payload = aggregate_log(results=results, failures=failures, args=args, elapsed_sec=elapsed)
    args.log_file.parent.mkdir(parents=True, exist_ok=True)
    args.log_file.write_text(log_payload)
    print(f"Saved metrics to {args.save_metrics}")
    print(f"Saved log to {args.log_file}")
    print(f"Processed new frames: {processed_new}")
    print(f"Failures: {len(failures)}")


if __name__ == "__main__":
    main()
