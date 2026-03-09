"""Shared camera utilities for 3D scene processing.

Provides common functions for loading camera poses, intrinsics,
and performing coordinate transformations used by both ScanNet and 3RScan
processing pipelines.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np


def load_cam2world(pose_path: Path) -> np.ndarray:
    """Load a 4x4 camera-to-world matrix from a pose file.

    Works with both ScanNet and 3RScan pose formats, which store the
    pose as a 4x4 matrix in a text file.

    Args:
        pose_path: Path to the pose text file.

    Returns:
        A (4, 4) camera-to-world transformation matrix.
    """
    return np.loadtxt(pose_path, dtype=np.float64).reshape(4, 4)


def invert_se3_to_opencv(cam2world: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Convert camera-to-world SE(3) matrix to OpenCV world-to-camera (R, t).

    OpenCV camera coordinates: x right, y down, z forward.
    PyTorch3D's OpenCV bridge expects exactly this (R, t) pair.

    Args:
        cam2world: (4, 4) camera-to-world transformation matrix.

    Returns:
        Tuple of (R_cv, t_cv) where R_cv is a (3, 3) rotation matrix and
        t_cv is a (3,) translation vector, both in OpenCV convention.
    """
    R_cw = cam2world[:3, :3]
    t_cw = cam2world[:3, 3]
    R_cv = R_cw.T
    t_cv = -R_cv @ t_cw
    return R_cv, t_cv


def load_intrinsics_txt(path: Path) -> Tuple[float, float, float, float]:
    """Load camera intrinsics from a 4x4 matrix text file.

    This format is used by ScanNet's ``intrinsic_color.txt`` files.

    Args:
        path: Path to the intrinsics text file.

    Returns:
        Tuple of (fx, fy, cx, cy) intrinsic parameters in pixels.
    """
    K = np.loadtxt(path, dtype=np.float64).reshape(4, 4)
    return float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])


def load_intrinsics_info(info_path: Path) -> Tuple[float, float, float, float]:
    """Parse 3RScan ``_info.txt`` file to obtain pinhole intrinsics.

    Args:
        info_path: Path to the ``_info.txt`` file distributed with the scan.

    Returns:
        Tuple of (fx, fy, cx, cy) intrinsic parameters in pixels.

    Raises:
        RuntimeError: If the calibration matrix cannot be located in the file.
    """
    lines = info_path.read_text().splitlines()
    K = None
    for L in lines:
        if L.startswith("m_calibrationColorIntrinsic"):
            vals = [float(x) for x in L.split("=")[1].split()]
            K = np.array(vals).reshape(4, 4)
    if K is None:
        raise RuntimeError("Could not parse intrinsics from _info.txt")
    return float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])


def compute_pose_distance(pose1: np.ndarray, pose2: np.ndarray) -> Tuple[float, float]:
    """Compute spatial distance between two camera poses.

    Args:
        pose1: (4, 4) camera-to-world transformation matrix.
        pose2: (4, 4) camera-to-world transformation matrix.

    Returns:
        Tuple of (position_dist, angle_dist) where position_dist is the
        Euclidean distance in meters and angle_dist is the angular distance
        in degrees between viewing directions.
    """
    pos1 = pose1[:3, 3]
    pos2 = pose2[:3, 3]
    position_dist = float(np.linalg.norm(pos1 - pos2))

    R1 = pose1[:3, :3]
    R2 = pose2[:3, :3]
    forward1 = -R1[:, 2]
    forward2 = -R2[:, 2]

    cos_angle = np.clip(np.dot(forward1, forward2), -1.0, 1.0)
    angle_dist = float(np.degrees(np.arccos(cos_angle)))

    return position_dist, angle_dist


def is_pose_too_similar(
    pose: np.ndarray,
    selected_poses: list,
    min_position_dist: float,
    min_angle_dist: float,
) -> bool:
    """Check if a pose is too similar to any already-selected pose.

    Args:
        pose: Candidate (4, 4) camera-to-world pose.
        selected_poses: List of already-selected (4, 4) poses.
        min_position_dist: Minimum position distance threshold (meters).
        min_angle_dist: Minimum angle distance threshold (degrees).

    Returns:
        True if the pose is too close to any selected pose, False otherwise.
    """
    for selected_pose in selected_poses:
        pos_dist, ang_dist = compute_pose_distance(pose, selected_pose)
        if pos_dist < min_position_dist and ang_dist < min_angle_dist:
            return True
    return False


# ---------------------------------------------------------------------
# Camera Pose JSON Loading (for annotation and UI tools)
# ---------------------------------------------------------------------


def load_camera_poses_json(
    scene_path: Union[str, Path],
    output_folder: str = "output",
    pose_filename: str = "camera_pose.json",
) -> Dict[str, List[List[float]]]:
    """Load camera poses from a JSON file containing per-frame 4x4 matrices.

    Used by annotation tools and the Streamlit UI to load the camera poses
    exported during NBV keyframe selection.

    Args:
        scene_path: Path to the scene directory (e.g. ``data/scans/scene0000_00``).
        output_folder: Subdirectory within the scene containing outputs.
        pose_filename: Name of the JSON file.

    Returns:
        Dictionary mapping frame ID strings to 4x4 pose matrices (as nested
        lists). Returns an empty dict if the file does not exist.

    Example:
        >>> poses = load_camera_poses_json('/data/scans/scene0000_00')
        >>> pose_matrix = np.array(poses['000123'])
    """
    scene_path = Path(scene_path)
    pose_file = scene_path / output_folder / pose_filename

    if not pose_file.exists():
        return {}

    return json.loads(pose_file.read_text())


def load_camera_poses_dict(
    pose_dir: Union[str, Path],
    frame_ids: List[str],
    pose_suffix: str = ".txt",
) -> Dict[str, np.ndarray]:
    """Load camera poses from individual text files into a dictionary.

    Useful when you have a list of frame IDs and need to load their poses
    from separate ``.txt`` files (as used by ScanNet/3RScan).

    Args:
        pose_dir: Directory containing pose files.
        frame_ids: List of frame IDs to load.
        pose_suffix: File extension/suffix for pose files.

    Returns:
        Dictionary mapping frame ID strings to (4, 4) numpy pose matrices.
        Missing poses are silently skipped.

    Example:
        >>> poses = load_camera_poses_dict('/data/scene/pose', ['000001', '000002'])
        >>> R, t = invert_se3_to_opencv(poses['000001'])
    """
    pose_dir = Path(pose_dir)
    poses = {}

    for fid in frame_ids:
        pose_path = pose_dir / f"{fid}{pose_suffix}"
        if pose_path.exists():
            poses[fid] = load_cam2world(pose_path)

    return poses
