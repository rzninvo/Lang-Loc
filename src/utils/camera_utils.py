"""
Shared camera utilities for 3D scene processing.

This module provides common functions for loading camera poses, intrinsics,
and performing coordinate transformations used by both ScanNet and 3RScan
processing pipelines.
"""
from pathlib import Path
from typing import Tuple

import numpy as np


def load_cam2world(pose_path: Path) -> np.ndarray:
    """
    Load a 4x4 camera-to-world matrix from a pose file.

    This function works with both ScanNet and 3RScan pose formats,
    which store the pose as a 4x4 matrix in a text file.

    Args:
        pose_path: Path to the pose text file.

    Returns:
        np.ndarray: (4, 4) camera-to-world transformation matrix.
    """
    return np.loadtxt(pose_path, dtype=np.float64).reshape(4, 4)


def invert_se3_to_opencv(cam2world: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert camera-to-world SE(3) matrix to OpenCV world-to-camera (R, t).

    OpenCV camera coordinates: x right, y down, z forward.
    PyTorch3D's OpenCV bridge expects exactly this (R, t) pair.

    Args:
        cam2world: (4, 4) camera-to-world transformation matrix.

    Returns:
        Tuple containing:
            - R_cv: (3, 3) rotation matrix world-to-camera (OpenCV convention).
            - t_cv: (3,) translation vector world-to-camera (OpenCV convention).
    """
    R_cw = cam2world[:3, :3]
    t_cw = cam2world[:3, 3]
    R_cv = R_cw.T
    t_cv = -R_cv @ t_cw
    return R_cv, t_cv


def load_intrinsics_txt(path: Path) -> Tuple[float, float, float, float]:
    """
    Load camera intrinsics from a 4x4 matrix text file.

    This format is used by ScanNet's 'intrinsic_color.txt' files.

    Args:
        path: Path to the intrinsics text file.

    Returns:
        Tuple of (fx, fy, cx, cy) intrinsic parameters in pixels.
    """
    K = np.loadtxt(path, dtype=np.float64).reshape(4, 4)
    return float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])


def load_intrinsics_info(info_path: Path) -> Tuple[float, float, float, float]:
    """
    Parse 3RScan `_info.txt` file to obtain pinhole intrinsics.

    Args:
        info_path: Path to the `_info.txt` file distributed with the scan.

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
    """
    Compute spatial distance between two camera poses.

    Args:
        pose1: (4, 4) camera-to-world transformation matrix.
        pose2: (4, 4) camera-to-world transformation matrix.

    Returns:
        Tuple containing:
            - position_dist: Euclidean distance (meters) between camera positions.
            - angle_dist: Angular distance (degrees) between viewing directions.
    """
    # Position distance
    pos1 = pose1[:3, 3]
    pos2 = pose2[:3, 3]
    position_dist = float(np.linalg.norm(pos1 - pos2))

    # Angular distance using viewing directions (forward = -Z axis in camera frame)
    R1 = pose1[:3, :3]
    R2 = pose2[:3, :3]
    forward1 = -R1[:, 2]  # -Z axis in world coords
    forward2 = -R2[:, 2]

    # Compute angle between viewing directions
    cos_angle = np.clip(np.dot(forward1, forward2), -1.0, 1.0)
    angle_dist = float(np.degrees(np.arccos(cos_angle)))

    return position_dist, angle_dist


def is_pose_too_similar(
    pose: np.ndarray,
    selected_poses: list,
    min_position_dist: float,
    min_angle_dist: float,
) -> bool:
    """
    Check if a pose is too similar to any already-selected pose.

    Args:
        pose: Candidate (4, 4) camera-to-world pose.
        selected_poses: List of already-selected (4, 4) poses.
        min_position_dist: Minimum position distance threshold (meters).
        min_angle_dist: Minimum angle distance threshold (degrees).

    Returns:
        True if pose is too close to any selected pose, False otherwise.
    """
    for selected_pose in selected_poses:
        pos_dist, ang_dist = compute_pose_distance(pose, selected_pose)
        if pos_dist < min_position_dist and ang_dist < min_angle_dist:
            return True
    return False
