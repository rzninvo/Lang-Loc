"""
Sensor data classes for reading and processing ScanNet .sens files.

This module provides utilities for parsing the binary .sens file format
used by ScanNet, including RGB-D frame extraction, depth decompression,
and camera pose/intrinsics export.
"""
import os
import struct
from concurrent.futures import ProcessPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import imageio.v2 as imageio
import numpy as np
import zlib
from PIL import Image
from tqdm import tqdm


COMPRESSION_TYPE_COLOR = {-1: 'unknown', 0: 'raw', 1: 'png', 2: 'jpeg'}
COMPRESSION_TYPE_DEPTH = {-1: 'unknown', 0: 'raw_ushort', 1: 'zlib_ushort', 2: 'occi_ushort'}


# ---------------------------------------------------------------------------
# Top-level worker functions (must be module-level for ProcessPoolExecutor)
# ---------------------------------------------------------------------------

def _export_one_depth(
    frame_idx: int,
    depth_data: bytes,
    compression_type: str,
    depth_height: int,
    depth_width: int,
    output_path: str,
    image_size: Optional[Tuple[int, int]],
) -> None:
    """Decompress, optionally resize, and write one depth frame as 16-bit PNG."""
    if compression_type == 'zlib_ushort':
        raw = zlib.decompress(depth_data)
    else:
        raise ValueError(f"Unsupported depth compression type: {compression_type}")
    depth = np.frombuffer(raw, dtype=np.uint16).reshape(depth_height, depth_width)
    if image_size is not None:
        depth = cv2.resize(depth, (image_size[1], image_size[0]),
                           interpolation=cv2.INTER_NEAREST)
    imageio.imwrite(os.path.join(output_path, f'{frame_idx:06d}.png'),
                    depth.astype(np.uint16))


def _export_one_color(
    frame_idx: int,
    color_data: bytes,
    compression_type: str,
    output_path: str,
    image_size: Optional[Tuple[int, int]],
) -> None:
    """Decompress, optionally resize, and write one color frame as JPEG."""
    if compression_type == 'jpeg':
        color = np.array(Image.open(BytesIO(color_data)))
    else:
        raise ValueError(f"Unsupported color compression type: {compression_type}")
    if image_size is not None:
        color = cv2.resize(color, (image_size[1], image_size[0]),
                           interpolation=cv2.INTER_NEAREST)
    imageio.imwrite(os.path.join(output_path, f'{frame_idx:06d}.jpg'), color)


def _export_one_pose(
    frame_idx: int,
    camera_to_world: np.ndarray,
    output_path: str,
) -> None:
    """Write one 4x4 camera pose to a text file."""
    filepath = os.path.join(output_path, f'{frame_idx:06d}.txt')
    with open(filepath, 'w') as f:
        for line in camera_to_world:
            np.savetxt(f, line[np.newaxis], fmt='%.6f')


class RGBDFrame:
    """Represents a single RGB-D frame from a .sens file."""

    camera_to_world: np.ndarray
    timestamp_color: int
    timestamp_depth: int
    color_size_bytes: int
    depth_size_bytes: int
    color_data: bytes
    depth_data: bytes

    def load(self, file_handle) -> None:
        """
        Load frame data from a binary file handle.

        Args:
            file_handle: Open file handle positioned at the start of frame data.
        """
        self.camera_to_world = np.asarray(
            struct.unpack('f' * 16, file_handle.read(16 * 4)),
            dtype=np.float32
        ).reshape(4, 4)
        self.timestamp_color = struct.unpack('Q', file_handle.read(8))[0]
        self.timestamp_depth = struct.unpack('Q', file_handle.read(8))[0]
        self.color_size_bytes = struct.unpack('Q', file_handle.read(8))[0]
        self.depth_size_bytes = struct.unpack('Q', file_handle.read(8))[0]
        self.color_data = file_handle.read(self.color_size_bytes)
        self.depth_data = file_handle.read(self.depth_size_bytes)

    def decompress_depth(self, compression_type: str) -> bytes:
        """
        Decompress depth data based on compression type.

        Args:
            compression_type: The compression format (e.g., 'zlib_ushort').

        Returns:
            Decompressed depth data as bytes.

        Raises:
            ValueError: If compression type is not supported.
        """
        if compression_type == 'zlib_ushort':
            return self.decompress_depth_zlib()
        else:
            raise ValueError(f"Unsupported depth compression type: {compression_type}")

    def decompress_depth_zlib(self) -> bytes:
        """Decompress zlib-compressed depth data."""
        return zlib.decompress(self.depth_data)

    def decompress_color(self, compression_type: str) -> np.ndarray:
        """
        Decompress color data based on compression type.

        Args:
            compression_type: The compression format (e.g., 'jpeg').

        Returns:
            Decompressed color image as numpy array.

        Raises:
            ValueError: If compression type is not supported.
        """
        if compression_type == 'jpeg':
            return self.decompress_color_jpeg()
        else:
            raise ValueError(f"Unsupported color compression type: {compression_type}")

    def decompress_color_jpeg(self) -> np.ndarray:
        """Decompress JPEG-compressed color data."""
        return np.array(Image.open(BytesIO(self.color_data)))


class SensorData:
    """
    Parser for ScanNet .sens binary files.

    Loads and provides access to RGB-D frames, camera intrinsics/extrinsics,
    and sensor metadata from a .sens file.

    Attributes:
        version: File format version (expected: 4).
        sensor_name: Name of the sensor used for capture.
        intrinsic_color: 4x4 color camera intrinsic matrix.
        extrinsic_color: 4x4 color camera extrinsic matrix.
        intrinsic_depth: 4x4 depth camera intrinsic matrix.
        extrinsic_depth: 4x4 depth camera extrinsic matrix.
        color_compression_type: Compression format for color frames.
        depth_compression_type: Compression format for depth frames.
        color_width: Width of color frames in pixels.
        color_height: Height of color frames in pixels.
        depth_width: Width of depth frames in pixels.
        depth_height: Height of depth frames in pixels.
        depth_shift: Depth scale factor.
        frames: List of RGBDFrame objects.
    """

    def __init__(self, filename: str) -> None:
        """
        Initialize and load a .sens file.

        Args:
            filename: Path to the .sens file.
        """
        self.version = 4
        self.load(filename)

    def load(self, filename: str) -> None:
        """
        Load sensor data from a .sens file.

        Args:
            filename: Path to the .sens file.

        Raises:
            AssertionError: If file version doesn't match expected version.
        """
        with open(filename, 'rb') as f:
            version = struct.unpack('I', f.read(4))[0]
            assert self.version == version, f"Version mismatch: expected {self.version}, got {version}"

            strlen = struct.unpack('Q', f.read(8))[0]
            self.sensor_name = f.read(strlen).decode('utf-8')
            self.intrinsic_color = np.asarray(
                struct.unpack('f' * 16, f.read(16 * 4)),
                dtype=np.float32
            ).reshape(4, 4)
            self.extrinsic_color = np.asarray(
                struct.unpack('f' * 16, f.read(16 * 4)),
                dtype=np.float32
            ).reshape(4, 4)
            self.intrinsic_depth = np.asarray(
                struct.unpack('f' * 16, f.read(16 * 4)),
                dtype=np.float32
            ).reshape(4, 4)
            self.extrinsic_depth = np.asarray(
                struct.unpack('f' * 16, f.read(16 * 4)),
                dtype=np.float32
            ).reshape(4, 4)
            self.color_compression_type = COMPRESSION_TYPE_COLOR[struct.unpack('i', f.read(4))[0]]
            self.depth_compression_type = COMPRESSION_TYPE_DEPTH[struct.unpack('i', f.read(4))[0]]
            self.color_width = struct.unpack('I', f.read(4))[0]
            self.color_height = struct.unpack('I', f.read(4))[0]
            self.depth_width = struct.unpack('I', f.read(4))[0]
            self.depth_height = struct.unpack('I', f.read(4))[0]
            self.depth_shift = struct.unpack('f', f.read(4))[0]
            num_frames = struct.unpack('Q', f.read(8))[0]

            self.frames: List[RGBDFrame] = []
            for _ in range(num_frames):
                frame = RGBDFrame()
                frame.load(f)
                self.frames.append(frame)

    def export_depth_images(
        self,
        output_path: str,
        image_size: Optional[Tuple[int, int]] = None,
        frame_skip: int = 1,
        num_workers: int = 1,
    ) -> None:
        """
        Export depth frames as 16-bit PNG images.

        Args:
            output_path: Directory to save depth images.
            image_size: Optional (height, width) to resize images.
            frame_skip: Export every Nth frame.
            num_workers: Number of parallel workers (1 = sequential).
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        indices = list(range(0, len(self.frames), frame_skip))
        print(f'Exporting {len(indices)} depth frames to {output_path}'
              f' (workers={num_workers})')
        out_str = str(output_path)

        if num_workers <= 1:
            for f in indices:
                _export_one_depth(
                    f, self.frames[f].depth_data,
                    self.depth_compression_type,
                    self.depth_height, self.depth_width,
                    out_str, image_size,
                )
        else:
            with ProcessPoolExecutor(max_workers=num_workers) as pool:
                futures = {
                    pool.submit(
                        _export_one_depth, f, self.frames[f].depth_data,
                        self.depth_compression_type,
                        self.depth_height, self.depth_width,
                        out_str, image_size,
                    ): f
                    for f in indices
                }
                for fut in tqdm(as_completed(futures), total=len(futures),
                                desc="Depth export"):
                    fut.result()

    def export_color_images(
        self,
        output_path: str,
        image_size: Optional[Tuple[int, int]] = None,
        frame_skip: int = 1,
        num_workers: int = 1,
    ) -> None:
        """
        Export color frames as JPEG images.

        Args:
            output_path: Directory to save color images.
            image_size: Optional (height, width) to resize images.
            frame_skip: Export every Nth frame.
            num_workers: Number of parallel workers (1 = sequential).
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        indices = list(range(0, len(self.frames), frame_skip))
        print(f'Exporting {len(indices)} color frames to {output_path}'
              f' (workers={num_workers})')
        out_str = str(output_path)

        if num_workers <= 1:
            for f in indices:
                _export_one_color(
                    f, self.frames[f].color_data,
                    self.color_compression_type,
                    out_str, image_size,
                )
        else:
            with ProcessPoolExecutor(max_workers=num_workers) as pool:
                futures = {
                    pool.submit(
                        _export_one_color, f, self.frames[f].color_data,
                        self.color_compression_type,
                        out_str, image_size,
                    ): f
                    for f in indices
                }
                for fut in tqdm(as_completed(futures), total=len(futures),
                                desc="Color export"):
                    fut.result()

    def save_mat_to_file(self, matrix: np.ndarray, filename: str) -> None:
        """
        Save a matrix to a text file.

        Args:
            matrix: Matrix to save.
            filename: Output file path.
        """
        with open(filename, 'w') as f:
            for line in matrix:
                np.savetxt(f, line[np.newaxis], fmt='%.6f')

    def export_poses(
        self,
        output_path: str,
        frame_skip: int = 1,
        num_workers: int = 1,
    ) -> None:
        """
        Export camera poses as 4x4 matrix text files.

        Args:
            output_path: Directory to save pose files.
            frame_skip: Export every Nth frame.
            num_workers: Number of parallel workers (1 = sequential).
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        indices = list(range(0, len(self.frames), frame_skip))
        print(f'Exporting {len(indices)} camera poses to {output_path}'
              f' (workers={num_workers})')
        out_str = str(output_path)

        if num_workers <= 1:
            for f in indices:
                _export_one_pose(f, self.frames[f].camera_to_world, out_str)
        else:
            with ProcessPoolExecutor(max_workers=num_workers) as pool:
                futures = {
                    pool.submit(
                        _export_one_pose, f,
                        self.frames[f].camera_to_world, out_str,
                    ): f
                    for f in indices
                }
                for fut in tqdm(as_completed(futures), total=len(futures),
                                desc="Pose export"):
                    fut.result()

    def export_intrinsics(self, output_path: str) -> None:
        """
        Export camera intrinsics and extrinsics matrices.

        Args:
            output_path: Directory to save intrinsic/extrinsic files.
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        print(f'Exporting camera intrinsics to {output_path}')

        self.save_mat_to_file(self.intrinsic_color, output_path / 'intrinsic_color.txt')
        self.save_mat_to_file(self.extrinsic_color, output_path / 'extrinsic_color.txt')
        self.save_mat_to_file(self.intrinsic_depth, output_path / 'intrinsic_depth.txt')
        self.save_mat_to_file(self.extrinsic_depth, output_path / 'extrinsic_depth.txt')
