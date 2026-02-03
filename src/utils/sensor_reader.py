"""
Command-line tool for extracting data from ScanNet .sens files.

This script provides a CLI interface to export depth images, color images,
camera poses, and intrinsics from the binary .sens format used by ScanNet.

Usage:
    python sensor_reader.py --filename /path/to/scene.sens --output_path /output/dir \
        --export_depth_images --export_color_images --export_poses --export_intrinsics
"""
import argparse
import os
from typing import Optional

from src.utils.sensor_data import SensorData


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract data from a ScanNet .sens file"
    )
    parser.add_argument(
        '--filename',
        required=True,
        help='Path to the .sens file'
    )
    parser.add_argument(
        '--output_path',
        required=True,
        help='Directory to save the exported data'
    )
    parser.add_argument(
        '--export_depth_images',
        action='store_true',
        help='Export all depth frames as 16-bit PNGs'
    )
    parser.add_argument(
        '--export_color_images',
        action='store_true',
        help='Export all color frames as JPEGs'
    )
    parser.add_argument(
        '--export_poses',
        action='store_true',
        help='Export all camera poses as 4x4 matrices'
    )
    parser.add_argument(
        '--export_intrinsics',
        action='store_true',
        help='Export camera intrinsics (4x4 matrices)'
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point for the sensor reader CLI."""
    opt = parse_args()

    os.makedirs(opt.output_path, exist_ok=True)
    print(f'Loading: {opt.filename}')
    sd = SensorData(opt.filename)
    print('Loaded successfully.')

    if opt.export_depth_images:
        sd.export_depth_images(os.path.join(opt.output_path, 'depth'))
    if opt.export_color_images:
        sd.export_color_images(os.path.join(opt.output_path, 'color'))
    if opt.export_poses:
        sd.export_poses(os.path.join(opt.output_path, 'pose'))
    if opt.export_intrinsics:
        sd.export_intrinsics(os.path.join(opt.output_path, 'intrinsic'))


if __name__ == '__main__':
    main()
