#!/usr/bin/env python3
"""
Sample ScanNet Images Script

This script samples every Nth image from a ScanNet scene's color folder and copies
the corresponding pose and depth data to create a manageable dataset for annotation.

Usage:
    python scripts/sample_scannet_images.py <scene_id> [--sample_rate N] [--output_dir dir]
"""

import os
import shutil
import argparse
from pathlib import Path
import glob

def sample_scannet_images(scene_id, sample_rate=150, output_dir="output"):
    """
    Sample every Nth image from ScanNet scene and copy corresponding data.
    
    Args:
        scene_id (str): Scene ID (e.g., 'scene0000_00')
        sample_rate (int): Sample every Nth image (default: 150)
        output_dir (str): Output directory name (default: 'output')
    """
    
    # Define paths
    base_path = Path("data/scans") / scene_id
    color_path = base_path / "color"
    depth_path = base_path / "depth"
    pose_path = base_path / "pose"
    output_path = base_path / output_dir
    
    # Check if source directories exist
    if not color_path.exists():
        print(f"❌ Error: Color directory not found at {color_path}")
        return False
    
    if not depth_path.exists():
        print(f"❌ Error: Depth directory not found at {depth_path}")
        return False
    
    if not pose_path.exists():
        print(f"❌ Error: Pose directory not found at {pose_path}")
        return False
    
    # Get all color images and sort them
    color_files = sorted(glob.glob(str(color_path / "*.jpg")))
    if not color_files:
        print(f"❌ Error: No color images found in {color_path}")
        return False
    
    print(f"📊 Found {len(color_files)} color images")
    print(f"🎯 Sampling every {sample_rate}th image...")
    
    # Sample images
    sampled_files = color_files[::sample_rate]
    print(f"✅ Selected {len(sampled_files)} images for annotation")
    
    # Create output directories
    output_color = output_path / "color"
    output_depth = output_path / "depth"
    output_pose = output_path / "pose"
    
    for dir_path in [output_color, output_depth, output_pose]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Copy files
    camera_poses = {}
    
    for i, color_file in enumerate(sampled_files, 1):
        # Get frame ID from filename (e.g., "005500.jpg" -> "005500")
        frame_id = Path(color_file).stem
        
        # Define corresponding files
        depth_file = depth_path / f"{frame_id}.png"
        pose_file = pose_path / f"{frame_id}.txt"
        
        # Copy color image
        dest_color = output_color / f"view_{i}.jpg"
        shutil.copy2(color_file, dest_color)
        
        # Copy depth image if exists
        if depth_file.exists():
            dest_depth = output_depth / f"view_{i}.png"
            shutil.copy2(depth_file, dest_depth)
        
        # Copy pose file if exists
        if pose_file.exists():
            dest_pose = output_pose / f"view_{i}.txt"
            shutil.copy2(pose_file, dest_pose)
            
            # Read pose matrix for camera_pose.json
            try:
                pose_matrix = []
                with open(pose_file, 'r') as f:
                    for line in f:
                        pose_matrix.append([float(x) for x in line.strip().split()])
                camera_poses[f"view_{i}"] = pose_matrix
            except Exception as e:
                print(f"⚠️ Warning: Could not read pose file {pose_file}: {e}")
        
        print(f"📁 Copied frame {frame_id} -> view_{i}")
    
    # Save camera poses
    import json
    pose_json_path = output_path / "camera_pose.json"
    with open(pose_json_path, 'w') as f:
        json.dump(camera_poses, f, indent=2)
    
    print(f"✅ Sampling complete!")
    print(f"📁 Output directory: {output_path}")
    print(f"📊 Total images: {len(sampled_files)}")
    print(f"📄 Camera poses saved to: {pose_json_path}")
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Sample ScanNet images for annotation")
    parser.add_argument("scene_id", type=str, help="Scene ID (e.g., scene0000_00)")
    parser.add_argument("--sample_rate", type=int, default=150, 
                       help="Sample every Nth image (default: 150)")
    parser.add_argument("--output_dir", type=str, default="output",
                       help="Output directory name (default: output)")
    
    args = parser.parse_args()
    
    print(f"🚀 Sampling ScanNet images for scene: {args.scene_id}")
    print(f"📊 Sample rate: every {args.sample_rate}th image")
    print(f"📁 Output directory: {args.output_dir}")
    print()
    
    success = sample_scannet_images(
        args.scene_id, 
        args.sample_rate, 
        args.output_dir
    )
    
    if success:
        print()
        print("🎉 Ready to run the annotation tool!")
        print("Run: streamlit run app/app.py")
    else:
        print("❌ Sampling failed. Please check the error messages above.")

if __name__ == "__main__":
    main() 
