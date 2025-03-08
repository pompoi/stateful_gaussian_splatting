"""
Script to prepare data for the stateful Gaussian splatting project.

This script helps with:
1. Copying data from the original FramesByNumber directory to the project's data directory
2. Validating the data structure
3. Creating a data manifest file
"""

import os
import shutil
import argparse
import json
from pathlib import Path
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare data for stateful Gaussian splatting")
    parser.add_argument("--source_dir", type=str, required=True, help="Path to the source FramesByNumber directory")
    parser.add_argument("--target_dir", type=str, default="data/FramesByNumber", help="Path to the target data directory")
    parser.add_argument("--copy", action="store_true", help="Copy data from source to target")
    parser.add_argument("--validate", action="store_true", help="Validate data structure")
    parser.add_argument("--create_manifest", action="store_true", help="Create data manifest file")
    return parser.parse_args()


def copy_data(source_dir, target_dir):
    """Copy data from source to target directory."""
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    
    if not source_path.exists():
        raise ValueError(f"Source directory {source_path} does not exist")
    
    # Create target directory if it doesn't exist
    os.makedirs(target_path, exist_ok=True)
    
    # Get all frame directories
    frame_dirs = [d for d in source_path.iterdir() if d.is_dir()]
    
    print(f"Found {len(frame_dirs)} frame directories")
    
    # Copy each frame directory
    for frame_dir in tqdm(frame_dirs, desc="Copying frame directories"):
        frame_num = frame_dir.name
        target_frame_dir = target_path / frame_num
        
        # Create target frame directory if it doesn't exist
        os.makedirs(target_frame_dir, exist_ok=True)
        
        # Copy all files in the frame directory
        for file_path in frame_dir.glob("*"):
            if file_path.is_file():
                shutil.copy2(file_path, target_frame_dir / file_path.name)
    
    print(f"Data copied from {source_path} to {target_path}")


def validate_data(data_dir):
    """Validate data structure."""
    data_path = Path(data_dir)
    
    if not data_path.exists():
        raise ValueError(f"Data directory {data_path} does not exist")
    
    # Get all frame directories
    frame_dirs = [d for d in data_path.iterdir() if d.is_dir()]
    
    print(f"Found {len(frame_dirs)} frame directories")
    
    # Check each frame directory
    valid_frames = 0
    invalid_frames = 0
    
    for frame_dir in tqdm(frame_dirs, desc="Validating frame directories"):
        frame_num = frame_dir.name
        
        # Check if the frame directory contains any files
        files = list(frame_dir.glob("*"))
        
        if len(files) == 0:
            print(f"Warning: Frame directory {frame_num} is empty")
            invalid_frames += 1
            continue
        
        # Check if the frame directory contains the expected files
        rgb_files = list(frame_dir.glob("*_rgb.png"))
        depth_files = list(frame_dir.glob("*_depth.png"))
        alpha_files = list(frame_dir.glob("*_alpha.png"))
        
        if len(rgb_files) == 0:
            print(f"Warning: Frame directory {frame_num} has no RGB files")
            invalid_frames += 1
            continue
        
        # Check if all cameras have all modalities
        cameras = set()
        
        for file_path in rgb_files:
            camera_name = file_path.stem.rsplit('_', 1)[0]  # Remove _rgb suffix
            cameras.add(camera_name)
        
        missing_modalities = False
        
        for camera in cameras:
            rgb_file = frame_dir / f"{camera}_rgb.png"
            depth_file = frame_dir / f"{camera}_depth.png"
            alpha_file = frame_dir / f"{camera}_alpha.png"
            
            if not rgb_file.exists():
                print(f"Warning: Camera {camera} in frame {frame_num} has no RGB file")
                missing_modalities = True
            
            if not depth_file.exists():
                print(f"Warning: Camera {camera} in frame {frame_num} has no depth file")
                missing_modalities = True
            
            if not alpha_file.exists():
                print(f"Warning: Camera {camera} in frame {frame_num} has no alpha file")
                missing_modalities = True
        
        if missing_modalities:
            invalid_frames += 1
        else:
            valid_frames += 1
    
    print(f"Validation complete: {valid_frames} valid frames, {invalid_frames} invalid frames")
    
    return valid_frames, invalid_frames


def create_manifest(data_dir):
    """Create data manifest file."""
    data_path = Path(data_dir)
    
    if not data_path.exists():
        raise ValueError(f"Data directory {data_path} does not exist")
    
    # Get all frame directories
    frame_dirs = [d for d in data_path.iterdir() if d.is_dir()]
    
    print(f"Found {len(frame_dirs)} frame directories")
    
    # Create manifest
    manifest = {
        "num_frames": len(frame_dirs),
        "frames": {}
    }
    
    # Process each frame directory
    for frame_dir in tqdm(frame_dirs, desc="Creating manifest"):
        frame_num = frame_dir.name
        
        # Get all files in the frame directory
        files = list(frame_dir.glob("*"))
        
        # Get all cameras
        cameras = set()
        
        for file_path in files:
            if "_rgb.png" in file_path.name:
                camera_name = file_path.stem.rsplit('_', 1)[0]  # Remove _rgb suffix
                cameras.add(camera_name)
        
        # Create frame entry
        frame_entry = {
            "num_cameras": len(cameras),
            "cameras": {}
        }
        
        # Create camera entries
        for camera in cameras:
            rgb_file = frame_dir / f"{camera}_rgb.png"
            depth_file = frame_dir / f"{camera}_depth.png"
            alpha_file = frame_dir / f"{camera}_alpha.png"
            
            camera_entry = {
                "rgb": str(rgb_file.relative_to(data_path)) if rgb_file.exists() else None,
                "depth": str(depth_file.relative_to(data_path)) if depth_file.exists() else None,
                "alpha": str(alpha_file.relative_to(data_path)) if alpha_file.exists() else None
            }
            
            frame_entry["cameras"][camera] = camera_entry
        
        manifest["frames"][frame_num] = frame_entry
    
    # Save manifest
    manifest_path = data_path.parent / "manifest.json"
    
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    
    print(f"Manifest saved to {manifest_path}")


def main():
    # Parse arguments
    args = parse_args()
    
    # Copy data if requested
    if args.copy:
        copy_data(args.source_dir, args.target_dir)
    
    # Validate data if requested
    if args.validate:
        validate_data(args.target_dir)
    
    # Create manifest if requested
    if args.create_manifest:
        create_manifest(args.target_dir)
    
    print("Done!")


if __name__ == "__main__":
    main()