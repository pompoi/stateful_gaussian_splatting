"""
Script to copy data from the local FramesByNumber directory to the project.

This script is specifically designed to work with the data structure created in the
previous steps, where we have a FramesByNumber directory with 300 frame folders,
each containing RGB, depth, and alpha images for multiple cameras.
"""

import os
import shutil
import argparse
from pathlib import Path
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Copy data from local FramesByNumber directory to the project")
    parser.add_argument("--source_dir", type=str, default="FramesByNumber", help="Path to the source FramesByNumber directory")
    parser.add_argument("--target_dir", type=str, default="data/FramesByNumber", help="Path to the target data directory")
    parser.add_argument("--frames", type=str, default="1-300", help="Range of frames to copy (e.g., '1-300' or '1,5,10')")
    parser.add_argument("--cameras", type=str, default=None, help="Comma-separated list of cameras to copy (e.g., 'CameraL_000,CameraR_000')")
    parser.add_argument("--modalities", type=str, default="rgb,depth,alpha", help="Comma-separated list of modalities to copy")
    parser.add_argument("--sample_rate", type=int, default=1, help="Sample every Nth frame (e.g., 2 means every other frame)")
    return parser.parse_args()


def parse_frames(frames_str):
    """Parse the frames string into a list of frame numbers."""
    frames = []
    
    if "-" in frames_str:
        # Range of frames
        start, end = frames_str.split("-")
        frames = list(range(int(start), int(end) + 1))
    else:
        # List of frames
        frames = [int(f) for f in frames_str.split(",")]
    
    return frames


def copy_data(source_dir, target_dir, frames, cameras=None, modalities=None, sample_rate=1):
    """Copy data from source to target directory."""
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    
    if not source_path.exists():
        raise ValueError(f"Source directory {source_path} does not exist")
    
    # Create target directory if it doesn't exist
    os.makedirs(target_path, exist_ok=True)
    
    # Parse modalities
    if modalities is None:
        modalities = ["rgb", "depth", "alpha"]
    elif isinstance(modalities, str):
        modalities = modalities.split(",")
    
    # Get frame directories to copy
    frame_dirs = []
    
    for frame_num in frames:
        if frame_num % sample_rate == 0:  # Apply sample rate
            frame_dir = source_path / str(frame_num)
            if frame_dir.exists():
                frame_dirs.append(frame_dir)
            else:
                print(f"Warning: Frame directory {frame_dir} does not exist, skipping")
    
    print(f"Found {len(frame_dirs)} frame directories to copy")
    
    # Copy each frame directory
    for frame_dir in tqdm(frame_dirs, desc="Copying frame directories"):
        frame_num = frame_dir.name
        target_frame_dir = target_path / frame_num
        
        # Create target frame directory if it doesn't exist
        os.makedirs(target_frame_dir, exist_ok=True)
        
        # Find all cameras if not specified
        if cameras is None:
            camera_names = set()
            
            for file_path in frame_dir.glob("*_rgb.png"):
                camera_name = file_path.stem.rsplit('_', 1)[0]  # Remove _rgb suffix
                camera_names.add(camera_name)
            
            camera_list = sorted(list(camera_names))
        else:
            camera_list = cameras.split(",")
        
        # Copy files for each camera and modality
        for camera in camera_list:
            for modality in modalities:
                source_file = frame_dir / f"{camera}_{modality}.png"
                target_file = target_frame_dir / f"{camera}_{modality}.png"
                
                if source_file.exists():
                    shutil.copy2(source_file, target_file)
                else:
                    print(f"Warning: File {source_file} does not exist, skipping")
    
    print(f"Data copied from {source_path} to {target_path}")
    print(f"Copied {len(frame_dirs)} frame directories")


def main():
    # Parse arguments
    args = parse_args()
    
    # Parse frames
    frames = parse_frames(args.frames)
    
    # Copy data
    copy_data(
        source_dir=args.source_dir,
        target_dir=args.target_dir,
        frames=frames,
        cameras=args.cameras,
        modalities=args.modalities,
        sample_rate=args.sample_rate
    )
    
    print("Done!")


if __name__ == "__main__":
    main()