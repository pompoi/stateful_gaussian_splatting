"""
Script to visualize data for the stateful Gaussian splatting project.

This script helps with:
1. Visualizing RGB, depth, and alpha images
2. Creating side-by-side comparisons of different cameras
3. Creating videos from image sequences
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import cv2
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize data for stateful Gaussian splatting")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the FramesByNumber directory")
    parser.add_argument("--output_dir", type=str, default="output/visualizations", help="Path to save visualizations")
    parser.add_argument("--frame", type=int, default=1, help="Frame number to visualize")
    parser.add_argument("--camera", type=str, default=None, help="Camera to visualize (e.g., CameraL_000)")
    parser.add_argument("--modality", type=str, default="rgb", choices=["rgb", "depth", "alpha", "all"], help="Modality to visualize")
    parser.add_argument("--create_video", action="store_true", help="Create a video from image sequences")
    parser.add_argument("--video_fps", type=int, default=30, help="Frames per second for the video")
    return parser.parse_args()


def visualize_image(image_path, output_path, title=None):
    """Visualize a single image."""
    # Load image
    img = Image.open(image_path)
    img_array = np.array(img)
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Display image
    if len(img_array.shape) == 3 and img_array.shape[2] == 3:
        # RGB image
        plt.imshow(img_array)
    else:
        # Grayscale image (depth or alpha)
        plt.imshow(img_array, cmap='viridis')
        plt.colorbar(label='Value')
    
    # Add title
    if title:
        plt.title(title)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    print(f"Visualization saved to {output_path}")


def visualize_modalities(frame_dir, camera, output_dir):
    """Visualize all modalities for a camera."""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get paths to modality images
    rgb_path = frame_dir / f"{camera}_rgb.png"
    depth_path = frame_dir / f"{camera}_depth.png"
    alpha_path = frame_dir / f"{camera}_alpha.png"
    
    # Visualize RGB image
    if rgb_path.exists():
        output_path = Path(output_dir) / f"{camera}_rgb.png"
        visualize_image(rgb_path, output_path, title=f"{camera} - RGB")
    
    # Visualize depth image
    if depth_path.exists():
        output_path = Path(output_dir) / f"{camera}_depth.png"
        visualize_image(depth_path, output_path, title=f"{camera} - Depth")
    
    # Visualize alpha image
    if alpha_path.exists():
        output_path = Path(output_dir) / f"{camera}_alpha.png"
        visualize_image(alpha_path, output_path, title=f"{camera} - Alpha")
    
    # Create side-by-side comparison
    if rgb_path.exists() and depth_path.exists() and alpha_path.exists():
        # Load images
        rgb_img = np.array(Image.open(rgb_path))
        depth_img = np.array(Image.open(depth_path))
        alpha_img = np.array(Image.open(alpha_path))
        
        # Ensure depth and alpha are 3-channel
        if len(depth_img.shape) == 2:
            depth_img = np.stack([depth_img] * 3, axis=2)
        if len(alpha_img.shape) == 2:
            alpha_img = np.stack([alpha_img] * 3, axis=2)
        
        # Create figure
        plt.figure(figsize=(15, 5))
        
        # Display RGB image
        plt.subplot(1, 3, 1)
        plt.imshow(rgb_img)
        plt.title("RGB")
        plt.axis('off')
        
        # Display depth image
        plt.subplot(1, 3, 2)
        plt.imshow(depth_img, cmap='viridis')
        plt.title("Depth")
        plt.axis('off')
        
        # Display alpha image
        plt.subplot(1, 3, 3)
        plt.imshow(alpha_img, cmap='viridis')
        plt.title("Alpha")
        plt.axis('off')
        
        # Save figure
        output_path = Path(output_dir) / f"{camera}_comparison.png"
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
        
        print(f"Comparison visualization saved to {output_path}")


def create_video(data_dir, output_dir, camera, modality, fps):
    """Create a video from image sequences."""
    data_path = Path(data_dir)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all frame directories
    frame_dirs = sorted([d for d in data_path.iterdir() if d.is_dir()], key=lambda x: int(x.name))
    
    if len(frame_dirs) == 0:
        print(f"No frame directories found in {data_path}")
        return
    
    print(f"Found {len(frame_dirs)} frame directories")
    
    # Check if the first frame has the required image
    first_frame_dir = frame_dirs[0]
    image_path = first_frame_dir / f"{camera}_{modality}.png"
    
    if not image_path.exists():
        print(f"Image {image_path} does not exist")
        return
    
    # Get image dimensions
    first_image = cv2.imread(str(image_path))
    height, width, channels = first_image.shape
    
    # Create video writer
    output_path = Path(output_dir) / f"{camera}_{modality}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    # Add frames to video
    for frame_dir in tqdm(frame_dirs, desc="Creating video"):
        image_path = frame_dir / f"{camera}_{modality}.png"
        
        if not image_path.exists():
            print(f"Warning: Image {image_path} does not exist, skipping")
            continue
        
        # Read image
        image = cv2.imread(str(image_path))
        
        # Add to video
        video_writer.write(image)
    
    # Release video writer
    video_writer.release()
    
    print(f"Video saved to {output_path}")


def main():
    # Parse arguments
    args = parse_args()
    
    # Get data directory
    data_path = Path(args.data_dir)
    
    if not data_path.exists():
        raise ValueError(f"Data directory {data_path} does not exist")
    
    # Get frame directory
    frame_dir = data_path / str(args.frame)
    
    if not frame_dir.exists():
        raise ValueError(f"Frame directory {frame_dir} does not exist")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # If camera is not specified, find all cameras
    if args.camera is None:
        cameras = set()
        
        for file_path in frame_dir.glob("*_rgb.png"):
            camera_name = file_path.stem.rsplit('_', 1)[0]  # Remove _rgb suffix
            cameras.add(camera_name)
        
        cameras = sorted(list(cameras))
        
        if len(cameras) == 0:
            raise ValueError(f"No cameras found in frame directory {frame_dir}")
        
        print(f"Found {len(cameras)} cameras: {', '.join(cameras)}")
    else:
        cameras = [args.camera]
    
    # Visualize data
    if args.create_video:
        # Create video for each camera
        for camera in cameras:
            create_video(args.data_dir, output_dir, camera, args.modality, args.video_fps)
    else:
        # Visualize images for each camera
        for camera in cameras:
            if args.modality == "all":
                visualize_modalities(frame_dir, camera, output_dir)
            else:
                image_path = frame_dir / f"{camera}_{args.modality}.png"
                
                if not image_path.exists():
                    print(f"Warning: Image {image_path} does not exist, skipping")
                    continue
                
                output_path = output_dir / f"{camera}_{args.modality}.png"
                visualize_image(image_path, output_path, title=f"{camera} - {args.modality}")
    
    print("Done!")


if __name__ == "__main__":
    main()