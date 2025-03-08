"""
Basic usage example for the stateful Gaussian splatting project.
"""

import os
import torch
import numpy as np
from PIL import Image
import argparse
from pathlib import Path

from stateful_gaussian_splatting import StatefulGaussianSplatter


def parse_args():
    parser = argparse.ArgumentParser(description="Basic usage example for stateful Gaussian splatting")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the FramesByNumber directory")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory to save outputs to")
    parser.add_argument("--config_path", type=str, default=None, help="Path to the configuration file")
    parser.add_argument("--device", type=str, default=None, help="Device to use (cuda, cpu)")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to train for")
    return parser.parse_args()


def main():
    # Parse arguments
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize the StatefulGaussianSplatter
    splatter = StatefulGaussianSplatter(
        data_path=args.data_path,
        config_path=args.config_path,
        device=args.device
    )
    
    # Train the model
    splatter.train(epochs=args.epochs, output_dir=args.output_dir)
    
    # Generate 3D representation
    splatter.generate_representation(output_path=os.path.join(args.output_dir, "model"))
    
    # Render from different viewpoints
    viewpoints = [
        [0, 0, 5],  # Front
        [5, 0, 0],  # Right
        [0, 5, 0],  # Top
        [3, 3, 3]   # Diagonal
    ]
    
    for i, viewpoint in enumerate(viewpoints):
        output_path = os.path.join(args.output_dir, f"render_view_{i}.png")
        splatter.render(viewpoint=viewpoint, output_path=output_path)
        print(f"Rendered view {i} saved to {output_path}")
    
    print("Done!")


if __name__ == "__main__":
    main()