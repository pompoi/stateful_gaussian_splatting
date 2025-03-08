"""
Setup script for the stateful Gaussian splatting project.

This script helps with:
1. Creating necessary directories
2. Setting up a virtual environment
3. Installing dependencies
4. Preparing the project for development
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Setup script for stateful Gaussian splatting")
    parser.add_argument("--data_dir", type=str, default="data", help="Path to the data directory")
    parser.add_argument("--venv_dir", type=str, default="venv", help="Path to the virtual environment directory")
    parser.add_argument("--skip_venv", action="store_true", help="Skip virtual environment creation")
    parser.add_argument("--skip_install", action="store_true", help="Skip dependency installation")
    return parser.parse_args()


def create_directories(data_dir):
    """Create necessary directories."""
    # Create data directory
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(os.path.join(data_dir, "FramesByNumber"), exist_ok=True)
    
    # Create output directory
    os.makedirs("output", exist_ok=True)
    
    print("Directories created successfully")


def create_virtual_environment(venv_dir):
    """Create a virtual environment."""
    if os.path.exists(venv_dir):
        print(f"Virtual environment already exists at {venv_dir}")
        return
    
    print(f"Creating virtual environment at {venv_dir}")
    
    try:
        subprocess.run([sys.executable, "-m", "venv", venv_dir], check=True)
        print("Virtual environment created successfully")
    except subprocess.CalledProcessError as e:
        print(f"Error creating virtual environment: {e}")
        sys.exit(1)


def install_dependencies(venv_dir, skip_venv):
    """Install dependencies."""
    # Get the path to the Python executable in the virtual environment
    if skip_venv:
        python = sys.executable
    else:
        if sys.platform == "win32":
            python = os.path.join(venv_dir, "Scripts", "python.exe")
        else:
            python = os.path.join(venv_dir, "bin", "python")
    
    # Check if the Python executable exists
    if not os.path.exists(python):
        print(f"Python executable not found at {python}")
        sys.exit(1)
    
    print("Installing dependencies")
    
    # Install the package in development mode
    try:
        subprocess.run([python, "-m", "pip", "install", "-e", "."], check=True)
        print("Dependencies installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"Error installing dependencies: {e}")
        sys.exit(1)


def main():
    # Parse arguments
    args = parse_args()
    
    # Create directories
    create_directories(args.data_dir)
    
    # Create virtual environment
    if not args.skip_venv:
        create_virtual_environment(args.venv_dir)
    
    # Install dependencies
    if not args.skip_install:
        install_dependencies(args.venv_dir, args.skip_venv)
    
    print("Setup complete!")
    
    # Print activation instructions
    if not args.skip_venv:
        if sys.platform == "win32":
            activate_cmd = f"{args.venv_dir}\\Scripts\\activate"
        else:
            activate_cmd = f"source {args.venv_dir}/bin/activate"
        
        print(f"\nTo activate the virtual environment, run:")
        print(f"  {activate_cmd}")
    
    # Print next steps
    print("\nNext steps:")
    print("1. Prepare your data:")
    print(f"   python scripts/prepare_data.py --source_dir /path/to/FramesByNumber --target_dir {args.data_dir}/FramesByNumber --copy --validate --create_manifest")
    print("2. Run the example:")
    print(f"   python examples/basic_usage.py --data_path {args.data_dir}/FramesByNumber")


if __name__ == "__main__":
    main()