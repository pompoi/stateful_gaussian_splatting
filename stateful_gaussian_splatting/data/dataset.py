"""
Dataset classes for multi-view camera images.
"""

import os
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
from torch.utils.data import Dataset, DataLoader


class MultiViewDataset(Dataset):
    """
    Dataset for multi-view camera images with different modalities (RGB, depth, alpha).
    
    This dataset is designed to work with the FramesByNumber directory structure:
    
    FramesByNumber/
      1/
        CameraL_000_rgb.png
        CameraL_000_depth.png
        CameraL_000_alpha.png
        CameraR_000_rgb.png
        ...
      2/
        ...
      ...
      300/
        ...
    """
    
    def __init__(
        self,
        root_dir: str,
        train_frames: List[int] = [1, 250],
        val_frames: List[int] = [251, 300],
        cameras: List[str] = None,
        modalities: List[str] = ["rgb", "depth", "alpha"],
        image_size: Tuple[int, int] = (512, 512),
        transform = None,
        split: str = "train"
    ):
        """
        Initialize the multi-view dataset.
        
        Args:
            root_dir: Path to the FramesByNumber directory.
            train_frames: Range of frames to use for training [start, end].
            val_frames: Range of frames to use for validation [start, end].
            cameras: List of camera names to use. If None, uses all cameras found.
            modalities: List of modalities to use (rgb, depth, alpha).
            image_size: Size to resize images to (width, height).
            transform: Optional transform to apply to the images.
            split: Dataset split to use (train, val).
        """
        self.root_dir = Path(root_dir)
        self.train_frames = range(train_frames[0], train_frames[1] + 1)
        self.val_frames = range(val_frames[0], val_frames[1] + 1)
        self.modalities = modalities
        self.image_size = image_size
        self.transform = transform
        self.split = split
        
        # Set frames based on split
        self.frames = self.train_frames if split == "train" else self.val_frames
        
        # Find all cameras if not provided
        if cameras is None:
            self.cameras = self._find_cameras()
        else:
            self.cameras = cameras
            
        # Create frame-camera pairs
        self.samples = self._create_samples()
        
        print(f"Initialized {split} dataset with {len(self.samples)} samples")
        print(f"Using {len(self.cameras)} cameras and {len(self.modalities)} modalities")
        
    def _find_cameras(self) -> List[str]:
        """
        Find all cameras in the dataset.
        
        Returns:
            List of camera names.
        """
        cameras = set()
        
        # Check the first frame to find all cameras
        frame_dir = self.root_dir / str(self.frames[0])
        if not frame_dir.exists():
            raise ValueError(f"Frame directory {frame_dir} does not exist")
            
        for file_path in frame_dir.glob("*_rgb.png"):
            camera_name = file_path.stem.rsplit('_', 1)[0]  # Remove _rgb suffix
            cameras.add(camera_name)
            
        return sorted(list(cameras))
    
    def _create_samples(self) -> List[Dict]:
        """
        Create a list of samples, where each sample is a dictionary containing:
        - frame_num: Frame number
        - camera_name: Camera name
        - paths: Dictionary mapping modality to file path
        
        Returns:
            List of sample dictionaries.
        """
        samples = []
        
        for frame_num in self.frames:
            frame_dir = self.root_dir / str(frame_num)
            
            if not frame_dir.exists():
                print(f"Warning: Frame directory {frame_dir} does not exist, skipping")
                continue
                
            for camera_name in self.cameras:
                paths = {}
                valid_sample = True
                
                # Check if all required modalities exist
                for modality in self.modalities:
                    file_path = frame_dir / f"{camera_name}_{modality}.png"
                    
                    if not file_path.exists():
                        print(f"Warning: File {file_path} does not exist, skipping sample")
                        valid_sample = False
                        break
                        
                    paths[modality] = file_path
                
                if valid_sample:
                    samples.append({
                        "frame_num": frame_num,
                        "camera_name": camera_name,
                        "paths": paths
                    })
        
        return samples
    
    def __len__(self) -> int:
        """
        Get the number of samples in the dataset.
        
        Returns:
            Number of samples.
        """
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample.
            
        Returns:
            Dictionary containing:
            - frame_num: Frame number
            - camera_name: Camera name
            - images: Dictionary mapping modality to image tensor
        """
        sample = self.samples[idx]
        frame_num = sample["frame_num"]
        camera_name = sample["camera_name"]
        paths = sample["paths"]
        
        # Load images for each modality
        images = {}
        for modality, path in paths.items():
            img = Image.open(path)
            
            # Resize image
            img = img.resize(self.image_size)
            
            # Convert to tensor
            img_tensor = torch.from_numpy(np.array(img))
            
            # Handle different modalities
            if modality == "rgb":
                # Convert to float and normalize to [0, 1]
                img_tensor = img_tensor.float() / 255.0
                # Ensure 3 channels (H, W, 3)
                if img_tensor.ndim == 2:
                    img_tensor = img_tensor.unsqueeze(-1).repeat(1, 1, 3)
                # Permute to (3, H, W)
                img_tensor = img_tensor.permute(2, 0, 1)
            elif modality in ["depth", "alpha"]:
                # Convert to float and normalize to [0, 1]
                img_tensor = img_tensor.float() / 255.0
                # Ensure 1 channel (H, W, 1)
                if img_tensor.ndim == 3:
                    img_tensor = img_tensor.mean(dim=-1, keepdim=True)
                elif img_tensor.ndim == 2:
                    img_tensor = img_tensor.unsqueeze(-1)
                # Permute to (1, H, W)
                img_tensor = img_tensor.permute(2, 0, 1)
            
            images[modality] = img_tensor
            
        # Apply transform if provided
        if self.transform is not None:
            for modality in images:
                images[modality] = self.transform(images[modality])
        
        return {
            "frame_num": frame_num,
            "camera_name": camera_name,
            "images": images
        }
    
    def get_dataloader(
        self,
        batch_size: int = 4,
        shuffle: bool = True,
        num_workers: int = 4
    ) -> DataLoader:
        """
        Get a data loader for the dataset.
        
        Args:
            batch_size: Batch size.
            shuffle: Whether to shuffle the dataset.
            num_workers: Number of worker processes.
            
        Returns:
            DataLoader for the dataset.
        """
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True
        )