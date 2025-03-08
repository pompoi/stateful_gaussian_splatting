"""
Main StatefulGaussianSplatter class for the stateful Gaussian splatting project.
"""

import os
import yaml
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Union, Optional, Tuple

from .utils.config import load_config
from .models.liquid_network import LiquidNetwork
from .models.gaussian_model import GaussianModel
from .data.dataset import MultiViewDataset
from .rendering.renderer import Renderer


class StatefulGaussianSplatter:
    """
    Main class for stateful Gaussian splatting using liquid neural networks.
    
    This class handles the entire pipeline from data loading to training and rendering.
    """
    
    def __init__(
        self, 
        data_path: str = None,
        config_path: str = None,
        device: str = None
    ):
        """
        Initialize the StatefulGaussianSplatter.
        
        Args:
            data_path: Path to the data directory containing the FramesByNumber structure.
            config_path: Path to the configuration file. If None, uses the default config.
            device: Device to use for computation. If None, uses CUDA if available, else CPU.
        """
        # Load configuration
        self.config = load_config(config_path)
        
        # Override data path if provided
        if data_path is not None:
            self.config['data']['root_dir'] = data_path
            
        # Set device
        if device is not None:
            self.config['system']['device'] = device
        self.device = torch.device(self.config['system']['device'] 
                                  if torch.cuda.is_available() and self.config['system']['device'] == 'cuda' 
                                  else 'cpu')
        
        # Set random seed for reproducibility
        torch.manual_seed(self.config['system']['seed'])
        np.random.seed(self.config['system']['seed'])
        
        # Initialize models
        self._init_models()
        
        # Initialize dataset
        self._init_dataset()
        
        # Initialize renderer
        self._init_renderer()
        
        print(f"StatefulGaussianSplatter initialized with device: {self.device}")
        print(f"Data path: {self.config['data']['root_dir']}")
        
    def _init_models(self):
        """Initialize the Gaussian model and Liquid Neural Network."""
        # Initialize Gaussian model
        self.gaussian_model = GaussianModel(
            num_gaussians=self.config['model']['gaussian']['num_gaussians'],
            initial_radius=self.config['model']['gaussian']['initial_radius'],
            opacity_init=self.config['model']['gaussian']['opacity_init'],
            sh_degree=self.config['model']['gaussian']['sh_degree'],
            device=self.device
        )
        
        # Initialize Liquid Neural Network if enabled
        if self.config['model']['liquid_network']['enabled']:
            self.liquid_network = LiquidNetwork(
                hidden_size=self.config['model']['liquid_network']['hidden_size'],
                num_layers=self.config['model']['liquid_network']['num_layers'],
                activation=self.config['model']['liquid_network']['activation'],
                dropout=self.config['model']['liquid_network']['dropout'],
                time_embedding_dim=self.config['model']['liquid_network']['time_embedding_dim'],
                use_attention=self.config['model']['liquid_network']['use_attention'],
                attention_heads=self.config['model']['liquid_network']['attention_heads'],
                device=self.device
            )
        else:
            self.liquid_network = None
    
    def _init_dataset(self):
        """Initialize the multi-view dataset."""
        self.dataset = MultiViewDataset(
            root_dir=self.config['data']['root_dir'],
            train_frames=self.config['data']['train_frames'],
            val_frames=self.config['data']['val_frames'],
            cameras=self.config['data']['cameras'],
            modalities=self.config['data']['modalities'],
            image_size=self.config['data']['image_size']
        )
        
    def _init_renderer(self):
        """Initialize the renderer."""
        self.renderer = Renderer(
            fov=self.config['model']['camera']['fov'],
            near=self.config['model']['camera']['near'],
            far=self.config['model']['camera']['far'],
            resolution=self.config['rendering']['resolution'],
            background_color=self.config['rendering']['background_color'],
            device=self.device
        )
    
    def train(self, epochs: int = None, output_dir: str = "output"):
        """
        Train the stateful Gaussian splatting model.
        
        Args:
            epochs: Number of epochs to train for. If None, uses the value from config.
            output_dir: Directory to save outputs to.
        """
        if epochs is None:
            epochs = self.config['training']['num_epochs']
            
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # TODO: Implement training loop
        print(f"Training for {epochs} epochs...")
        print("Training not yet implemented.")
        
    def generate_representation(self, output_path: str = "output/model"):
        """
        Generate and save the 3D representation.
        
        Args:
            output_path: Path to save the model to.
        """
        # TODO: Implement representation generation
        print(f"Generating 3D representation and saving to {output_path}...")
        print("Representation generation not yet implemented.")
        
    def render(self, viewpoint: List[float] = None, output_path: str = "output/render.png"):
        """
        Render the scene from a specific viewpoint.
        
        Args:
            viewpoint: Camera position [x, y, z].
            output_path: Path to save the rendered image to.
        """
        # TODO: Implement rendering
        print(f"Rendering from viewpoint {viewpoint} and saving to {output_path}...")
        print("Rendering not yet implemented.")
        
    def load_checkpoint(self, checkpoint_path: str):
        """
        Load a checkpoint.
        
        Args:
            checkpoint_path: Path to the checkpoint file.
        """
        # TODO: Implement checkpoint loading
        print(f"Loading checkpoint from {checkpoint_path}...")
        print("Checkpoint loading not yet implemented.")
        
    def save_checkpoint(self, checkpoint_path: str):
        """
        Save a checkpoint.
        
        Args:
            checkpoint_path: Path to save the checkpoint to.
        """
        # TODO: Implement checkpoint saving
        print(f"Saving checkpoint to {checkpoint_path}...")
        print("Checkpoint saving not yet implemented.")