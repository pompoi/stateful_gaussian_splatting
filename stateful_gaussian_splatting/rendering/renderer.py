"""
Renderer for Gaussian splatting.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Optional, Union
import math


class Camera:
    """
    Camera class for rendering.
    """
    
    def __init__(
        self,
        position: torch.Tensor,
        look_at: torch.Tensor,
        up: torch.Tensor,
        fov: float,
        near: float,
        far: float,
        resolution: Tuple[int, int],
        device: torch.device = None
    ):
        """
        Initialize the camera.
        
        Args:
            position: Camera position (x, y, z).
            look_at: Point the camera is looking at (x, y, z).
            up: Up vector (x, y, z).
            fov: Field of view in degrees.
            near: Near clipping plane.
            far: Far clipping plane.
            resolution: Resolution of the rendered image (width, height).
            device: Device to use for computation.
        """
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.position = position.to(self.device)
        self.look_at = look_at.to(self.device)
        self.up = up.to(self.device)
        self.fov = fov
        self.near = near
        self.far = far
        self.resolution = resolution
        
        # Compute view matrix
        self.view_matrix = self._compute_view_matrix()
        
        # Compute projection matrix
        self.projection_matrix = self._compute_projection_matrix()
        
    def _compute_view_matrix(self) -> torch.Tensor:
        """
        Compute the view matrix.
        
        Returns:
            View matrix of shape (4, 4).
        """
        # Compute forward, right, and up vectors
        forward = F.normalize(self.look_at - self.position, dim=0)
        right = F.normalize(torch.cross(forward, self.up), dim=0)
        up = F.normalize(torch.cross(right, forward), dim=0)
        
        # Construct view matrix
        view_matrix = torch.zeros(4, 4, device=self.device)
        
        # Rotation part
        view_matrix[0, 0:3] = right
        view_matrix[1, 0:3] = up
        view_matrix[2, 0:3] = -forward
        
        # Translation part
        view_matrix[0, 3] = -torch.dot(right, self.position)
        view_matrix[1, 3] = -torch.dot(up, self.position)
        view_matrix[2, 3] = torch.dot(forward, self.position)
        view_matrix[3, 3] = 1.0
        
        return view_matrix
    
    def _compute_projection_matrix(self) -> torch.Tensor:
        """
        Compute the projection matrix.
        
        Returns:
            Projection matrix of shape (4, 4).
        """
        aspect_ratio = self.resolution[0] / self.resolution[1]
        fov_rad = math.radians(self.fov)
        
        # Construct projection matrix
        proj_matrix = torch.zeros(4, 4, device=self.device)
        
        # Perspective projection
        f = 1.0 / math.tan(fov_rad / 2.0)
        proj_matrix[0, 0] = f / aspect_ratio
        proj_matrix[1, 1] = f
        proj_matrix[2, 2] = (self.far + self.near) / (self.near - self.far)
        proj_matrix[2, 3] = (2.0 * self.far * self.near) / (self.near - self.far)
        proj_matrix[3, 2] = -1.0
        
        return proj_matrix
    
    def get_rays(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get rays for each pixel in the image.
        
        Returns:
            Tuple of (origins, directions) where:
                - origins: Ray origins of shape (height, width, 3)
                - directions: Ray directions of shape (height, width, 3)
        """
        width, height = self.resolution
        
        # Generate pixel coordinates
        i, j = torch.meshgrid(
            torch.linspace(0, width - 1, width, device=self.device),
            torch.linspace(0, height - 1, height, device=self.device),
            indexing='ij'
        )
        
        # Convert to NDC coordinates
        x = (2.0 * i - width) / width
        y = (2.0 * j - height) / height
        
        # Create homogeneous coordinates
        ndc = torch.stack([x, y, torch.ones_like(x), torch.ones_like(x)], dim=-1)
        
        # Transform to world space
        inv_proj = torch.inverse(self.projection_matrix)
        inv_view = torch.inverse(self.view_matrix)
        
        # Apply inverse projection and view matrices
        camera_space = torch.matmul(ndc.unsqueeze(-2), inv_proj.T).squeeze(-2)
        camera_space = camera_space / camera_space[..., 3:4]
        
        world_space = torch.matmul(camera_space.unsqueeze(-2), inv_view.T).squeeze(-2)
        world_space = world_space / world_space[..., 3:4]
        
        # Extract ray origins and directions
        origins = self.position.expand(height, width, 3)
        directions = F.normalize(world_space[..., 0:3] - origins, dim=-1)
        
        return origins, directions


class Renderer(nn.Module):
    """
    Renderer for Gaussian splatting.
    """
    
    def __init__(
        self,
        fov: float = 60.0,
        near: float = 0.1,
        far: float = 100.0,
        resolution: Tuple[int, int] = (1024, 1024),
        background_color: List[float] = [0, 0, 0],
        device: torch.device = None
    ):
        """
        Initialize the renderer.
        
        Args:
            fov: Field of view in degrees.
            near: Near clipping plane.
            far: Far clipping plane.
            resolution: Resolution of the rendered image (width, height).
            background_color: Background color [r, g, b].
            device: Device to use for computation.
        """
        super().__init__()
        
        self.fov = fov
        self.near = near
        self.far = far
        self.resolution = resolution
        self.background_color = torch.tensor(background_color, dtype=torch.float32)
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.background_color = self.background_color.to(self.device)
        
    def create_camera(
        self,
        position: torch.Tensor,
        look_at: torch.Tensor = None,
        up: torch.Tensor = None
    ) -> Camera:
        """
        Create a camera for rendering.
        
        Args:
            position: Camera position (x, y, z).
            look_at: Point the camera is looking at (x, y, z). If None, looks at the origin.
            up: Up vector (x, y, z). If None, uses the y-axis.
            
        Returns:
            Camera object.
        """
        if look_at is None:
            look_at = torch.tensor([0.0, 0.0, 0.0], device=self.device)
        
        if up is None:
            up = torch.tensor([0.0, 1.0, 0.0], device=self.device)
            
        return Camera(
            position=position,
            look_at=look_at,
            up=up,
            fov=self.fov,
            near=self.near,
            far=self.far,
            resolution=self.resolution,
            device=self.device
        )
    
    def forward(
        self,
        gaussian_model: nn.Module,
        camera: Camera,
        liquid_network: Optional[nn.Module] = None,
        time_step: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Render the scene.
        
        Args:
            gaussian_model: Gaussian model.
            camera: Camera for rendering.
            liquid_network: Optional liquid neural network for stateful processing.
            time_step: Optional time step for the liquid network.
            
        Returns:
            Tuple of (rgb, depth, alpha) where:
                - rgb: RGB image of shape (3, height, width)
                - depth: Depth image of shape (1, height, width)
                - alpha: Alpha image of shape (1, height, width)
        """
        # TODO: Implement rendering logic
        # For now, just return a placeholder image
        
        width, height = self.resolution
        
        # Create placeholder images
        rgb = torch.ones(3, height, width, device=self.device) * 0.5
        depth = torch.ones(1, height, width, device=self.device) * 0.5
        alpha = torch.ones(1, height, width, device=self.device) * 0.5
        
        return rgb, depth, alpha
    
    def render_to_image(
        self,
        gaussian_model: nn.Module,
        camera: Camera,
        liquid_network: Optional[nn.Module] = None,
        time_step: Optional[torch.Tensor] = None
    ) -> np.ndarray:
        """
        Render the scene and convert to a numpy image.
        
        Args:
            gaussian_model: Gaussian model.
            camera: Camera for rendering.
            liquid_network: Optional liquid neural network for stateful processing.
            time_step: Optional time step for the liquid network.
            
        Returns:
            RGB image as a numpy array of shape (height, width, 3).
        """
        rgb, _, _ = self.forward(gaussian_model, camera, liquid_network, time_step)
        
        # Convert to numpy
        rgb_np = rgb.permute(1, 2, 0).cpu().numpy()
        
        # Clip to [0, 1]
        rgb_np = np.clip(rgb_np, 0.0, 1.0)
        
        # Convert to uint8
        rgb_np = (rgb_np * 255).astype(np.uint8)
        
        return rgb_np