"""
Configuration utilities for the stateful Gaussian splatting project.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.
    
    Args:
        config_path: Path to the configuration file. If None, uses the default config.
        
    Returns:
        Dictionary containing the configuration.
    """
    if config_path is None:
        # Use default config
        default_config_path = Path(__file__).parent.parent.parent / "config" / "default.yaml"
        config_path = default_config_path
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    return config


def save_config(config: Dict[str, Any], config_path: str) -> None:
    """
    Save configuration to a YAML file.
    
    Args:
        config: Configuration dictionary.
        config_path: Path to save the configuration to.
    """
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)