"""Configuration loader for AI Content Detection System."""

import yaml
import os
from pathlib import Path
from typing import Dict, Any


def load_config(config_path: str = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file. If None, uses default config.
        
    Returns:
        Dictionary containing configuration.
    """
    if config_path is None:
        # Get project root directory
        project_root = Path(__file__).parent.parent.parent
        config_path = project_root / "config" / "config.yaml"
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    return config


def get_model_config(model_type: str, model_name: str) -> Dict[str, Any]:
    """
    Get specific model configuration.
    
    Args:
        model_type: Type of model ('text' or 'image')
        model_name: Name of the specific model
        
    Returns:
        Model configuration dictionary
    """
    config = load_config()
    return config['models'][model_type][model_name]


def get_api_config() -> Dict[str, Any]:
    """Get API configuration."""
    config = load_config()
    return config['api']


def get_training_config() -> Dict[str, Any]:
    """Get training configuration."""
    config = load_config()
    return config['training']