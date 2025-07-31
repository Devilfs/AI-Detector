"""Device utilities for AI Content Detection System."""

import torch
from .config_loader import load_config
from .logger import get_logger

logger = get_logger(__name__)


def get_device() -> torch.device:
    """
    Get the best available device for computation.
    
    Returns:
        torch.device: Best available device (CUDA, MPS, or CPU)
    """
    config = load_config()
    device_setting = config['inference']['device']
    
    if device_setting == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device("mps")
            logger.info("Using Apple MPS device")
        else:
            device = torch.device("cpu")
            logger.info("Using CPU device")
    else:
        device = torch.device(device_setting)
        logger.info(f"Using configured device: {device}")
    
    return device


def get_device_info() -> dict:
    """
    Get detailed device information.
    
    Returns:
        Dictionary with device information
    """
    info = {
        "device": str(get_device()),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    
    if torch.cuda.is_available():
        info["cuda_device_name"] = torch.cuda.get_device_name()
        info["cuda_memory_total"] = torch.cuda.get_device_properties(0).total_memory
    
    if hasattr(torch.backends, 'mps'):
        info["mps_available"] = torch.backends.mps.is_available()
    
    return info