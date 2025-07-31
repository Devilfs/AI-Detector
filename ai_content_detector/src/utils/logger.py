"""Logging utilities for AI Content Detection System."""

import logging
import os
from pathlib import Path
from .config_loader import load_config


def setup_logger(name: str = None) -> logging.Logger:
    """
    Set up logger with configuration from config file.
    
    Args:
        name: Logger name. If None, uses root logger.
        
    Returns:
        Configured logger instance.
    """
    config = load_config()
    logging_config = config['logging']
    
    # Create logs directory if it doesn't exist
    log_file = logging_config['log_file']
    log_dir = Path(log_file).parent
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, logging_config['level']),
        format=logging_config['format'],
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    if name:
        return logging.getLogger(name)
    else:
        return logging.getLogger()


def get_logger(name: str) -> logging.Logger:
    """Get logger by name."""
    return logging.getLogger(name)