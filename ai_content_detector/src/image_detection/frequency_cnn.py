"""Frequency domain CNN for AI-generated image detection."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
from typing import Dict, List, Any, Optional, Union
import cv2

from ..utils import get_device, get_logger, get_model_config

logger = get_logger(__name__)


class FrequencyBlock(nn.Module):
    """Block for processing frequency domain features."""
    
    def __init__(self, in_channels: int, out_channels: int):
        super(FrequencyBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout2d(0.1)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.dropout(x)
        return x


class FrequencyCNNModel(nn.Module):
    """Custom CNN for frequency domain analysis."""
    
    def __init__(self, input_channels: int = 3, num_classes: int = 2):
        super(FrequencyCNNModel, self).__init__()
        
        # Frequency processing blocks
        self.freq_block1 = FrequencyBlock(input_channels, 32)
        self.freq_block2 = FrequencyBlock(32, 64)
        self.freq_block3 = FrequencyBlock(64, 128)
        self.freq_block4 = FrequencyBlock(128, 256)
        
        # Global pooling and classifier
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.freq_block1(x)
        x = self.freq_block2(x)
        x = self.freq_block3(x)
        x = self.freq_block4(x)
        
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        
        return x


class FrequencyCNN:
    """
    Frequency domain CNN for detecting AI-generated images.
    
    Analyzes frequency domain characteristics that differ between
    real and AI-generated images, including DCT coefficients
    and spectral patterns.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize frequency CNN.
        
        Args:
            model_path: Path to pre-trained model. If None, uses base model.
        """
        self.config = get_model_config('image', 'frequency_cnn')
        self.device = get_device()
        
        # Initialize model
        self.model = FrequencyCNNModel(
            input_channels=self.config['input_channels'],
            num_classes=self.config['num_classes']
        )
        
        if model_path:
            self._load_model(model_path)
        
        self.model.to(self.device)
        self.model.eval()
        
        logger.info("Initialized frequency domain CNN")
    
    def _load_model(self, model_path: str):
        """Load pre-trained model weights."""
        try:
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            logger.info(f"Loaded frequency CNN weights from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load model weights: {e}")
            raise
    
    def extract_frequency_features(self, image: Union[Image.Image, np.ndarray]) -> np.ndarray:
        """
        Extract frequency domain features from image.
        
        Args:
            image: Input image
            
        Returns:
            Frequency domain features
        """
        # Convert to numpy array if PIL Image
        if isinstance(image, Image.Image):
            img_array = np.array(image)
        else:
            img_array = image
        
        # Ensure RGB format
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            # Convert to grayscale for some frequency analysis
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        
        features = []
        
        # 1. DCT (Discrete Cosine Transform) features
        dct = cv2.dct(np.float32(gray))
        features.append(dct)
        
        # 2. FFT (Fast Fourier Transform) features
        fft = np.fft.fft2(gray)
        fft_magnitude = np.abs(fft)
        fft_phase = np.angle(fft)
        features.extend([fft_magnitude, fft_phase])
        
        # 3. Log-polar transform for rotation invariance
        h, w = gray.shape
        center = (w // 2, h // 2)
        log_polar = cv2.logPolar(gray.astype(np.float32), center, 40, cv2.WARP_FILL_OUTLIERS)
        features.append(log_polar)
        
        # Stack and normalize features
        freq_features = np.stack(features, axis=-1)
        
        # Resize to consistent size
        freq_features = cv2.resize(freq_features, (224, 224))
        
        return freq_features
    
    def preprocess_image(self, image: Union[Image.Image, np.ndarray, str]) -> torch.Tensor:
        """
        Preprocess image for frequency analysis.
        
        Args:
            image: Input image
            
        Returns:
            Preprocessed tensor with frequency features
        """
        # Load image if path provided
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert('RGB')
        
        # Extract frequency features
        freq_features = self.extract_frequency_features(image)
        
        # Normalize features
        freq_features = (freq_features - freq_features.mean()) / (freq_features.std() + 1e-8)
        
        # Convert to tensor
        if len(freq_features.shape) == 2:
            freq_features = np.expand_dims(freq_features, -1)
        
        # Transpose to (C, H, W) format
        freq_features = np.transpose(freq_features, (2, 0, 1))
        
        tensor = torch.FloatTensor(freq_features)
        tensor = tensor.unsqueeze(0).to(self.device)
        
        return tensor
    
    def predict(self, image: Union[Image.Image, np.ndarray, str]) -> Dict[str, Any]:
        """
        Predict if image is AI-generated based on frequency analysis.
        
        Args:
            image: Input image
            
        Returns:
            Dictionary with prediction results
        """
        try:
            # Preprocess image
            input_tensor = self.preprocess_image(image)
            
            # Forward pass
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = F.softmax(outputs, dim=1)
            
            # Extract results
            prob_real = probabilities[0][0].item()
            prob_ai = probabilities[0][1].item()
            
            is_ai = prob_ai > prob_real
            confidence = max(prob_ai, prob_real)
            
            return {
                'is_ai': is_ai,
                'confidence': confidence,
                'prob_real': prob_real,
                'prob_ai': prob_ai,
                'method': 'frequency_cnn'
            }
            
        except Exception as e:
            logger.error(f"Error in frequency CNN prediction: {str(e)}")
            return {
                'is_ai': False,
                'confidence': 0.5,
                'prob_real': 0.5,
                'prob_ai': 0.5,
                'method': 'frequency_cnn',
                'error': str(e)
            }
    
    def analyze_frequency_patterns(self, image: Union[Image.Image, np.ndarray, str]) -> Dict[str, Any]:
        """
        Analyze frequency patterns in detail.
        
        Args:
            image: Input image
            
        Returns:
            Detailed frequency analysis
        """
        try:
            # Convert to numpy if needed
            if isinstance(image, str):
                image = np.array(Image.open(image).convert('RGB'))
            elif isinstance(image, Image.Image):
                image = np.array(image)
            
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
            
            # FFT analysis
            fft = np.fft.fft2(gray)
            fft_magnitude = np.abs(fft)
            fft_shifted = np.fft.fftshift(fft_magnitude)
            
            # Calculate frequency domain statistics
            freq_stats = {
                'mean_magnitude': np.mean(fft_magnitude),
                'std_magnitude': np.std(fft_magnitude),
                'max_magnitude': np.max(fft_magnitude),
                'energy_concentration': self._calculate_energy_concentration(fft_shifted),
                'high_freq_energy': self._calculate_high_freq_energy(fft_shifted),
                'spectral_centroid': self._calculate_spectral_centroid(fft_shifted)
            }
            
            # DCT analysis
            dct = cv2.dct(np.float32(gray))
            dct_stats = {
                'dct_mean': np.mean(dct),
                'dct_std': np.std(dct),
                'dct_energy': np.sum(dct**2),
                'low_freq_dct_energy': np.sum(dct[:10, :10]**2),
                'high_freq_dct_energy': np.sum(dct[10:, 10:]**2)
            }
            
            return {
                'frequency_statistics': freq_stats,
                'dct_statistics': dct_stats,
                'image_shape': gray.shape
            }
            
        except Exception as e:
            logger.error(f"Error in frequency pattern analysis: {str(e)}")
            return {
                'frequency_statistics': {},
                'dct_statistics': {},
                'error': str(e)
            }
    
    def _calculate_energy_concentration(self, fft_shifted: np.ndarray) -> float:
        """Calculate energy concentration in center of frequency domain."""
        h, w = fft_shifted.shape
        center_h, center_w = h // 2, w // 2
        
        # Define center region (10% of image)
        region_size = min(h, w) // 10
        center_energy = np.sum(fft_shifted[
            center_h - region_size:center_h + region_size,
            center_w - region_size:center_w + region_size
        ]**2)
        
        total_energy = np.sum(fft_shifted**2)
        
        return center_energy / (total_energy + 1e-8)
    
    def _calculate_high_freq_energy(self, fft_shifted: np.ndarray) -> float:
        """Calculate energy in high frequency regions."""
        h, w = fft_shifted.shape
        center_h, center_w = h // 2, w // 2
        
        # Create high frequency mask (outer 30% of image)
        y, x = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((x - center_w)**2 + (y - center_h)**2)
        max_dist = min(h, w) // 2
        high_freq_mask = dist_from_center > (0.7 * max_dist)
        
        high_freq_energy = np.sum(fft_shifted[high_freq_mask]**2)
        total_energy = np.sum(fft_shifted**2)
        
        return high_freq_energy / (total_energy + 1e-8)
    
    def _calculate_spectral_centroid(self, fft_shifted: np.ndarray) -> float:
        """Calculate spectral centroid (center of mass of spectrum)."""
        h, w = fft_shifted.shape
        
        # Create frequency coordinate grids
        freq_y = np.arange(h) - h // 2
        freq_x = np.arange(w) - w // 2
        freq_y_grid, freq_x_grid = np.meshgrid(freq_y, freq_x, indexing='ij')
        
        # Calculate distance from DC component
        freq_dist = np.sqrt(freq_y_grid**2 + freq_x_grid**2)
        
        # Weight by magnitude
        weights = fft_shifted**2
        centroid = np.sum(freq_dist * weights) / (np.sum(weights) + 1e-8)
        
        return centroid
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            'model_type': 'frequency_cnn',
            'num_parameters': sum(p.numel() for p in self.model.parameters()),
            'input_channels': self.config['input_channels'],
            'num_classes': self.config['num_classes'],
            'device': str(self.device)
        }