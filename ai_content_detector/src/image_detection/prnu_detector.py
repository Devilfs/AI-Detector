"""PRNU-based detector for AI-generated image detection."""

import numpy as np
from PIL import Image
import cv2
from typing import Dict, List, Any, Optional, Union
from scipy import ndimage, signal
from sklearn.metrics.pairwise import cosine_similarity

from ..utils import get_logger

logger = get_logger(__name__)


class PRNUDetector:
    """
    PRNU (Photo Response Non-Uniformity) detector for AI-generated images.
    
    Real camera sensors have unique noise patterns (PRNU) that can be
    extracted and used for authentication. AI-generated images typically
    lack these authentic sensor patterns.
    """
    
    def __init__(self):
        """Initialize PRNU detector."""
        self.noise_variance_threshold = 0.1
        self.correlation_threshold = 0.02
        self.block_size = 64
        
        logger.info("Initialized PRNU detector")
    
    def extract_noise_residual(self, image: Union[Image.Image, np.ndarray]) -> np.ndarray:
        """
        Extract noise residual from image using denoising filter.
        
        Args:
            image: Input image
            
        Returns:
            Noise residual pattern
        """
        # Convert to numpy array if PIL Image
        if isinstance(image, Image.Image):
            img_array = np.array(image)
        else:
            img_array = image.copy()
        
        # Convert to grayscale if color image
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        
        # Apply denoising filter (Gaussian filter)
        denoised = cv2.GaussianBlur(gray.astype(np.float32), (5, 5), 1.0)
        
        # Calculate noise residual
        noise_residual = gray.astype(np.float32) - denoised
        
        return noise_residual
    
    def enhance_prnu_pattern(self, noise_residual: np.ndarray) -> np.ndarray:
        """
        Enhance PRNU pattern in noise residual.
        
        Args:
            noise_residual: Noise residual from image
            
        Returns:
            Enhanced PRNU pattern
        """
        # Apply Wiener filter to enhance PRNU
        # Estimate noise variance
        noise_var = np.var(noise_residual)
        
        # Apply 2D Wiener filtering
        # Using simple implementation with FFT
        fft_noise = np.fft.fft2(noise_residual)
        
        # Estimate signal power spectrum (simplified)
        signal_power = np.abs(fft_noise)**2
        noise_power = noise_var * np.ones_like(signal_power)
        
        # Wiener filter
        wiener_filter = signal_power / (signal_power + noise_power + 1e-8)
        
        # Apply filter
        enhanced_fft = fft_noise * wiener_filter
        enhanced_prnu = np.real(np.fft.ifft2(enhanced_fft))
        
        return enhanced_prnu
    
    def calculate_prnu_features(self, image: Union[Image.Image, np.ndarray]) -> Dict[str, float]:
        """
        Calculate PRNU-based features for classification.
        
        Args:
            image: Input image
            
        Returns:
            Dictionary of PRNU features
        """
        try:
            # Extract noise residual
            noise_residual = self.extract_noise_residual(image)
            
            # Enhance PRNU pattern
            prnu_pattern = self.enhance_prnu_pattern(noise_residual)
            
            # Calculate various PRNU features
            features = {}
            
            # 1. Noise variance
            features['noise_variance'] = np.var(noise_residual)
            
            # 2. PRNU strength (correlation with itself after rotation)
            rotated_prnu = np.rot90(prnu_pattern)
            correlation = np.corrcoef(prnu_pattern.flatten(), rotated_prnu.flatten())[0, 1]
            features['prnu_self_correlation'] = correlation if not np.isnan(correlation) else 0
            
            # 3. Spatial correlation structure
            features['spatial_correlation'] = self._calculate_spatial_correlation(prnu_pattern)
            
            # 4. Frequency domain characteristics
            fft_prnu = np.fft.fft2(prnu_pattern)
            features['freq_energy_ratio'] = self._calculate_freq_energy_ratio(fft_prnu)
            
            # 5. Block-wise variance analysis
            features['block_variance_consistency'] = self._calculate_block_variance(prnu_pattern)
            
            # 6. Pattern regularity
            features['pattern_regularity'] = self._calculate_pattern_regularity(prnu_pattern)
            
            return features
            
        except Exception as e:
            logger.error(f"Error calculating PRNU features: {str(e)}")
            return {
                'noise_variance': 0,
                'prnu_self_correlation': 0,
                'spatial_correlation': 0,
                'freq_energy_ratio': 0,
                'block_variance_consistency': 0,
                'pattern_regularity': 0
            }
    
    def _calculate_spatial_correlation(self, prnu_pattern: np.ndarray) -> float:
        """Calculate spatial correlation in PRNU pattern."""
        h, w = prnu_pattern.shape
        
        # Calculate correlation with shifted versions
        correlations = []
        for shift in [1, 2, 4, 8]:
            if shift < min(h, w) // 4:
                shifted_h = prnu_pattern[shift:, :]
                original_h = prnu_pattern[:-shift, :]
                corr_h = np.corrcoef(shifted_h.flatten(), original_h.flatten())[0, 1]
                
                shifted_w = prnu_pattern[:, shift:]
                original_w = prnu_pattern[:, :-shift]
                corr_w = np.corrcoef(shifted_w.flatten(), original_w.flatten())[0, 1]
                
                correlations.extend([corr_h, corr_w])
        
        # Return mean correlation (filter out NaN values)
        valid_correlations = [c for c in correlations if not np.isnan(c)]
        return np.mean(valid_correlations) if valid_correlations else 0
    
    def _calculate_freq_energy_ratio(self, fft_prnu: np.ndarray) -> float:
        """Calculate energy ratio between high and low frequencies."""
        magnitude = np.abs(fft_prnu)
        h, w = magnitude.shape
        
        # Define frequency regions
        center_h, center_w = h // 2, w // 2
        
        # Low frequency region (center 25%)
        low_freq_size = min(h, w) // 8
        low_freq_energy = np.sum(magnitude[
            center_h - low_freq_size:center_h + low_freq_size,
            center_w - low_freq_size:center_w + low_freq_size
        ]**2)
        
        # High frequency region (outer 25%)
        y, x = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((x - center_w)**2 + (y - center_h)**2)
        max_dist = min(h, w) // 2
        high_freq_mask = dist_from_center > (0.75 * max_dist)
        high_freq_energy = np.sum(magnitude[high_freq_mask]**2)
        
        # Calculate ratio
        return high_freq_energy / (low_freq_energy + 1e-8)
    
    def _calculate_block_variance(self, prnu_pattern: np.ndarray) -> float:
        """Calculate consistency of variance across image blocks."""
        h, w = prnu_pattern.shape
        block_variances = []
        
        # Divide image into blocks
        for i in range(0, h - self.block_size, self.block_size):
            for j in range(0, w - self.block_size, self.block_size):
                block = prnu_pattern[i:i + self.block_size, j:j + self.block_size]
                block_variances.append(np.var(block))
        
        # Calculate coefficient of variation of block variances
        if len(block_variances) > 1:
            mean_var = np.mean(block_variances)
            std_var = np.std(block_variances)
            return std_var / (mean_var + 1e-8)
        else:
            return 0
    
    def _calculate_pattern_regularity(self, prnu_pattern: np.ndarray) -> float:
        """Calculate regularity/randomness of PRNU pattern."""
        # Use autocorrelation to measure pattern regularity
        autocorr = signal.correlate2d(prnu_pattern, prnu_pattern, mode='same')
        
        # Normalize autocorrelation
        autocorr = autocorr / np.max(np.abs(autocorr))
        
        # Calculate peak-to-sidelobe ratio
        h, w = autocorr.shape
        center = autocorr[h//2, w//2]
        
        # Exclude center peak
        autocorr_no_center = autocorr.copy()
        autocorr_no_center[h//2, w//2] = 0
        
        max_sidelobe = np.max(np.abs(autocorr_no_center))
        
        return center / (max_sidelobe + 1e-8)
    
    def predict(self, image: Union[Image.Image, np.ndarray, str]) -> Dict[str, Any]:
        """
        Predict if image is AI-generated based on PRNU analysis.
        
        Args:
            image: Input image
            
        Returns:
            Dictionary with prediction results
        """
        try:
            # Load image if path provided
            if isinstance(image, str):
                image = Image.open(image).convert('RGB')
            
            # Calculate PRNU features
            features = self.calculate_prnu_features(image)
            
            # Simple rule-based classification
            # These thresholds should be learned from data
            ai_indicators = 0
            total_indicators = 0
            
            # Check noise variance (AI images often have lower sensor noise)
            if features['noise_variance'] < self.noise_variance_threshold:
                ai_indicators += 1
            total_indicators += 1
            
            # Check PRNU correlation (AI images lack authentic sensor patterns)
            if abs(features['prnu_self_correlation']) < self.correlation_threshold:
                ai_indicators += 1
            total_indicators += 1
            
            # Check spatial correlation (AI images may have more regular patterns)
            if features['spatial_correlation'] > 0.1:
                ai_indicators += 1
            total_indicators += 1
            
            # Check frequency characteristics
            if features['freq_energy_ratio'] < 0.5:
                ai_indicators += 1
            total_indicators += 1
            
            # Check block variance consistency
            if features['block_variance_consistency'] < 0.5:
                ai_indicators += 1
            total_indicators += 1
            
            # Calculate final prediction
            ai_score = ai_indicators / total_indicators
            is_ai = ai_score > 0.5
            confidence = max(ai_score, 1 - ai_score)
            
            return {
                'is_ai': is_ai,
                'confidence': confidence,
                'ai_score': ai_score,
                'method': 'prnu',
                'features': features
            }
            
        except Exception as e:
            logger.error(f"Error in PRNU prediction: {str(e)}")
            return {
                'is_ai': False,
                'confidence': 0.5,
                'ai_score': 0.5,
                'method': 'prnu',
                'error': str(e)
            }
    
    def compare_prnu_patterns(self, image1: Union[Image.Image, np.ndarray], 
                             image2: Union[Image.Image, np.ndarray]) -> float:
        """
        Compare PRNU patterns between two images.
        
        Args:
            image1: First image
            image2: Second image
            
        Returns:
            Similarity score between PRNU patterns
        """
        try:
            # Extract PRNU patterns
            prnu1 = self.enhance_prnu_pattern(self.extract_noise_residual(image1))
            prnu2 = self.enhance_prnu_pattern(self.extract_noise_residual(image2))
            
            # Resize to same dimensions if needed
            if prnu1.shape != prnu2.shape:
                min_h = min(prnu1.shape[0], prnu2.shape[0])
                min_w = min(prnu1.shape[1], prnu2.shape[1])
                prnu1 = cv2.resize(prnu1, (min_w, min_h))
                prnu2 = cv2.resize(prnu2, (min_w, min_h))
            
            # Calculate normalized cross-correlation
            correlation = np.corrcoef(prnu1.flatten(), prnu2.flatten())[0, 1]
            
            return correlation if not np.isnan(correlation) else 0
            
        except Exception as e:
            logger.error(f"Error comparing PRNU patterns: {str(e)}")
            return 0
    
    def get_detector_info(self) -> Dict[str, Any]:
        """Get detector information."""
        return {
            'detector_type': 'prnu',
            'noise_variance_threshold': self.noise_variance_threshold,
            'correlation_threshold': self.correlation_threshold,
            'block_size': self.block_size
        }