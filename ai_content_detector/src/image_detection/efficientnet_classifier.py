"""EfficientNet-based classifier for AI-generated image detection."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
from typing import Dict, List, Any, Optional, Union
import timm

from ..utils import get_device, get_logger, get_model_config

logger = get_logger(__name__)


class EfficientNetClassifier:
    """
    EfficientNet-based classifier for detecting AI-generated images.
    
    Uses pre-trained EfficientNet with custom classification head
    for distinguishing between real and AI-generated images.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize EfficientNet classifier.
        
        Args:
            model_path: Path to pre-trained model. If None, uses base model.
        """
        self.config = get_model_config('image', 'efficientnet')
        self.device = get_device()
        
        # Initialize model
        self.model = self._create_model()
        
        if model_path:
            self._load_model(model_path)
        
        self.model.to(self.device)
        self.model.eval()
        
        # Initialize transforms
        self.transform = self._get_transforms()
        
        logger.info(f"Initialized EfficientNet classifier: {self.config['model_name']}")
    
    def _create_model(self) -> nn.Module:
        """Create EfficientNet model with custom head."""
        # Load pre-trained EfficientNet
        model = timm.create_model(
            self.config['model_name'],
            pretrained=True,
            num_classes=self.config['num_classes']
        )
        
        return model
    
    def _load_model(self, model_path: str):
        """Load pre-trained model weights."""
        try:
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            logger.info(f"Loaded model weights from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load model weights: {e}")
            raise
    
    def _get_transforms(self) -> transforms.Compose:
        """Get image preprocessing transforms."""
        input_size = self.config['input_size']
        
        return transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def preprocess_image(self, image: Union[Image.Image, np.ndarray, str]) -> torch.Tensor:
        """
        Preprocess image for model input.
        
        Args:
            image: Input image (PIL Image, numpy array, or file path)
            
        Returns:
            Preprocessed tensor
        """
        # Convert to PIL Image if necessary
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert('RGB')
        elif not isinstance(image, Image.Image):
            raise ValueError("Unsupported image type")
        
        # Apply transforms
        tensor = self.transform(image)
        
        # Add batch dimension and move to device
        tensor = tensor.unsqueeze(0).to(self.device)
        
        return tensor
    
    def predict(self, image: Union[Image.Image, np.ndarray, str]) -> Dict[str, Any]:
        """
        Predict if image is AI-generated.
        
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
                'method': 'efficientnet'
            }
            
        except Exception as e:
            logger.error(f"Error in EfficientNet prediction: {str(e)}")
            return {
                'is_ai': False,
                'confidence': 0.5,
                'prob_real': 0.5,
                'prob_ai': 0.5,
                'method': 'efficientnet',
                'error': str(e)
            }
    
    def predict_batch(self, images: List[Union[Image.Image, np.ndarray, str]]) -> List[Dict[str, Any]]:
        """
        Predict for multiple images.
        
        Args:
            images: List of images to analyze
            
        Returns:
            List of prediction results
        """
        results = []
        
        try:
            # Preprocess all images
            batch_tensors = []
            valid_indices = []
            
            for i, image in enumerate(images):
                try:
                    tensor = self.preprocess_image(image)
                    batch_tensors.append(tensor.squeeze(0))
                    valid_indices.append(i)
                except Exception as e:
                    logger.error(f"Failed to preprocess image {i}: {e}")
                    results.append({
                        'is_ai': False,
                        'confidence': 0.5,
                        'prob_real': 0.5,
                        'prob_ai': 0.5,
                        'method': 'efficientnet',
                        'error': str(e)
                    })
            
            if batch_tensors:
                # Stack tensors and run batch inference
                batch_tensor = torch.stack(batch_tensors).to(self.device)
                
                with torch.no_grad():
                    outputs = self.model(batch_tensor)
                    probabilities = F.softmax(outputs, dim=1)
                
                # Process results
                batch_results = []
                for i in range(len(batch_tensors)):
                    prob_real = probabilities[i][0].item()
                    prob_ai = probabilities[i][1].item()
                    
                    is_ai = prob_ai > prob_real
                    confidence = max(prob_ai, prob_real)
                    
                    batch_results.append({
                        'is_ai': is_ai,
                        'confidence': confidence,
                        'prob_real': prob_real,
                        'prob_ai': prob_ai,
                        'method': 'efficientnet'
                    })
                
                # Insert batch results at correct positions
                batch_idx = 0
                final_results = []
                for i in range(len(images)):
                    if i in valid_indices:
                        final_results.append(batch_results[batch_idx])
                        batch_idx += 1
                    else:
                        # Use error result already in results list
                        final_results.append(results[i - batch_idx])
                
                results = final_results
        
        except Exception as e:
            logger.error(f"Error in batch prediction: {str(e)}")
            # Return default results for all images
            results = []
            for _ in images:
                results.append({
                    'is_ai': False,
                    'confidence': 0.5,
                    'prob_real': 0.5,
                    'prob_ai': 0.5,
                    'method': 'efficientnet',
                    'error': str(e)
                })
        
        return results
    
    def get_features(self, image: Union[Image.Image, np.ndarray, str]) -> np.ndarray:
        """
        Extract features from the penultimate layer.
        
        Args:
            image: Input image
            
        Returns:
            Feature vector
        """
        try:
            # Preprocess image
            input_tensor = self.preprocess_image(image)
            
            # Forward pass through feature extractor
            with torch.no_grad():
                # Remove classifier head and get features
                features = self.model.forward_features(input_tensor)
                features = self.model.global_pool(features)
                features = features.flatten(1)
            
            return features.cpu().numpy()
            
        except Exception as e:
            logger.error(f"Error extracting features: {str(e)}")
            return np.zeros((1, 1280))  # Default feature size for EfficientNet-B0
    
    def get_attention_map(self, image: Union[Image.Image, np.ndarray, str]) -> np.ndarray:
        """
        Generate attention map using Grad-CAM.
        
        Args:
            image: Input image
            
        Returns:
            Attention map as numpy array
        """
        try:
            # Preprocess image
            input_tensor = self.preprocess_image(image)
            input_tensor.requires_grad_(True)
            
            # Forward pass
            outputs = self.model(input_tensor)
            
            # Get prediction for AI class
            ai_score = outputs[0][1]
            
            # Backward pass
            self.model.zero_grad()
            ai_score.backward()
            
            # Get gradients and activations from last conv layer
            # This is a simplified version - full Grad-CAM would require hooks
            gradients = input_tensor.grad.data
            
            # Create attention map
            attention = torch.mean(torch.abs(gradients), dim=1, keepdim=True)
            attention = F.interpolate(
                attention, 
                size=(224, 224), 
                mode='bilinear', 
                align_corners=False
            )
            
            attention = attention.squeeze().cpu().numpy()
            
            # Normalize to [0, 1]
            attention = (attention - attention.min()) / (attention.max() - attention.min() + 1e-8)
            
            return attention
            
        except Exception as e:
            logger.error(f"Error generating attention map: {str(e)}")
            return np.zeros((224, 224))
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            'model_name': self.config['model_name'],
            'num_parameters': sum(p.numel() for p in self.model.parameters()),
            'input_size': self.config['input_size'],
            'num_classes': self.config['num_classes'],
            'device': str(self.device)
        }