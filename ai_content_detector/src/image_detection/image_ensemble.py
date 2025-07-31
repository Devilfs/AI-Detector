"""Image ensemble model combining multiple detection methods."""

from typing import Dict, List, Any, Optional, Union
import numpy as np
from PIL import Image

from .efficientnet_classifier import EfficientNetClassifier
from .frequency_cnn import FrequencyCNN
from .prnu_detector import PRNUDetector
from ..utils import get_logger, get_model_config

logger = get_logger(__name__)


class ImageEnsemble:
    """
    Ensemble model combining EfficientNet, Frequency CNN, and PRNU detection
    for robust AI-generated image detection.
    """
    
    def __init__(self, 
                 efficientnet_model_path: Optional[str] = None,
                 frequency_cnn_model_path: Optional[str] = None):
        """
        Initialize image ensemble.
        
        Args:
            efficientnet_model_path: Path to fine-tuned EfficientNet model
            frequency_cnn_model_path: Path to trained frequency CNN model
        """
        self.config = get_model_config('image', 'ensemble_weights')
        
        # Initialize individual models
        logger.info("Initializing image ensemble components...")
        
        try:
            self.efficientnet = EfficientNetClassifier(efficientnet_model_path)
            logger.info("✓ EfficientNet classifier initialized")
        except Exception as e:
            logger.error(f"Failed to initialize EfficientNet: {e}")
            self.efficientnet = None
        
        try:
            self.frequency_cnn = FrequencyCNN(frequency_cnn_model_path)
            logger.info("✓ Frequency CNN initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Frequency CNN: {e}")
            self.frequency_cnn = None
        
        try:
            self.prnu_detector = PRNUDetector()
            logger.info("✓ PRNU detector initialized")
        except Exception as e:
            logger.error(f"Failed to initialize PRNU detector: {e}")
            self.prnu_detector = None
        
        # Ensemble weights
        self.weights = {
            'efficientnet': self.config.get('efficientnet', 0.6),
            'frequency_cnn': self.config.get('frequency_cnn', 0.4),
            'prnu': 0.2  # Not in config, using default
        }
        
        logger.info(f"Image ensemble initialized with weights: {self.weights}")
    
    def predict(self, image: Union[Image.Image, np.ndarray, str]) -> Dict[str, Any]:
        """
        Predict if image is AI-generated using ensemble of methods.
        
        Args:
            image: Input image to analyze
            
        Returns:
            Dictionary with ensemble prediction results
        """
        # Validate image input
        if isinstance(image, str):
            try:
                image = Image.open(image).convert('RGB')
            except Exception as e:
                return {
                    'is_ai': False,
                    'confidence': 0.5,
                    'method': 'ensemble',
                    'error': f'Failed to load image: {str(e)}'
                }
        elif isinstance(image, np.ndarray):
            try:
                image = Image.fromarray(image).convert('RGB')
            except Exception as e:
                return {
                    'is_ai': False,
                    'confidence': 0.5,
                    'method': 'ensemble',
                    'error': f'Failed to convert image: {str(e)}'
                }
        
        # Get predictions from each model
        predictions = {}
        valid_models = []
        
        # EfficientNet prediction
        if self.efficientnet:
            try:
                efficientnet_result = self.efficientnet.predict(image)
                predictions['efficientnet'] = efficientnet_result
                valid_models.append('efficientnet')
            except Exception as e:
                logger.error(f"EfficientNet prediction failed: {e}")
        
        # Frequency CNN prediction
        if self.frequency_cnn:
            try:
                frequency_result = self.frequency_cnn.predict(image)
                predictions['frequency_cnn'] = frequency_result
                valid_models.append('frequency_cnn')
            except Exception as e:
                logger.error(f"Frequency CNN prediction failed: {e}")
        
        # PRNU prediction
        if self.prnu_detector:
            try:
                prnu_result = self.prnu_detector.predict(image)
                predictions['prnu'] = prnu_result
                valid_models.append('prnu')
            except Exception as e:
                logger.error(f"PRNU prediction failed: {e}")
        
        if not valid_models:
            return {
                'is_ai': False,
                'confidence': 0.5,
                'method': 'ensemble',
                'error': 'No valid models available'
            }
        
        # Calculate ensemble prediction
        ensemble_result = self._calculate_ensemble(predictions, valid_models)
        
        # Add individual predictions for transparency
        ensemble_result['individual_predictions'] = predictions
        ensemble_result['method'] = 'ensemble'
        ensemble_result['valid_models'] = valid_models
        
        return ensemble_result
    
    def _calculate_ensemble(self, predictions: Dict[str, Any], valid_models: List[str]) -> Dict[str, Any]:
        """
        Calculate ensemble prediction from individual model results.
        
        Args:
            predictions: Dictionary of individual model predictions
            valid_models: List of models that produced valid predictions
            
        Returns:
            Ensemble prediction dictionary
        """
        # Normalize weights for available models
        total_weight = sum(self.weights[model] for model in valid_models)
        normalized_weights = {model: self.weights[model] / total_weight for model in valid_models}
        
        # Calculate weighted average of AI probabilities
        weighted_ai_prob = 0
        confidence_scores = []
        
        for model in valid_models:
            pred = predictions[model]
            
            # Extract AI probability
            if 'prob_ai' in pred:
                ai_prob = pred['prob_ai']
            elif 'ai_score' in pred:
                ai_prob = pred['ai_score']
            else:
                # Convert binary prediction to probability
                ai_prob = pred['confidence'] if pred['is_ai'] else (1 - pred['confidence'])
            
            weighted_ai_prob += ai_prob * normalized_weights[model]
            confidence_scores.append(pred['confidence'])
        
        # Final ensemble decision
        is_ai = weighted_ai_prob > 0.5
        
        # Calculate ensemble confidence
        # Use combination of weighted probability and individual confidences
        prob_confidence = abs(weighted_ai_prob - 0.5) * 2  # Distance from 0.5, scaled to [0,1]
        avg_individual_confidence = np.mean(confidence_scores)
        
        # Weighted combination of probability confidence and individual confidences
        ensemble_confidence = 0.6 * prob_confidence + 0.4 * avg_individual_confidence
        ensemble_confidence = min(0.99, max(0.51, ensemble_confidence))
        
        return {
            'is_ai': is_ai,
            'confidence': ensemble_confidence,
            'ai_probability': weighted_ai_prob,
            'real_probability': 1 - weighted_ai_prob,
            'ensemble_weights': normalized_weights
        }
    
    def analyze_detailed(self, image: Union[Image.Image, np.ndarray, str]) -> Dict[str, Any]:
        """
        Perform detailed analysis with explanations from all models.
        
        Args:
            image: Input image to analyze
            
        Returns:
            Detailed analysis results
        """
        basic_result = self.predict(image)
        
        # Add detailed analysis from individual models
        detailed_analysis = {
            'basic_prediction': basic_result,
            'detailed_features': {}
        }
        
        # Convert image for analysis
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert('RGB')
        
        # Get detailed frequency analysis
        if self.frequency_cnn:
            try:
                freq_analysis = self.frequency_cnn.analyze_frequency_patterns(image)
                detailed_analysis['detailed_features']['frequency'] = freq_analysis
            except Exception as e:
                logger.error(f"Detailed frequency analysis failed: {e}")
        
        # Get PRNU features (already detailed in the prediction)
        if 'prnu' in basic_result.get('individual_predictions', {}):
            prnu_pred = basic_result['individual_predictions']['prnu']
            if 'features' in prnu_pred:
                detailed_analysis['detailed_features']['prnu'] = prnu_pred['features']
        
        # Get EfficientNet features
        if self.efficientnet:
            try:
                features = self.efficientnet.get_features(image)
                detailed_analysis['detailed_features']['efficientnet_features'] = {
                    'feature_vector_shape': features.shape,
                    'feature_vector_mean': np.mean(features),
                    'feature_vector_std': np.std(features)
                }
            except Exception as e:
                logger.error(f"EfficientNet feature extraction failed: {e}")
        
        # Add image statistics
        detailed_analysis['image_statistics'] = self._get_image_statistics(image)
        
        return detailed_analysis
    
    def _get_image_statistics(self, image: Image.Image) -> Dict[str, Any]:
        """
        Get basic image statistics.
        
        Args:
            image: Input image
            
        Returns:
            Dictionary with image statistics
        """
        img_array = np.array(image)
        
        stats = {
            'width': image.width,
            'height': image.height,
            'channels': len(img_array.shape),
            'file_format': image.format,
            'mode': image.mode
        }
        
        if len(img_array.shape) == 3:
            stats.update({
                'mean_rgb': np.mean(img_array, axis=(0, 1)).tolist(),
                'std_rgb': np.std(img_array, axis=(0, 1)).tolist(),
                'brightness': np.mean(img_array),
                'contrast': np.std(img_array)
            })
        else:
            stats.update({
                'mean_intensity': np.mean(img_array),
                'std_intensity': np.std(img_array),
                'brightness': np.mean(img_array),
                'contrast': np.std(img_array)
            })
        
        return stats
    
    def predict_batch(self, images: List[Union[Image.Image, np.ndarray, str]]) -> List[Dict[str, Any]]:
        """
        Predict for multiple images.
        
        Args:
            images: List of images to analyze
            
        Returns:
            List of prediction results
        """
        results = []
        
        # Use EfficientNet batch prediction if available
        efficientnet_batch_results = []
        if self.efficientnet:
            try:
                efficientnet_batch_results = self.efficientnet.predict_batch(images)
            except Exception as e:
                logger.error(f"EfficientNet batch prediction failed: {e}")
        
        # Process each image
        for i, image in enumerate(images):
            # Get individual predictions
            predictions = {}
            valid_models = []
            
            # Use batch EfficientNet result if available
            if efficientnet_batch_results and i < len(efficientnet_batch_results):
                predictions['efficientnet'] = efficientnet_batch_results[i]
                valid_models.append('efficientnet')
            elif self.efficientnet:
                try:
                    predictions['efficientnet'] = self.efficientnet.predict(image)
                    valid_models.append('efficientnet')
                except Exception as e:
                    logger.error(f"EfficientNet prediction failed for image {i}: {e}")
            
            # Other models (individual predictions)
            if self.frequency_cnn:
                try:
                    predictions['frequency_cnn'] = self.frequency_cnn.predict(image)
                    valid_models.append('frequency_cnn')
                except Exception as e:
                    logger.error(f"Frequency CNN prediction failed for image {i}: {e}")
            
            if self.prnu_detector:
                try:
                    predictions['prnu'] = self.prnu_detector.predict(image)
                    valid_models.append('prnu')
                except Exception as e:
                    logger.error(f"PRNU prediction failed for image {i}: {e}")
            
            # Calculate ensemble
            if valid_models:
                ensemble_result = self._calculate_ensemble(predictions, valid_models)
                ensemble_result['individual_predictions'] = predictions
                ensemble_result['method'] = 'ensemble'
                ensemble_result['valid_models'] = valid_models
            else:
                ensemble_result = {
                    'is_ai': False,
                    'confidence': 0.5,
                    'method': 'ensemble',
                    'error': 'No valid models available'
                }
            
            results.append(ensemble_result)
        
        return results
    
    def get_attention_map(self, image: Union[Image.Image, np.ndarray, str]) -> np.ndarray:
        """
        Generate attention map showing areas important for AI detection.
        
        Args:
            image: Input image
            
        Returns:
            Attention map as numpy array
        """
        if self.efficientnet:
            try:
                return self.efficientnet.get_attention_map(image)
            except Exception as e:
                logger.error(f"Failed to generate attention map: {e}")
        
        # Return default map if EfficientNet is not available
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert('RGB')
        
        return np.ones((224, 224)) * 0.5
    
    def compare_images(self, image1: Union[Image.Image, np.ndarray, str], 
                      image2: Union[Image.Image, np.ndarray, str]) -> Dict[str, Any]:
        """
        Compare two images for similarity in AI generation patterns.
        
        Args:
            image1: First image
            image2: Second image
            
        Returns:
            Comparison results
        """
        results = {
            'prnu_similarity': 0,
            'feature_similarity': 0,
            'overall_similarity': 0
        }
        
        # PRNU pattern comparison
        if self.prnu_detector:
            try:
                prnu_sim = self.prnu_detector.compare_prnu_patterns(image1, image2)
                results['prnu_similarity'] = prnu_sim
            except Exception as e:
                logger.error(f"PRNU comparison failed: {e}")
        
        # Feature-based comparison
        if self.efficientnet:
            try:
                features1 = self.efficientnet.get_features(image1)
                features2 = self.efficientnet.get_features(image2)
                
                # Calculate cosine similarity
                from sklearn.metrics.pairwise import cosine_similarity
                feature_sim = cosine_similarity(features1, features2)[0, 0]
                results['feature_similarity'] = feature_sim
            except Exception as e:
                logger.error(f"Feature comparison failed: {e}")
        
        # Overall similarity (weighted average)
        valid_similarities = [v for v in [results['prnu_similarity'], results['feature_similarity']] if v != 0]
        if valid_similarities:
            results['overall_similarity'] = np.mean(valid_similarities)
        
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the ensemble and its components."""
        info = {
            'ensemble_weights': self.weights,
            'available_models': []
        }
        
        if self.efficientnet:
            info['available_models'].append('efficientnet')
            info['efficientnet_info'] = self.efficientnet.get_model_info()
        
        if self.frequency_cnn:
            info['available_models'].append('frequency_cnn')
            info['frequency_cnn_info'] = self.frequency_cnn.get_model_info()
        
        if self.prnu_detector:
            info['available_models'].append('prnu')
            info['prnu_info'] = self.prnu_detector.get_detector_info()
        
        return info