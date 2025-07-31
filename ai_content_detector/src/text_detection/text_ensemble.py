"""Text ensemble model combining multiple detection methods."""

from typing import Dict, List, Any, Optional
import numpy as np

from .roberta_classifier import RoBERTaClassifier
from .perplexity_analyzer import PerplexityAnalyzer
from .burstiness_analyzer import BurstinessAnalyzer
from ..utils import get_logger, get_model_config

logger = get_logger(__name__)


class TextEnsemble:
    """
    Ensemble model combining RoBERTa, perplexity, and burstiness analysis
    for robust AI-generated text detection.
    """
    
    def __init__(self, 
                 roberta_model_path: Optional[str] = None,
                 perplexity_model_path: Optional[str] = None):
        """
        Initialize text ensemble.
        
        Args:
            roberta_model_path: Path to fine-tuned RoBERTa model
            perplexity_model_path: Path to custom perplexity model
        """
        self.config = get_model_config('text', 'ensemble_weights')
        
        # Initialize individual models
        logger.info("Initializing text ensemble components...")
        
        try:
            self.roberta = RoBERTaClassifier(roberta_model_path)
            logger.info("✓ RoBERTa classifier initialized")
        except Exception as e:
            logger.error(f"Failed to initialize RoBERTa: {e}")
            self.roberta = None
        
        try:
            self.perplexity_analyzer = PerplexityAnalyzer(perplexity_model_path)
            logger.info("✓ Perplexity analyzer initialized")
        except Exception as e:
            logger.error(f"Failed to initialize perplexity analyzer: {e}")
            self.perplexity_analyzer = None
        
        try:
            self.burstiness_analyzer = BurstinessAnalyzer()
            logger.info("✓ Burstiness analyzer initialized")
        except Exception as e:
            logger.error(f"Failed to initialize burstiness analyzer: {e}")
            self.burstiness_analyzer = None
        
        # Ensemble weights
        self.weights = {
            'roberta': self.config.get('roberta', 0.5),
            'perplexity': self.config.get('perplexity', 0.3),
            'burstiness': 0.2  # Not in config, using default
        }
        
        logger.info(f"Text ensemble initialized with weights: {self.weights}")
    
    def predict(self, text: str) -> Dict[str, Any]:
        """
        Predict if text is AI-generated using ensemble of methods.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary with ensemble prediction results
        """
        if not text or len(text.strip()) < 50:
            return {
                'is_ai': False,
                'confidence': 0.5,
                'method': 'ensemble',
                'error': 'Text too short for reliable analysis'
            }
        
        # Get predictions from each model
        predictions = {}
        valid_models = []
        
        # RoBERTa prediction
        if self.roberta:
            try:
                roberta_result = self.roberta.predict(text)
                predictions['roberta'] = roberta_result
                valid_models.append('roberta')
            except Exception as e:
                logger.error(f"RoBERTa prediction failed: {e}")
        
        # Perplexity prediction
        if self.perplexity_analyzer:
            try:
                perplexity_result = self.perplexity_analyzer.predict(text)
                predictions['perplexity'] = perplexity_result
                valid_models.append('perplexity')
            except Exception as e:
                logger.error(f"Perplexity prediction failed: {e}")
        
        # Burstiness prediction
        if self.burstiness_analyzer:
            try:
                burstiness_result = self.burstiness_analyzer.predict(text)
                predictions['burstiness'] = burstiness_result
                valid_models.append('burstiness')
            except Exception as e:
                logger.error(f"Burstiness prediction failed: {e}")
        
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
            'human_probability': 1 - weighted_ai_prob,
            'ensemble_weights': normalized_weights
        }
    
    def analyze_detailed(self, text: str) -> Dict[str, Any]:
        """
        Perform detailed analysis with explanations from all models.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Detailed analysis results
        """
        basic_result = self.predict(text)
        
        # Add detailed analysis from individual models
        detailed_analysis = {
            'basic_prediction': basic_result,
            'detailed_features': {}
        }
        
        # Get detailed perplexity statistics
        if self.perplexity_analyzer:
            try:
                perplexity_stats = self.perplexity_analyzer.get_statistics(text)
                detailed_analysis['detailed_features']['perplexity'] = perplexity_stats
            except Exception as e:
                logger.error(f"Detailed perplexity analysis failed: {e}")
        
        # Burstiness features are already detailed in the prediction
        if 'burstiness' in basic_result.get('individual_predictions', {}):
            burstiness_pred = basic_result['individual_predictions']['burstiness']
            if 'features' in burstiness_pred:
                detailed_analysis['detailed_features']['burstiness'] = burstiness_pred['features']
        
        # Add text statistics
        detailed_analysis['text_statistics'] = self._get_text_statistics(text)
        
        return detailed_analysis
    
    def _get_text_statistics(self, text: str) -> Dict[str, Any]:
        """
        Get basic text statistics.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with text statistics
        """
        words = text.split()
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        
        return {
            'character_count': len(text),
            'word_count': len(words),
            'sentence_count': len(sentences),
            'avg_words_per_sentence': len(words) / len(sentences) if sentences else 0,
            'avg_chars_per_word': len(text.replace(' ', '')) / len(words) if words else 0
        }
    
    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Predict for multiple texts.
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            List of prediction results
        """
        results = []
        
        # Use RoBERTa batch prediction if available
        roberta_batch_results = []
        if self.roberta:
            try:
                roberta_batch_results = self.roberta.predict_batch(texts)
            except Exception as e:
                logger.error(f"RoBERTa batch prediction failed: {e}")
        
        # Process each text
        for i, text in enumerate(texts):
            # Get individual predictions
            predictions = {}
            valid_models = []
            
            # Use batch RoBERTa result if available
            if roberta_batch_results and i < len(roberta_batch_results):
                predictions['roberta'] = roberta_batch_results[i]
                valid_models.append('roberta')
            elif self.roberta:
                try:
                    predictions['roberta'] = self.roberta.predict(text)
                    valid_models.append('roberta')
                except Exception as e:
                    logger.error(f"RoBERTa prediction failed for text {i}: {e}")
            
            # Other models (individual predictions)
            if self.perplexity_analyzer:
                try:
                    predictions['perplexity'] = self.perplexity_analyzer.predict(text)
                    valid_models.append('perplexity')
                except Exception as e:
                    logger.error(f"Perplexity prediction failed for text {i}: {e}")
            
            if self.burstiness_analyzer:
                try:
                    predictions['burstiness'] = self.burstiness_analyzer.predict(text)
                    valid_models.append('burstiness')
                except Exception as e:
                    logger.error(f"Burstiness prediction failed for text {i}: {e}")
            
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
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the ensemble and its components."""
        info = {
            'ensemble_weights': self.weights,
            'available_models': []
        }
        
        if self.roberta:
            info['available_models'].append('roberta')
            info['roberta_info'] = self.roberta.get_model_info()
        
        if self.perplexity_analyzer:
            info['available_models'].append('perplexity')
        
        if self.burstiness_analyzer:
            info['available_models'].append('burstiness')
        
        return info