"""RoBERTa-based classifier for AI-generated text detection."""

import torch
import torch.nn as nn
from transformers import (
    RobertaTokenizer, 
    RobertaForSequenceClassification,
    RobertaConfig
)
from typing import Dict, List, Any, Optional
import numpy as np

from ..utils import get_device, get_logger, get_model_config

logger = get_logger(__name__)


class RoBERTaClassifier:
    """
    RoBERTa-based classifier for detecting AI-generated text.
    
    This model fine-tunes RoBERTa on AI vs human text classification.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize RoBERTa classifier.
        
        Args:
            model_path: Path to pre-trained model. If None, uses base model.
        """
        self.config = get_model_config('text', 'roberta')
        self.device = get_device()
        
        # Initialize tokenizer
        self.tokenizer = RobertaTokenizer.from_pretrained(
            self.config['model_name']
        )
        
        # Initialize model
        if model_path:
            self.model = RobertaForSequenceClassification.from_pretrained(
                model_path
            )
            logger.info(f"Loaded model from {model_path}")
        else:
            self.model = RobertaForSequenceClassification.from_pretrained(
                self.config['model_name'],
                num_labels=self.config['num_labels']
            )
            logger.info(f"Initialized base model: {self.config['model_name']}")
        
        self.model.to(self.device)
        self.model.eval()
    
    def preprocess_text(self, text: str) -> Dict[str, torch.Tensor]:
        """
        Preprocess text for model input.
        
        Args:
            text: Input text to preprocess
            
        Returns:
            Dictionary with tokenized inputs
        """
        # Clean and truncate text if necessary
        text = text.strip()
        if len(text.split()) > 500:  # Rough word limit
            words = text.split()
            text = ' '.join(words[:500])
        
        # Tokenize
        inputs = self.tokenizer(
            text,
            max_length=self.config['max_length'],
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        return inputs
    
    def predict(self, text: str) -> Dict[str, Any]:
        """
        Predict if text is AI-generated.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary with prediction results
        """
        try:
            # Preprocess
            inputs = self.preprocess_text(text)
            
            # Forward pass
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=-1)
            
            # Extract results
            prob_human = probabilities[0][0].item()
            prob_ai = probabilities[0][1].item()
            
            is_ai = prob_ai > prob_human
            confidence = max(prob_ai, prob_human)
            
            return {
                'is_ai': is_ai,
                'confidence': confidence,
                'prob_human': prob_human,
                'prob_ai': prob_ai,
                'method': 'roberta'
            }
            
        except Exception as e:
            logger.error(f"Error in RoBERTa prediction: {str(e)}")
            return {
                'is_ai': False,
                'confidence': 0.5,
                'prob_human': 0.5,
                'prob_ai': 0.5,
                'method': 'roberta',
                'error': str(e)
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
        
        try:
            # Batch tokenization
            inputs = self.tokenizer(
                texts,
                max_length=self.config['max_length'],
                padding=True,
                truncation=True,
                return_tensors="pt"
            )
            
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Forward pass
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=-1)
            
            # Process results
            for i in range(len(texts)):
                prob_human = probabilities[i][0].item()
                prob_ai = probabilities[i][1].item()
                
                is_ai = prob_ai > prob_human
                confidence = max(prob_ai, prob_human)
                
                results.append({
                    'is_ai': is_ai,
                    'confidence': confidence,
                    'prob_human': prob_human,
                    'prob_ai': prob_ai,
                    'method': 'roberta'
                })
        
        except Exception as e:
            logger.error(f"Error in batch prediction: {str(e)}")
            # Return default results
            for _ in texts:
                results.append({
                    'is_ai': False,
                    'confidence': 0.5,
                    'prob_human': 0.5,
                    'prob_ai': 0.5,
                    'method': 'roberta',
                    'error': str(e)
                })
        
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            'model_name': self.config['model_name'],
            'num_parameters': sum(p.numel() for p in self.model.parameters()),
            'max_length': self.config['max_length'],
            'device': str(self.device),
            'vocab_size': self.tokenizer.vocab_size
        }