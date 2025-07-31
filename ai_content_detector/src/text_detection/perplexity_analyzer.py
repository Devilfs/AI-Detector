"""Perplexity-based analyzer for AI-generated text detection."""

import torch
import torch.nn.functional as F
from transformers import GPT2TokenizerFast, GPT2LMHeadModel
from typing import Dict, List, Any, Optional
import numpy as np
import math

from ..utils import get_device, get_logger, get_model_config

logger = get_logger(__name__)


class PerplexityAnalyzer:
    """
    Perplexity-based analyzer for detecting AI-generated text.
    
    Uses GPT-2 to calculate perplexity scores. AI-generated text typically
    has lower perplexity when evaluated by similar models.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize perplexity analyzer.
        
        Args:
            model_path: Path to custom model. If None, uses GPT-2.
        """
        self.config = get_model_config('text', 'gpt2_perplexity')
        self.device = get_device()
        
        # Initialize tokenizer and model
        model_name = model_path or self.config['model_name']
        
        self.tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        
        # Set padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"Initialized perplexity analyzer with {model_name}")
    
    def calculate_perplexity(self, text: str) -> float:
        """
        Calculate perplexity of text.
        
        Args:
            text: Input text
            
        Returns:
            Perplexity score
        """
        try:
            # Tokenize text
            encodings = self.tokenizer(
                text,
                return_tensors='pt',
                truncation=True,
                max_length=1024
            )
            
            input_ids = encodings.input_ids.to(self.device)
            
            # Calculate perplexity using sliding window
            stride = self.config.get('stride', 512)
            max_length = input_ids.size(1)
            
            nlls = []
            prev_end_loc = 0
            
            for begin_loc in range(0, max_length, stride):
                end_loc = min(begin_loc + stride, max_length)
                trg_len = end_loc - prev_end_loc
                
                input_ids_chunk = input_ids[:, begin_loc:end_loc]
                target_ids = input_ids_chunk.clone()
                target_ids[:, :-trg_len] = -100
                
                with torch.no_grad():
                    outputs = self.model(input_ids_chunk, labels=target_ids)
                    neg_log_likelihood = outputs.loss * trg_len
                
                nlls.append(neg_log_likelihood)
                prev_end_loc = end_loc
                
                if end_loc == max_length:
                    break
            
            # Calculate perplexity
            total_nll = torch.stack(nlls).sum()
            perplexity = torch.exp(total_nll / max_length).item()
            
            return perplexity
            
        except Exception as e:
            logger.error(f"Error calculating perplexity: {str(e)}")
            return float('inf')
    
    def predict(self, text: str) -> Dict[str, Any]:
        """
        Predict if text is AI-generated based on perplexity.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary with prediction results
        """
        try:
            perplexity = self.calculate_perplexity(text)
            
            # Lower perplexity suggests AI-generated text
            # These thresholds should be calibrated on real data
            if perplexity < 20:
                is_ai = True
                confidence = min(0.9, (50 - perplexity) / 30)
            elif perplexity < 50:
                is_ai = False
                confidence = min(0.8, (perplexity - 20) / 30)
            else:
                is_ai = False
                confidence = 0.9
            
            # Ensure confidence is reasonable
            confidence = max(0.5, min(0.99, confidence))
            
            return {
                'is_ai': is_ai,
                'confidence': confidence,
                'perplexity': perplexity,
                'method': 'perplexity'
            }
            
        except Exception as e:
            logger.error(f"Error in perplexity prediction: {str(e)}")
            return {
                'is_ai': False,
                'confidence': 0.5,
                'perplexity': float('inf'),
                'method': 'perplexity',
                'error': str(e)
            }
    
    def analyze_sentences(self, text: str) -> List[Dict[str, Any]]:
        """
        Analyze perplexity of individual sentences.
        
        Args:
            text: Input text
            
        Returns:
            List of sentence-level analysis results
        """
        sentences = self._split_sentences(text)
        results = []
        
        for i, sentence in enumerate(sentences):
            if len(sentence.strip()) > 10:  # Skip very short sentences
                perplexity = self.calculate_perplexity(sentence)
                results.append({
                    'sentence_id': i,
                    'sentence': sentence,
                    'perplexity': perplexity,
                    'length': len(sentence.split())
                })
        
        return results
    
    def _split_sentences(self, text: str) -> List[str]:
        """
        Simple sentence splitting.
        
        Args:
            text: Input text
            
        Returns:
            List of sentences
        """
        # Simple sentence splitting (could be improved with spaCy/NLTK)
        import re
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def get_statistics(self, text: str) -> Dict[str, Any]:
        """
        Get detailed perplexity statistics.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with statistics
        """
        sentence_analysis = self.analyze_sentences(text)
        
        if not sentence_analysis:
            return {
                'overall_perplexity': float('inf'),
                'sentence_count': 0,
                'avg_sentence_perplexity': float('inf'),
                'min_perplexity': float('inf'),
                'max_perplexity': float('inf'),
                'std_perplexity': 0
            }
        
        perplexities = [s['perplexity'] for s in sentence_analysis]
        
        return {
            'overall_perplexity': self.calculate_perplexity(text),
            'sentence_count': len(sentence_analysis),
            'avg_sentence_perplexity': np.mean(perplexities),
            'min_perplexity': min(perplexities),
            'max_perplexity': max(perplexities),
            'std_perplexity': np.std(perplexities),
            'sentence_analysis': sentence_analysis
        }