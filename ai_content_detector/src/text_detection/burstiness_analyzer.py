"""Burstiness analyzer for AI-generated text detection."""

import numpy as np
import re
from typing import Dict, List, Any
from collections import Counter
import math

from ..utils import get_logger

logger = get_logger(__name__)


class BurstinessAnalyzer:
    """
    Burstiness analyzer for detecting AI-generated text.
    
    Analyzes sentence-level variance and patterns that differ between
    human and AI-generated text. AI text often has more consistent
    sentence structures and lengths.
    """
    
    def __init__(self):
        """Initialize burstiness analyzer."""
        self.features = [
            'sentence_length_variance',
            'word_repetition_burstiness',
            'punctuation_patterns',
            'syntactic_complexity',
            'vocabulary_diversity'
        ]
        logger.info("Initialized burstiness analyzer")
    
    def analyze_sentence_lengths(self, text: str) -> Dict[str, float]:
        """
        Analyze sentence length patterns.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with sentence length statistics
        """
        sentences = self._split_sentences(text)
        
        if len(sentences) < 2:
            return {
                'sentence_count': len(sentences),
                'avg_length': 0,
                'length_variance': 0,
                'length_std': 0,
                'burstiness_score': 0
            }
        
        lengths = [len(sentence.split()) for sentence in sentences]
        
        avg_length = np.mean(lengths)
        variance = np.var(lengths)
        std_dev = np.std(lengths)
        
        # Burstiness calculation (B = σ²/μ - 1)
        # Higher burstiness suggests more human-like variation
        burstiness = (variance / avg_length) - 1 if avg_length > 0 else 0
        
        return {
            'sentence_count': len(sentences),
            'avg_length': avg_length,
            'length_variance': variance,
            'length_std': std_dev,
            'burstiness_score': burstiness,
            'min_length': min(lengths),
            'max_length': max(lengths)
        }
    
    def analyze_word_repetition(self, text: str) -> Dict[str, float]:
        """
        Analyze word repetition patterns.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with word repetition statistics
        """
        words = re.findall(r'\b\w+\b', text.lower())
        
        if not words:
            return {
                'total_words': 0,
                'unique_words': 0,
                'repetition_rate': 0,
                'type_token_ratio': 0
            }
        
        word_counts = Counter(words)
        unique_words = len(word_counts)
        total_words = len(words)
        
        # Type-token ratio (vocabulary diversity)
        ttr = unique_words / total_words if total_words > 0 else 0
        
        # Repetition rate
        repeated_words = sum(1 for count in word_counts.values() if count > 1)
        repetition_rate = repeated_words / unique_words if unique_words > 0 else 0
        
        return {
            'total_words': total_words,
            'unique_words': unique_words,
            'repetition_rate': repetition_rate,
            'type_token_ratio': ttr,
            'most_common_words': word_counts.most_common(5)
        }
    
    def analyze_punctuation_patterns(self, text: str) -> Dict[str, float]:
        """
        Analyze punctuation usage patterns.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with punctuation statistics
        """
        # Count different punctuation marks
        punctuation_counts = {
            'periods': text.count('.'),
            'commas': text.count(','),
            'exclamations': text.count('!'),
            'questions': text.count('?'),
            'semicolons': text.count(';'),
            'colons': text.count(':'),
            'quotes': text.count('"') + text.count("'"),
            'parentheses': text.count('(') + text.count(')')
        }
        
        total_chars = len(text)
        total_punct = sum(punctuation_counts.values())
        
        punctuation_density = total_punct / total_chars if total_chars > 0 else 0
        
        # Calculate punctuation diversity (entropy)
        if total_punct > 0:
            punct_probs = [count / total_punct for count in punctuation_counts.values()]
            punct_entropy = -sum(p * math.log2(p) for p in punct_probs if p > 0)
        else:
            punct_entropy = 0
        
        return {
            'punctuation_density': punctuation_density,
            'punctuation_entropy': punct_entropy,
            'total_punctuation': total_punct,
            **punctuation_counts
        }
    
    def analyze_syntactic_complexity(self, text: str) -> Dict[str, float]:
        """
        Analyze syntactic complexity patterns.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with syntactic complexity metrics
        """
        sentences = self._split_sentences(text)
        
        if not sentences:
            return {
                'avg_words_per_sentence': 0,
                'complex_sentence_ratio': 0,
                'conjunction_usage': 0
            }
        
        # Calculate average words per sentence
        word_counts = [len(sentence.split()) for sentence in sentences]
        avg_words = np.mean(word_counts)
        
        # Count complex sentences (with conjunctions)
        complex_indicators = ['and', 'but', 'or', 'because', 'since', 'although', 'while', 'if']
        complex_sentences = 0
        total_conjunctions = 0
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            sentence_conjunctions = sum(sentence_lower.count(conj) for conj in complex_indicators)
            total_conjunctions += sentence_conjunctions
            
            if sentence_conjunctions > 0:
                complex_sentences += 1
        
        complex_ratio = complex_sentences / len(sentences) if sentences else 0
        conjunction_density = total_conjunctions / len(sentences) if sentences else 0
        
        return {
            'avg_words_per_sentence': avg_words,
            'complex_sentence_ratio': complex_ratio,
            'conjunction_usage': conjunction_density,
            'sentence_count': len(sentences)
        }
    
    def predict(self, text: str) -> Dict[str, Any]:
        """
        Predict if text is AI-generated based on burstiness features.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary with prediction results
        """
        try:
            # Get all feature analyses
            length_analysis = self.analyze_sentence_lengths(text)
            word_analysis = self.analyze_word_repetition(text)
            punct_analysis = self.analyze_punctuation_patterns(text)
            syntax_analysis = self.analyze_syntactic_complexity(text)
            
            # Calculate composite burstiness score
            # These weights should be calibrated on real data
            burstiness_factors = [
                length_analysis['burstiness_score'] * 0.3,
                word_analysis['type_token_ratio'] * 0.25,
                punct_analysis['punctuation_entropy'] * 0.2,
                syntax_analysis['complex_sentence_ratio'] * 0.25
            ]
            
            composite_burstiness = sum(burstiness_factors)
            
            # Lower burstiness suggests AI-generated text
            # These thresholds should be calibrated
            if composite_burstiness < 0.3:
                is_ai = True
                confidence = min(0.8, (0.6 - composite_burstiness) / 0.3)
            elif composite_burstiness < 0.6:
                is_ai = False
                confidence = min(0.7, (composite_burstiness - 0.3) / 0.3)
            else:
                is_ai = False
                confidence = 0.8
            
            confidence = max(0.5, min(0.95, confidence))
            
            return {
                'is_ai': is_ai,
                'confidence': confidence,
                'burstiness_score': composite_burstiness,
                'method': 'burstiness',
                'features': {
                    'sentence_analysis': length_analysis,
                    'word_analysis': word_analysis,
                    'punctuation_analysis': punct_analysis,
                    'syntax_analysis': syntax_analysis
                }
            }
            
        except Exception as e:
            logger.error(f"Error in burstiness analysis: {str(e)}")
            return {
                'is_ai': False,
                'confidence': 0.5,
                'burstiness_score': 0,
                'method': 'burstiness',
                'error': str(e)
            }
    
    def _split_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences.
        
        Args:
            text: Input text
            
        Returns:
            List of sentences
        """
        # Improved sentence splitting
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip() and len(s.strip()) > 5]