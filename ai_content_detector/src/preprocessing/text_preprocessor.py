"""Text preprocessing utilities for AI content detection."""

import re
import string
import unicodedata
import numpy as np
from typing import List, Dict, Any, Optional
from ..utils import get_logger

logger = get_logger(__name__)


class TextPreprocessor:
    """
    Text preprocessing utilities for AI content detection.
    
    Handles cleaning, normalization, and feature extraction from text.
    """
    
    def __init__(self):
        """Initialize text preprocessor."""
        self.min_length = 10
        self.max_length = 10000
        logger.info("Text preprocessor initialized")
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text.
        
        Args:
            text: Raw input text
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Convert to string if not already
        text = str(text)
        
        # Normalize unicode characters
        text = unicodedata.normalize('NFKC', text)
        
        # Remove or replace problematic characters
        text = self._remove_special_characters(text)
        
        # Normalize whitespace
        text = self._normalize_whitespace(text)
        
        return text.strip()
    
    def _remove_special_characters(self, text: str) -> str:
        """Remove or replace special characters."""
        # Remove zero-width characters
        text = re.sub(r'[\u200b\u200c\u200d\ufeff]', '', text)
        
        # Replace smart quotes with regular quotes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        
        # Replace em/en dashes with regular dashes
        text = text.replace('—', '-').replace('–', '-')
        
        # Replace ellipsis
        text = text.replace('…', '...')
        
        return text
    
    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace in text."""
        # Replace multiple whitespace with single space
        text = re.sub(r'\s+', ' ', text)
        
        # Normalize line breaks
        text = re.sub(r'\r\n|\r|\n', '\n', text)
        
        # Remove excessive line breaks
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text
    
    def validate_text(self, text: str) -> Dict[str, Any]:
        """
        Validate text for processing.
        
        Args:
            text: Text to validate
            
        Returns:
            Validation results
        """
        if not text:
            return {
                'valid': False,
                'reason': 'Empty text',
                'length': 0
            }
        
        text_length = len(text.strip())
        
        if text_length < self.min_length:
            return {
                'valid': False,
                'reason': f'Text too short (minimum {self.min_length} characters)',
                'length': text_length
            }
        
        if text_length > self.max_length:
            return {
                'valid': False,
                'reason': f'Text too long (maximum {self.max_length} characters)',
                'length': text_length
            }
        
        # Check for mostly non-text content
        non_text_ratio = self._calculate_non_text_ratio(text)
        if non_text_ratio > 0.5:
            return {
                'valid': False,
                'reason': 'Text contains too many non-text characters',
                'length': text_length,
                'non_text_ratio': non_text_ratio
            }
        
        return {
            'valid': True,
            'length': text_length,
            'non_text_ratio': non_text_ratio
        }
    
    def _calculate_non_text_ratio(self, text: str) -> float:
        """Calculate ratio of non-text characters."""
        if not text:
            return 1.0
        
        # Count printable characters
        printable_chars = sum(1 for c in text if c.isprintable() and not c.isspace())
        total_chars = len(text.replace(' ', '').replace('\n', '').replace('\t', ''))
        
        if total_chars == 0:
            return 1.0
        
        return 1 - (printable_chars / total_chars)
    
    def extract_features(self, text: str) -> Dict[str, Any]:
        """
        Extract basic features from text.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of extracted features
        """
        if not text:
            return {}
        
        # Basic statistics
        words = text.split()
        sentences = self._split_sentences(text)
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        features = {
            'char_count': len(text),
            'word_count': len(words),
            'sentence_count': len(sentences),
            'paragraph_count': len(paragraphs),
            'avg_word_length': np.mean([len(word) for word in words]) if words else 0,
            'avg_sentence_length': np.mean([len(sent.split()) for sent in sentences]) if sentences else 0,
        }
        
        # Vocabulary features
        unique_words = set(word.lower() for word in words if word.isalpha())
        features['unique_word_count'] = len(unique_words)
        features['lexical_diversity'] = len(unique_words) / len(words) if words else 0
        
        # Punctuation features
        features['punctuation_count'] = sum(1 for c in text if c in string.punctuation)
        features['punctuation_ratio'] = features['punctuation_count'] / len(text) if text else 0
        
        # Character-level features
        features['uppercase_ratio'] = sum(1 for c in text if c.isupper()) / len(text) if text else 0
        features['digit_ratio'] = sum(1 for c in text if c.isdigit()) / len(text) if text else 0
        
        return features
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting (could be improved with spaCy/NLTK)
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def chunk_text(self, text: str, max_length: int = 512, overlap: int = 50) -> List[str]:
        """
        Split text into chunks for processing.
        
        Args:
            text: Input text
            max_length: Maximum chunk length in characters
            overlap: Overlap between chunks in characters
            
        Returns:
            List of text chunks
        """
        if len(text) <= max_length:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = min(start + max_length, len(text))
            
            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence ending within last 100 characters
                search_start = max(end - 100, start)
                sentence_ends = [m.end() for m in re.finditer(r'[.!?]+\s+', text[search_start:end])]
                
                if sentence_ends:
                    # Use the last sentence ending
                    end = search_start + sentence_ends[-1]
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - overlap
            
            # Prevent infinite loop
            if start >= end:
                break
        
        return chunks
    
    def preprocess_for_model(self, text: str, model_type: str = "general") -> str:
        """
        Preprocess text for specific model types.
        
        Args:
            text: Input text
            model_type: Type of model ('roberta', 'gpt2', 'general')
            
        Returns:
            Preprocessed text
        """
        # Clean text
        text = self.clean_text(text)
        
        # Model-specific preprocessing
        if model_type == "roberta":
            # RoBERTa handles most preprocessing internally
            # Just ensure reasonable length
            if len(text) > 5000:  # Rough character limit
                text = text[:5000]
        
        elif model_type == "gpt2":
            # GPT-2 preprocessing
            # Remove excessive whitespace
            text = re.sub(r'\n{3,}', '\n\n', text)
            
        elif model_type == "general":
            # General preprocessing
            pass
        
        return text