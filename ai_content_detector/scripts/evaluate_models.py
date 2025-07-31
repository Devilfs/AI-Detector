#!/usr/bin/env python3
"""
Evaluation script for AI Content Detection System.
Benchmarks text and image detection models on various datasets.
"""

import sys
import os
from pathlib import Path
import argparse
import time
import json
from typing import Dict, List, Any
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.text_detection import TextEnsemble
from src.image_detection import ImageEnsemble
from src.utils import setup_logger
from PIL import Image


class EvaluationFramework:
    """Framework for evaluating AI content detection models."""
    
    def __init__(self, output_dir: str = "evaluation_results"):
        """Initialize evaluation framework."""
        self.logger = setup_logger(__name__)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize models
        self.text_ensemble = None
        self.image_ensemble = None
        
        self.logger.info("Evaluation framework initialized")
    
    def load_models(self):
        """Load detection models."""
        self.logger.info("Loading detection models...")
        
        try:
            self.text_ensemble = TextEnsemble()
            self.logger.info("✓ Text ensemble loaded")
        except Exception as e:
            self.logger.error(f"Failed to load text ensemble: {e}")
        
        try:
            self.image_ensemble = ImageEnsemble()
            self.logger.info("✓ Image ensemble loaded")
        except Exception as e:
            self.logger.error(f"Failed to load image ensemble: {e}")
    
    def evaluate_text_detection(self, dataset_path: str) -> Dict[str, Any]:
        """
        Evaluate text detection models.
        
        Args:
            dataset_path: Path to text dataset (should contain ai/ and human/ folders)
            
        Returns:
            Evaluation results
        """
        if self.text_ensemble is None:
            raise ValueError("Text ensemble not loaded")
        
        self.logger.info(f"Evaluating text detection on {dataset_path}")
        
        # Load dataset
        texts, labels, sources = self._load_text_dataset(dataset_path)
        
        if len(texts) == 0:
            raise ValueError("No texts found in dataset")
        
        # Run predictions
        predictions = []
        confidences = []
        processing_times = []
        
        for i, text in enumerate(texts):
            if i % 100 == 0:
                self.logger.info(f"Processing text {i+1}/{len(texts)}")
            
            start_time = time.time()
            result = self.text_ensemble.predict(text)
            processing_time = (time.time() - start_time) * 1000
            
            predictions.append(1 if result['is_ai'] else 0)
            confidences.append(result['confidence'])
            processing_times.append(processing_time)
        
        # Calculate metrics
        metrics = self._calculate_metrics(labels, predictions, confidences)
        metrics['avg_processing_time_ms'] = np.mean(processing_times)
        metrics['total_samples'] = len(texts)
        
        # Per-source analysis
        if sources:
            source_metrics = self._analyze_by_source(labels, predictions, sources)
            metrics['source_analysis'] = source_metrics
        
        # Save results
        self._save_text_results(metrics, texts, labels, predictions, confidences)
        
        return metrics
    
    def evaluate_image_detection(self, dataset_path: str) -> Dict[str, Any]:
        """
        Evaluate image detection models.
        
        Args:
            dataset_path: Path to image dataset (should contain ai/ and real/ folders)
            
        Returns:
            Evaluation results
        """
        if self.image_ensemble is None:
            raise ValueError("Image ensemble not loaded")
        
        self.logger.info(f"Evaluating image detection on {dataset_path}")
        
        # Load dataset
        image_paths, labels = self._load_image_dataset(dataset_path)
        
        if len(image_paths) == 0:
            raise ValueError("No images found in dataset")
        
        # Run predictions
        predictions = []
        confidences = []
        processing_times = []
        
        for i, img_path in enumerate(image_paths):
            if i % 50 == 0:
                self.logger.info(f"Processing image {i+1}/{len(image_paths)}")
            
            try:
                image = Image.open(img_path).convert('RGB')
                
                start_time = time.time()
                result = self.image_ensemble.predict(image)
                processing_time = (time.time() - start_time) * 1000
                
                predictions.append(1 if result['is_ai'] else 0)
                confidences.append(result['confidence'])
                processing_times.append(processing_time)
                
            except Exception as e:
                self.logger.error(f"Failed to process {img_path}: {e}")
                predictions.append(0)  # Default to human
                confidences.append(0.5)
                processing_times.append(0)
        
        # Calculate metrics
        metrics = self._calculate_metrics(labels, predictions, confidences)
        metrics['avg_processing_time_ms'] = np.mean(processing_times)
        metrics['total_samples'] = len(image_paths)
        
        # Save results
        self._save_image_results(metrics, image_paths, labels, predictions, confidences)
        
        return metrics
    
    def _load_text_dataset(self, dataset_path: str) -> tuple:
        """Load text dataset from directory structure."""
        dataset_path = Path(dataset_path)
        texts = []
        labels = []
        sources = []
        
        # Load AI-generated texts
        ai_dir = dataset_path / "ai"
        if ai_dir.exists():
            for file_path in ai_dir.glob("*.txt"):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        text = f.read().strip()
                        if len(text) > 50:  # Minimum length filter
                            texts.append(text)
                            labels.append(1)
                            sources.append(file_path.stem)
                except Exception as e:
                    self.logger.warning(f"Failed to read {file_path}: {e}")
        
        # Load human-written texts
        human_dir = dataset_path / "human"
        if human_dir.exists():
            for file_path in human_dir.glob("*.txt"):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        text = f.read().strip()
                        if len(text) > 50:  # Minimum length filter
                            texts.append(text)
                            labels.append(0)
                            sources.append(file_path.stem)
                except Exception as e:
                    self.logger.warning(f"Failed to read {file_path}: {e}")
        
        self.logger.info(f"Loaded {len(texts)} texts ({sum(labels)} AI, {len(labels) - sum(labels)} human)")
        
        return texts, labels, sources
    
    def _load_image_dataset(self, dataset_path: str) -> tuple:
        """Load image dataset from directory structure."""
        dataset_path = Path(dataset_path)
        image_paths = []
        labels = []
        
        # Supported image extensions
        img_extensions = ['.jpg', '.jpeg', '.png', '.webp', '.bmp']
        
        # Load AI-generated images
        ai_dir = dataset_path / "ai"
        if ai_dir.exists():
            for ext in img_extensions:
                for file_path in ai_dir.glob(f"*{ext}"):
                    image_paths.append(str(file_path))
                    labels.append(1)
        
        # Load real images
        real_dir = dataset_path / "real"
        if real_dir.exists():
            for ext in img_extensions:
                for file_path in real_dir.glob(f"*{ext}"):
                    image_paths.append(str(file_path))
                    labels.append(0)
        
        self.logger.info(f"Loaded {len(image_paths)} images ({sum(labels)} AI, {len(labels) - sum(labels)} real)")
        
        return image_paths, labels
    
    def _calculate_metrics(self, labels: List[int], predictions: List[int], confidences: List[float]) -> Dict[str, float]:
        """Calculate evaluation metrics."""
        metrics = {
            'accuracy': accuracy_score(labels, predictions),
            'precision': precision_score(labels, predictions, zero_division=0),
            'recall': recall_score(labels, predictions, zero_division=0),
            'f1_score': f1_score(labels, predictions, zero_division=0),
        }
        
        # Calculate AUC if possible
        try:
            from sklearn.metrics import roc_auc_score
            # Convert confidences to AI probabilities
            ai_probs = [conf if pred == 1 else (1 - conf) for pred, conf in zip(predictions, confidences)]
            metrics['auc'] = roc_auc_score(labels, ai_probs)
        except Exception as e:
            self.logger.warning(f"Could not calculate AUC: {e}")
            metrics['auc'] = 0.0
        
        return metrics
    
    def _analyze_by_source(self, labels: List[int], predictions: List[int], sources: List[str]) -> Dict[str, Dict[str, float]]:
        """Analyze performance by source/model."""
        source_metrics = {}
        
        unique_sources = set(sources)
        for source in unique_sources:
            source_indices = [i for i, s in enumerate(sources) if s == source]
            source_labels = [labels[i] for i in source_indices]
            source_predictions = [predictions[i] for i in source_indices]
            
            if len(source_labels) > 0:
                source_metrics[source] = {
                    'accuracy': accuracy_score(source_labels, source_predictions),
                    'precision': precision_score(source_labels, source_predictions, zero_division=0),
                    'recall': recall_score(source_labels, source_predictions, zero_division=0),
                    'f1_score': f1_score(source_labels, source_predictions, zero_division=0),
                    'samples': len(source_labels)
                }
        
        return source_metrics
    
    def _save_text_results(self, metrics: Dict[str, Any], texts: List[str], 
                          labels: List[int], predictions: List[int], confidences: List[float]):
        """Save text evaluation results."""
        # Save metrics
        with open(self.output_dir / "text_metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Save detailed results
        results_df = pd.DataFrame({
            'text_preview': [text[:100] + "..." if len(text) > 100 else text for text in texts],
            'true_label': ['AI' if label == 1 else 'Human' for label in labels],
            'predicted_label': ['AI' if pred == 1 else 'Human' for pred in predictions],
            'confidence': confidences,
            'correct': [l == p for l, p in zip(labels, predictions)]
        })
        results_df.to_csv(self.output_dir / "text_results.csv", index=False)
        
        # Create confusion matrix plot
        self._plot_confusion_matrix(labels, predictions, "text")
        
        self.logger.info(f"Text evaluation results saved to {self.output_dir}")
    
    def _save_image_results(self, metrics: Dict[str, Any], image_paths: List[str], 
                           labels: List[int], predictions: List[int], confidences: List[float]):
        """Save image evaluation results."""
        # Save metrics
        with open(self.output_dir / "image_metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Save detailed results
        results_df = pd.DataFrame({
            'image_path': image_paths,
            'true_label': ['AI' if label == 1 else 'Real' for label in labels],
            'predicted_label': ['AI' if pred == 1 else 'Real' for pred in predictions],
            'confidence': confidences,
            'correct': [l == p for l, p in zip(labels, predictions)]
        })
        results_df.to_csv(self.output_dir / "image_results.csv", index=False)
        
        # Create confusion matrix plot
        self._plot_confusion_matrix(labels, predictions, "image")
        
        self.logger.info(f"Image evaluation results saved to {self.output_dir}")
    
    def _plot_confusion_matrix(self, labels: List[int], predictions: List[int], data_type: str):
        """Plot and save confusion matrix."""
        cm = confusion_matrix(labels, predictions)
        
        plt.figure(figsize=(8, 6))
        if data_type == "text":
            class_names = ['Human', 'AI Generated']
        else:
            class_names = ['Real', 'AI Generated']
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title(f'{data_type.title()} Detection - Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f"{data_type}_confusion_matrix.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def run_comprehensive_evaluation(self, text_dataset: str = None, image_dataset: str = None):
        """Run comprehensive evaluation on both text and image datasets."""
        self.logger.info("Starting comprehensive evaluation...")
        
        self.load_models()
        
        results = {}
        
        # Evaluate text detection
        if text_dataset and self.text_ensemble:
            try:
                text_results = self.evaluate_text_detection(text_dataset)
                results['text'] = text_results
                self.logger.info(f"Text F1 Score: {text_results['f1_score']:.3f}")
            except Exception as e:
                self.logger.error(f"Text evaluation failed: {e}")
        
        # Evaluate image detection
        if image_dataset and self.image_ensemble:
            try:
                image_results = self.evaluate_image_detection(image_dataset)
                results['image'] = image_results
                self.logger.info(f"Image F1 Score: {image_results['f1_score']:.3f}")
            except Exception as e:
                self.logger.error(f"Image evaluation failed: {e}")
        
        # Save combined results
        with open(self.output_dir / "comprehensive_results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        # Print summary
        self._print_summary(results)
        
        return results
    
    def _print_summary(self, results: Dict[str, Any]):
        """Print evaluation summary."""
        print("\n" + "="*60)
        print("AI CONTENT DETECTION - EVALUATION SUMMARY")
        print("="*60)
        
        if 'text' in results:
            text_metrics = results['text']
            print(f"\n📝 TEXT DETECTION:")
            print(f"   Accuracy:  {text_metrics['accuracy']:.3f}")
            print(f"   Precision: {text_metrics['precision']:.3f}")
            print(f"   Recall:    {text_metrics['recall']:.3f}")
            print(f"   F1 Score:  {text_metrics['f1_score']:.3f}")
            print(f"   AUC:       {text_metrics.get('auc', 0):.3f}")
            print(f"   Avg Time:  {text_metrics['avg_processing_time_ms']:.1f}ms")
            print(f"   Samples:   {text_metrics['total_samples']}")
        
        if 'image' in results:
            image_metrics = results['image']
            print(f"\n🖼️  IMAGE DETECTION:")
            print(f"   Accuracy:  {image_metrics['accuracy']:.3f}")
            print(f"   Precision: {image_metrics['precision']:.3f}")
            print(f"   Recall:    {image_metrics['recall']:.3f}")
            print(f"   F1 Score:  {image_metrics['f1_score']:.3f}")
            print(f"   AUC:       {image_metrics.get('auc', 0):.3f}")
            print(f"   Avg Time:  {image_metrics['avg_processing_time_ms']:.1f}ms")
            print(f"   Samples:   {image_metrics['total_samples']}")
        
        print(f"\n📊 Results saved to: {self.output_dir}")
        print("="*60)


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate AI Content Detection System")
    parser.add_argument("--text-dataset", type=str, help="Path to text dataset directory")
    parser.add_argument("--image-dataset", type=str, help="Path to image dataset directory")
    parser.add_argument("--output-dir", type=str, default="evaluation_results", 
                       help="Output directory for results")
    parser.add_argument("--mode", choices=["text", "image", "both"], default="both",
                       help="Evaluation mode")
    
    args = parser.parse_args()
    
    # Create evaluation framework
    evaluator = EvaluationFramework(args.output_dir)
    
    # Run evaluation based on mode
    if args.mode == "text" and args.text_dataset:
        evaluator.load_models()
        evaluator.evaluate_text_detection(args.text_dataset)
    elif args.mode == "image" and args.image_dataset:
        evaluator.load_models()
        evaluator.evaluate_image_detection(args.image_dataset)
    elif args.mode == "both":
        evaluator.run_comprehensive_evaluation(args.text_dataset, args.image_dataset)
    else:
        print("Please provide appropriate dataset paths for the selected mode.")
        parser.print_help()


if __name__ == "__main__":
    main()