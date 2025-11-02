"""
Evaluator module for assessing N-gram model performance.
"""
import json
from typing import List, Dict, Optional
from tqdm import tqdm
from .tokenizer import CodeTokenizer
from .model import NGramModel


class NGramEvaluator:
    """Evaluator for N-gram code suggestion models."""
    
    def __init__(self, model: NGramModel):
        """
        Initialize evaluator.
        
        Args:
            model: Trained NGramModel to evaluate
        """
        self.model = model
        self.tokenizer = CodeTokenizer()
    
    def evaluate_dataset(self, dataset_path: str, max_samples: Optional[int] = None) -> Dict:
        """
        Evaluate model on a dataset.
        
        Args:
            dataset_path: Path to JSON dataset file
            max_samples: Maximum number of samples to evaluate
            
        Returns:
            Dictionary with evaluation metrics
        """
        print(f"Loading dataset from {dataset_path}...")
        
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if max_samples:
            data = data[:max_samples]
        
        total_predictions = 0
        correct_predictions = 0
        top_3_correct = 0
        top_5_correct = 0
        
        print(f"Evaluating on {len(data)} samples...")
        
        for item in tqdm(data, desc="Evaluating"):
            if 'output' not in item or not item['output']:
                continue
            
            code = item['output']
            tokens = self.tokenizer.tokenize(code)
            
            if len(tokens) < self.model.n:
                continue
            
            # Test each position in the sequence
            for i in range(self.model.n - 1, len(tokens)):
                context = tokens[:i]
                actual_token = tokens[i]
                
                # Get predictions
                predictions = self.model.get_suggestions(context, top_k=5)
                
                if not predictions:
                    continue
                
                total_predictions += 1
                
                # Check if actual token is in top-1
                if predictions[0][0] == actual_token:
                    correct_predictions += 1
                    top_3_correct += 1
                    top_5_correct += 1
                # Check if in top-3
                elif any(token == actual_token for token, _ in predictions[:3]):
                    top_3_correct += 1
                    top_5_correct += 1
                # Check if in top-5
                elif any(token == actual_token for token, _ in predictions[:5]):
                    top_5_correct += 1
        
        # Calculate metrics
        accuracy_top1 = correct_predictions / total_predictions if total_predictions > 0 else 0
        accuracy_top3 = top_3_correct / total_predictions if total_predictions > 0 else 0
        accuracy_top5 = top_5_correct / total_predictions if total_predictions > 0 else 0
        
        return {
            'total_predictions': total_predictions,
            'accuracy_top1': accuracy_top1,
            'accuracy_top3': accuracy_top3,
            'accuracy_top5': accuracy_top5,
        }
    
    def print_evaluation_report(self, metrics: Dict):
        """
        Print a formatted evaluation report.
        
        Args:
            metrics: Dictionary with evaluation metrics
        """
        print("\n" + "=" * 60)
        print("Evaluation Report")
        print("=" * 60)
        print(f"Total predictions: {metrics['total_predictions']:,}")
        print(f"Top-1 Accuracy: {metrics['accuracy_top1']:.2%}")
        print(f"Top-3 Accuracy: {metrics['accuracy_top3']:.2%}")
        print(f"Top-5 Accuracy: {metrics['accuracy_top5']:.2%}")
        print("=" * 60)
