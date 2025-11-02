"""
Trainer module for building N-gram models from code datasets.
"""
import json
from typing import List, Dict, Optional
from tqdm import tqdm
from .tokenizer import CodeTokenizer
from .model import NGramModel


class NGramTrainer:
    """Trainer for N-gram code suggestion models."""
    
    def __init__(self, n: int = 3):
        """
        Initialize trainer.
        
        Args:
            n: Size of n-grams to use
        """
        self.n = n
        self.tokenizer = CodeTokenizer()
        self.model = NGramModel(n=n)
    
    def train_from_dataset(self, dataset_path: str, max_samples: Optional[int] = None):
        """
        Train model from JSON dataset.
        
        Args:
            dataset_path: Path to JSON dataset file
            max_samples: Maximum number of samples to use (None for all)
        """
        print(f"Loading dataset from {dataset_path}...")
        
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if max_samples:
            data = data[:max_samples]
        
        print(f"Training on {len(data)} code samples...")
        
        for item in tqdm(data, desc="Training"):
            # Extract code from 'output' field
            if 'output' in item and item['output']:
                code = item['output']
                self._process_code(code)
    
    def train_from_code_list(self, code_snippets: List[str]):
        """
        Train model from a list of code snippets.
        
        Args:
            code_snippets: List of code strings
        """
        print(f"Training on {len(code_snippets)} code snippets...")
        
        for code in tqdm(code_snippets, desc="Training"):
            self._process_code(code)
    
    def _process_code(self, code: str):
        """
        Process a single code snippet.
        
        Args:
            code: Code string to process
        """
        # Tokenize the code
        tokens = self.tokenizer.tokenize(code)
        
        # Add to model
        if tokens:
            self.model.add_sequence(tokens)
    
    def save_model(self, filepath: str):
        """
        Save trained model to file.
        
        Args:
            filepath: Path to save model
        """
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def get_model(self) -> NGramModel:
        """
        Get the trained model.
        
        Returns:
            Trained NGramModel
        """
        return self.model
    
    def get_stats(self) -> Dict:
        """
        Get training statistics.
        
        Returns:
            Dictionary with training statistics
        """
        return {
            'n': self.n,
            'num_contexts': len(self.model.context_counts),
            'total_ngrams': sum(self.model.context_counts.values()),
            'unique_tokens': len(set(
                token 
                for counter in self.model.ngram_counts.values() 
                for token in counter.keys()
            ))
        }