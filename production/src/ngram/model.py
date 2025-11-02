"""
N-gram language model for code suggestion.
Implements n-gram frequency counting and probability calculation.
"""
from collections import defaultdict, Counter
from typing import List, Tuple, Dict
import pickle


class NGramModel:
    """N-gram language model for code."""
    
    def __init__(self, n: int = 3):
        """
        Initialize N-gram model.
        
        Args:
            n: Size of n-grams (default: 3 for trigrams)
        """
        self.n = n
        self.ngram_counts = defaultdict(Counter)  # (context) -> {next_token: count}
        self.context_counts = Counter()  # Total counts for each context
        
    def add_sequence(self, tokens: List[str]):
        """
        Add a sequence of tokens to the model.
        
        Args:
            tokens: List of tokens to add
        """
        if len(tokens) < self.n:
            return
            
        for i in range(len(tokens) - self.n + 1):
            # Get context (first n-1 tokens) and next token
            context = tuple(tokens[i:i + self.n - 1])
            next_token = tokens[i + self.n - 1]
            
            # Update counts
            self.ngram_counts[context][next_token] += 1
            self.context_counts[context] += 1
    
    def get_suggestions(self, context: List[str], top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Get top-k suggestions for the given context with backoff.
        
        Args:
            context: List of tokens representing current context
            top_k: Number of suggestions to return
            
        Returns:
            List of (token, probability) tuples, sorted by probability
        """
        suggestions = []
        
        # Try full context first
        for i in range(min(len(context), self.n - 1), 0, -1):
            context_tuple = tuple(context[-i:])
            
            if context_tuple in self.ngram_counts:
                next_tokens = self.ngram_counts[context_tuple]
                total_count = self.context_counts[context_tuple]
                
                # Calculate probabilities
                for token, count in next_tokens.items():
                    probability = count / total_count
                    suggestions.append((token, probability))
                
                # If we found suggestions, break
                if suggestions:
                    break
        
        # Sort by probability (descending) and return top-k
        suggestions.sort(key=lambda x: x[1], reverse=True)
        return suggestions[:top_k]
    
    def get_probability(self, context: List[str], token: str) -> float:
        """
        Get probability of a token given the context.
        
        Args:
            context: List of tokens representing current context
            token: Token to get probability for
            
        Returns:
            Probability of token given context
        """
        context_tuple = tuple(context[-(self.n - 1):]) if len(context) >= self.n - 1 else tuple(context)
        
        if context_tuple not in self.ngram_counts:
            return 0.0
        
        token_count = self.ngram_counts[context_tuple].get(token, 0)
        total_count = self.context_counts[context_tuple]
        
        return token_count / total_count if total_count > 0 else 0.0
    
    def save(self, filepath: str):
        """Save model to file."""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'n': self.n,
                'ngram_counts': dict(self.ngram_counts),
                'context_counts': dict(self.context_counts)
            }, f)
    
    def load(self, filepath: str):
        """Load model from file."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.n = data['n']
            self.ngram_counts = defaultdict(Counter, data['ngram_counts'])
            self.context_counts = Counter(data['context_counts'])