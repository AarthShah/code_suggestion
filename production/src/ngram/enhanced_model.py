"""
Enhanced N-gram model with large N (up to 15) and vocabulary management
"""
from collections import Counter, defaultdict
from typing import List, Tuple, Optional
import pickle
from tqdm import tqdm


class EnhancedNGramModel:
    """N-gram language model with smoothing and backoff"""
    
    def __init__(self, n: int = 15, smoothing: float = 1.0):
        """
        Initialize enhanced N-gram model
        
        Args:
            n: N-gram size (recommended: 15 for code)
            smoothing: Laplace smoothing parameter (k)
        """
        self.n = n
        self.smoothing = smoothing
        self.ngram_counts = defaultdict(Counter)  # (context) -> Counter({next_token: count})
        self.context_counts = Counter()  # Total count for each context
        self.vocabulary = set()
    
    def add_sequence(self, tokens: List[str]):
        """
        Add a token sequence to the model
        
        Args:
            tokens: List of tokens
        """
        # Add start/end tokens
        padded = ['<s>'] * (self.n - 1) + tokens + ['<e>']
        
        # Update vocabulary
        self.vocabulary.update(tokens)
        
        # Extract n-grams
        for i in range(len(padded) - self.n + 1):
            context = tuple(padded[i:i+self.n-1])
            next_token = padded[i+self.n-1]
            
            self.ngram_counts[context][next_token] += 1
            self.context_counts[context] += 1
    
    def get_probability(self, context: Tuple[str, ...], token: str) -> float:
        """
        Get probability of token given context with smoothing
        
        Args:
            context: Tuple of context tokens (length n-1)
            token: Next token
            
        Returns:
            Probability with Laplace smoothing
        """
        context_count = self.context_counts.get(context, 0)
        token_count = self.ngram_counts[context].get(token, 0)
        vocab_size = len(self.vocabulary)
        
        # Laplace smoothing: P(w|context) = (count(w, context) + k) / (count(context) + k * V)
        probability = (token_count + self.smoothing) / (context_count + self.smoothing * vocab_size)
        
        return probability
    
    def get_suggestions(self, tokens: List[str], top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Get top-k suggestions for next token with backoff
        
        Args:
            tokens: Context tokens
            top_k: Number of suggestions to return
            
        Returns:
            List of (token, probability) tuples
        """
        # Try different context lengths (backoff from n-1 to 1)
        for context_len in range(min(self.n - 1, len(tokens)), 0, -1):
            context = tuple(tokens[-context_len:])
            
            if context in self.ngram_counts:
                # Get all possible next tokens
                next_token_counts = self.ngram_counts[context]
                total_count = sum(next_token_counts.values())
                
                # Calculate probabilities
                suggestions = []
                for token, count in next_token_counts.items():
                    probability = count / total_count
                    suggestions.append((token, probability))
                
                # Sort by probability
                suggestions.sort(key=lambda x: x[1], reverse=True)
                
                return suggestions[:top_k]
        
        # If no context found, return empty list
        return []
    
    def perplexity(self, test_sequences: List[List[str]]) -> float:
        """
        Calculate perplexity on test sequences
        
        Args:
            test_sequences: List of token sequences
            
        Returns:
            Perplexity score (lower is better)
        """
        import math
        
        log_prob_sum = 0
        token_count = 0
        
        for tokens in test_sequences:
            padded = ['<s>'] * (self.n - 1) + tokens + ['<e>']
            
            for i in range(len(padded) - self.n + 1):
                context = tuple(padded[i:i+self.n-1])
                next_token = padded[i+self.n-1]
                
                prob = self.get_probability(context, next_token)
                if prob > 0:
                    log_prob_sum += math.log(prob)
                    token_count += 1
        
        if token_count == 0:
            return float('inf')
        
        avg_log_prob = log_prob_sum / token_count
        perplexity = math.exp(-avg_log_prob)
        
        return perplexity
    
    def save(self, filepath: str):
        """Save model to file"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'n': self.n,
                'smoothing': self.smoothing,
                'ngram_counts': dict(self.ngram_counts),
                'context_counts': dict(self.context_counts),
                'vocabulary': self.vocabulary
            }, f)
    
    def load(self, filepath: str):
        """Load model from file"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.n = data['n']
            self.smoothing = data['smoothing']
            self.ngram_counts = defaultdict(Counter, data['ngram_counts'])
            self.context_counts = Counter(data['context_counts'])
            self.vocabulary = data['vocabulary']
    
    def train_from_sequences(self, sequences: List[List[str]], show_progress: bool = True):
        """
        Train model from token sequences
        
        Args:
            sequences: List of token sequences
            show_progress: Show progress bar
        """
        iterator = tqdm(sequences, desc="Training N-gram model") if show_progress else sequences
        
        for tokens in iterator:
            self.add_sequence(tokens)
    
    @property
    def stats(self) -> dict:
        """Get model statistics"""
        total_ngrams = sum(sum(counts.values()) for counts in self.ngram_counts.values())
        
        return {
            'n': self.n,
            'unique_contexts': len(self.ngram_counts),
            'total_ngrams': total_ngrams,
            'vocabulary_size': len(self.vocabulary),
            'smoothing': self.smoothing
        }
