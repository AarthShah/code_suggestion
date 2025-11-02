"""
Code suggester module using trained N-gram model.
"""
from typing import List, Tuple
from .tokenizer import CodeTokenizer
from .model import NGramModel


class CodeSuggester:
    """Provides code suggestions based on N-gram model."""
    
    def __init__(self, model: NGramModel):
        """
        Initialize suggester with a trained model.
        
        Args:
            model: Trained NGramModel
        """
        self.model = model
        self.tokenizer = CodeTokenizer()
    
    def suggest(self, code: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Get code suggestions for the given partial code.
        
        Args:
            code: Partial code string
            top_k: Number of suggestions to return
            
        Returns:
            List of (token, probability) tuples
        """
        # Tokenize input code
        tokens = self.tokenizer.tokenize(code)
        
        if not tokens:
            return []
        
        # Get suggestions from model
        suggestions = self.model.get_suggestions(tokens, top_k=top_k)
        
        return suggestions
    
    def suggest_next_token(self, code: str) -> str:
        """
        Get the most likely next token.
        
        Args:
            code: Partial code string
            
        Returns:
            Most likely next token (empty string if no suggestion)
        """
        suggestions = self.suggest(code, top_k=1)
        
        if suggestions:
            return suggestions[0][0]
        return ""
    
    def complete_code(self, code: str, max_tokens: int = 10, 
                     min_probability: float = 0.1) -> str:
        """
        Auto-complete code by generating multiple tokens.
        
        Args:
            code: Partial code string
            max_tokens: Maximum number of tokens to generate
            min_probability: Minimum probability threshold for suggestions
            
        Returns:
            Completed code string
        """
        current_code = code
        tokens_generated = 0
        
        while tokens_generated < max_tokens:
            suggestions = self.suggest(current_code, top_k=1)
            
            if not suggestions or suggestions[0][1] < min_probability:
                break
            
            next_token = suggestions[0][0]
            current_code += " " + next_token
            tokens_generated += 1
        
        return current_code
    
    def get_multiple_completions(self, code: str, num_completions: int = 3,
                                max_tokens: int = 5) -> List[str]:
        """
        Get multiple possible code completions.
        
        Args:
            code: Partial code string
            num_completions: Number of different completions to generate
            max_tokens: Maximum tokens per completion
            
        Returns:
            List of completed code strings
        """
        completions = []
        
        for _ in range(num_completions):
            suggestions = self.suggest(code, top_k=num_completions)
            
            if not suggestions:
                break
            
            for token, prob in suggestions[:num_completions]:
                completion = code + " " + token
                
                # Generate a few more tokens
                for _ in range(max_tokens - 1):
                    next_suggestions = self.suggest(completion, top_k=1)
                    if next_suggestions and next_suggestions[0][1] > 0.05:
                        completion += " " + next_suggestions[0][0]
                    else:
                        break
                
                completions.append(completion)
                
                if len(completions) >= num_completions:
                    break
            
            if len(completions) >= num_completions:
                break
        
        return completions[:num_completions]