"""
Hybrid N-gram + LSTM code completion system
Combines fast n-gram predictions with LSTM for complex patterns
"""
from typing import List, Tuple, Optional
from .enhanced_model import EnhancedNGramModel
from .lstm_model import CodeLSTM
from .ast_tokenizer import ASTTokenizer, VocabularyManager


class HybridCodeCompleter:
    """
    Hybrid model that combines N-gram and LSTM predictions
    
    Strategy:
    - Use N-gram for fast, frequent patterns
    - Use LSTM for rare patterns and long-range dependencies
    - Combine predictions with weighted voting
    """
    
    def __init__(
        self,
        ngram_model: EnhancedNGramModel,
        lstm_model: Optional[CodeLSTM] = None,
        vocabulary: Optional[VocabularyManager] = None,
        tokenizer: Optional[ASTTokenizer] = None,
        ngram_weight: float = 0.6,
        lstm_weight: float = 0.4
    ):
        """
        Initialize hybrid completer
        
        Args:
            ngram_model: Trained N-gram model
            lstm_model: Trained LSTM model (optional)
            vocabulary: Vocabulary manager
            tokenizer: AST tokenizer
            ngram_weight: Weight for n-gram predictions (0-1)
            lstm_weight: Weight for LSTM predictions (0-1)
        """
        self.ngram_model = ngram_model
        self.lstm_model = lstm_model
        self.vocabulary = vocabulary
        self.tokenizer = tokenizer or ASTTokenizer()
        self.ngram_weight = ngram_weight
        self.lstm_weight = lstm_weight
    
    def suggest(self, code: str, top_k: int = 5, use_lstm: bool = True) -> List[Tuple[str, float]]:
        """
        Get code suggestions for partial code
        
        Args:
            code: Partial code string
            top_k: Number of suggestions to return
            use_lstm: Whether to use LSTM (if available)
            
        Returns:
            List of (token, probability) tuples
        """
        # Tokenize input
        tokens = self.tokenizer.tokenize(code)
        
        if not tokens:
            return []
        
        # Get N-gram suggestions
        ngram_suggestions = self.ngram_model.get_suggestions(tokens, top_k=top_k*2)
        
        # If LSTM is not available or not requested, return N-gram suggestions
        if not use_lstm or self.lstm_model is None or self.vocabulary is None:
            return ngram_suggestions[:top_k]
        
        # Get LSTM suggestions
        lstm_suggestions = self._get_lstm_suggestions(tokens, top_k=top_k*2)
        
        # Combine suggestions
        combined = self._combine_suggestions(ngram_suggestions, lstm_suggestions)
        
        return combined[:top_k]
    
    def _get_lstm_suggestions(self, tokens: List[str], top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Get LSTM suggestions
        
        Args:
            tokens: Context tokens
            top_k: Number of suggestions
            
        Returns:
            List of (token, probability) tuples
        """
        if self.lstm_model is None or self.vocabulary is None:
            return []
        
        # Encode tokens
        token_ids = self.vocabulary.encode(tokens)
        
        # Get predictions
        predictions = self.lstm_model.predict(token_ids, top_k=top_k)
        
        # Decode predictions
        suggestions = []
        for token_id, prob in predictions:
            token = self.vocabulary.id_to_token.get(token_id, '<UNK>')
            if token not in ['<UNK>', '<PAD>', '<s>', '<e>']:
                suggestions.append((token, prob))
        
        return suggestions
    
    def _combine_suggestions(
        self,
        ngram_suggestions: List[Tuple[str, float]],
        lstm_suggestions: List[Tuple[str, float]]
    ) -> List[Tuple[str, float]]:
        """
        Combine N-gram and LSTM suggestions with weighted voting
        
        Args:
            ngram_suggestions: N-gram suggestions
            lstm_suggestions: LSTM suggestions
            
        Returns:
            Combined suggestions
        """
        # Create dictionaries for easy lookup
        ngram_dict = {token: prob for token, prob in ngram_suggestions}
        lstm_dict = {token: prob for token, prob in lstm_suggestions}
        
        # Get all unique tokens
        all_tokens = set(ngram_dict.keys()) | set(lstm_dict.keys())
        
        # Calculate combined scores
        combined_scores = {}
        for token in all_tokens:
            ngram_prob = ngram_dict.get(token, 0.0)
            lstm_prob = lstm_dict.get(token, 0.0)
            
            # Weighted combination
            combined_score = (
                self.ngram_weight * ngram_prob +
                self.lstm_weight * lstm_prob
            )
            
            combined_scores[token] = combined_score
        
        # Sort by combined score
        sorted_suggestions = sorted(
            combined_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return sorted_suggestions
    
    def complete_code(
        self,
        code: str,
        max_tokens: int = 10,
        min_probability: float = 0.05,
        use_lstm: bool = True
    ) -> str:
        """
        Auto-complete code by generating multiple tokens
        
        Args:
            code: Partial code string
            max_tokens: Maximum tokens to generate
            min_probability: Minimum probability threshold
            use_lstm: Whether to use LSTM
            
        Returns:
            Completed code string
        """
        current_code = code
        tokens_generated = 0
        
        while tokens_generated < max_tokens:
            suggestions = self.suggest(current_code, top_k=1, use_lstm=use_lstm)
            
            if not suggestions or suggestions[0][1] < min_probability:
                break
            
            next_token = suggestions[0][0]
            current_code += " " + next_token
            tokens_generated += 1
        
        return current_code
    
    def get_multiple_completions(
        self,
        code: str,
        num_completions: int = 3,
        max_tokens: int = 5,
        use_lstm: bool = True
    ) -> List[str]:
        """
        Get multiple possible completions
        
        Args:
            code: Partial code string
            num_completions: Number of completions
            max_tokens: Max tokens per completion
            use_lstm: Whether to use LSTM
            
        Returns:
            List of completed code strings
        """
        completions = []
        suggestions = self.suggest(code, top_k=num_completions, use_lstm=use_lstm)
        
        for token, prob in suggestions[:num_completions]:
            completion = code + " " + token
            
            # Generate a few more tokens
            for _ in range(max_tokens - 1):
                next_suggestions = self.suggest(completion, top_k=1, use_lstm=use_lstm)
                if next_suggestions and next_suggestions[0][1] > 0.05:
                    completion += " " + next_suggestions[0][0]
                else:
                    break
            
            completions.append(completion)
        
        return completions
