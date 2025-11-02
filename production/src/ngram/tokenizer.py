"""
Tokenizer module for code suggestion system.
Handles tokenization of code snippets into meaningful tokens.
"""
import re
from typing import List


class CodeTokenizer:
    """Tokenizer for code snippets."""
    
    def __init__(self):
        # Pattern to match keywords, identifiers, operators, and punctuation
        # Order matters - more specific patterns first
        self.token_pattern = re.compile(
            r'//.*$|'  # C++ style comments (before operators)
            r'/\*.*?\*/|'  # C style comments
            r'#.*$|'  # Python comments
            r'\'\'\'.*?\'\'\'|'  # Triple single-quoted strings
            r'""".*?"""|'  # Triple double-quoted strings
            r"'(?:[^'\\]|\\.)*'|"  # Single-quoted strings with escapes
            r'"(?:[^"\\]|\\.)*"|'  # Double-quoted strings with escapes
            r'->|'  # Arrow operator
            r'==|!=|<=|>=|<<=|>>=|'  # Comparison and shift operators
            r'\+=|-=|\*=|/=|%=|&=|\|=|\^=|'  # Compound assignment
            r'&&|\|\||'  # Logical operators
            r'<<|>>|'  # Shift operators
            r'\+\+|--|'  # Increment/decrement
            r'\b\w+\b|'  # Words (keywords, identifiers, numbers)
            r'[+\-*/%&|^~]|'  # Arithmetic and bitwise operators
            r'[<>=!]|'  # Comparison operators (single)
            r'[(){}\[\];:,.]'  # Punctuation
        )
    
    def tokenize(self, code: str) -> List[str]:
        """
        Tokenize code into a list of tokens.
        
        Args:
            code: String containing code to tokenize
            
        Returns:
            List of tokens
        """
        # Remove extra whitespace and newlines
        code = ' '.join(code.split())
        
        # Find all tokens
        tokens = self.token_pattern.findall(code)
        
        # Filter empty tokens and strip whitespace
        tokens = [token.strip() for token in tokens if token.strip()]
        
        return tokens
    
    def tokenize_lines(self, code: str) -> List[List[str]]:
        """
        Tokenize code line by line.
        
        Args:
            code: String containing code to tokenize
            
        Returns:
            List of token lists, one per line
        """
        lines = code.split('\n')
        return [self.tokenize(line) for line in lines if line.strip()]