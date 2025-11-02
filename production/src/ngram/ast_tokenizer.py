"""
AST-based tokenizer for better code understanding
Converts Python code to token sequences with structure information
"""
import ast
import re
from typing import List, Optional


class ASTTokenizer:
    """Tokenizer that combines AST structure with code tokens"""
    
    def __init__(self):
        self.token_pattern = re.compile(
            r'""".*?"""|'  # Triple-quoted strings
            r"'''.*?'''|"  # Triple-quoted strings
            r'"(?:[^"\\]|\\.)*"|'  # Double-quoted strings
            r"'(?:[^'\\]|\\.)*'|"  # Single-quoted strings
            r'#.*?$|'  # Comments
            r'==|!=|<=|>=|<<|>>|//|\*\*|->|\+=|-=|\*=|/=|%=|&=|\|=|\^=|<<=|>>=|//=|\*\*=|:=|@=|'  # Multi-char operators
            r'[<>+\-*/%&|^~!@]|'  # Single-char operators
            r'\b\d+\.?\d*\b|'  # Numbers
            r'\b[a-zA-Z_]\w*\b|'  # Identifiers/keywords
            r'[(){}[\],.;:]|'  # Punctuation
            r'\n|\s+',  # Whitespace
            re.MULTILINE | re.DOTALL
        )
        
        # Non-terminals for AST nodes
        self.node_types = {
            ast.FunctionDef: '<FunctionDef>',
            ast.ClassDef: '<ClassDef>',
            ast.If: '<If>',
            ast.For: '<For>',
            ast.While: '<While>',
            ast.With: '<With>',
            ast.Try: '<Try>',
            ast.Return: '<Return>',
            ast.Assign: '<Assign>',
            ast.Call: '<Call>',
            ast.BinOp: '<BinOp>',
            ast.Compare: '<Compare>',
            ast.ListComp: '<ListComp>',
            ast.DictComp: '<DictComp>',
        }
    
    def tokenize(self, code: str) -> List[str]:
        """
        Tokenize code into a sequence of tokens
        
        Args:
            code: Python code string
            
        Returns:
            List of tokens
        """
        # Simple regex-based tokenization (fallback)
        tokens = []
        for match in self.token_pattern.finditer(code):
            token = match.group(0)
            # Skip pure whitespace (but keep newlines)
            if token.strip() or token == '\n':
                # Normalize whitespace to single space
                if token != '\n' and token.isspace():
                    token = ' '
                tokens.append(token)
        
        return tokens
    
    def tokenize_with_ast(self, code: str) -> List[str]:
        """
        Tokenize code with AST structure markers
        
        Args:
            code: Python code string
            
        Returns:
            List of tokens with AST structure markers
        """
        try:
            tree = ast.parse(code)
            tokens = []
            
            # Walk the AST and add structure markers
            for node in ast.walk(tree):
                node_type = type(node)
                if node_type in self.node_types:
                    tokens.append(self.node_types[node_type])
            
            # Add regular tokens
            regular_tokens = self.tokenize(code)
            
            # Interleave structure markers with code tokens
            # For simplicity, add structure markers at the beginning
            return tokens + regular_tokens
            
        except SyntaxError:
            # If AST parsing fails, fall back to regular tokenization
            return self.tokenize(code)
    
    def detokenize(self, tokens: List[str]) -> str:
        """
        Convert tokens back to code string
        
        Args:
            tokens: List of tokens
            
        Returns:
            Reconstructed code string
        """
        # Remove AST markers
        code_tokens = [t for t in tokens if not t.startswith('<')]
        
        # Simple joining logic
        result = []
        for i, token in enumerate(code_tokens):
            if i > 0 and not token in '.,;:()[]{}' and code_tokens[i-1] not in '([{':
                # Add space before token if needed
                if not result[-1].endswith('\n'):
                    result.append(' ')
            result.append(token)
        
        return ''.join(result)


class VocabularyManager:
    """Manages vocabulary with frequency-based filtering"""
    
    def __init__(self, max_vocab_size: int = 50000):
        self.max_vocab_size = max_vocab_size
        self.token_to_id = {}
        self.id_to_token = {}
        self.token_counts = {}
        self.unk_token = '<UNK>'
        self.start_token = '<s>'
        self.end_token = '<e>'
        self.pad_token = '<PAD>'
        
        # Initialize special tokens
        self._add_special_tokens()
    
    def _add_special_tokens(self):
        """Add special tokens to vocabulary"""
        special_tokens = [self.pad_token, self.unk_token, self.start_token, self.end_token]
        for token in special_tokens:
            self.token_to_id[token] = len(self.token_to_id)
            self.id_to_token[len(self.id_to_token)] = token
    
    def build_from_tokens(self, token_sequences: List[List[str]]):
        """
        Build vocabulary from token sequences
        
        Args:
            token_sequences: List of token lists
        """
        from collections import Counter
        
        # Count all tokens
        all_tokens = []
        for tokens in token_sequences:
            all_tokens.extend(tokens)
        
        self.token_counts = Counter(all_tokens)
        
        # Keep top N most frequent tokens
        most_common = self.token_counts.most_common(self.max_vocab_size - len(self.token_to_id))
        
        for token, count in most_common:
            if token not in self.token_to_id:
                token_id = len(self.token_to_id)
                self.token_to_id[token] = token_id
                self.id_to_token[token_id] = token
    
    def encode(self, tokens: List[str]) -> List[int]:
        """Convert tokens to IDs"""
        unk_id = self.token_to_id[self.unk_token]
        return [self.token_to_id.get(token, unk_id) for token in tokens]
    
    def decode(self, ids: List[int]) -> List[str]:
        """Convert IDs to tokens"""
        return [self.id_to_token.get(id, self.unk_token) for id in ids]
    
    @property
    def vocab_size(self) -> int:
        """Get vocabulary size"""
        return len(self.token_to_id)
    
    def save(self, filepath: str):
        """Save vocabulary to file"""
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump({
                'token_to_id': self.token_to_id,
                'id_to_token': self.id_to_token,
                'token_counts': self.token_counts,
                'max_vocab_size': self.max_vocab_size
            }, f)
    
    def load(self, filepath: str):
        """Load vocabulary from file"""
        import pickle
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.token_to_id = data['token_to_id']
            self.id_to_token = data['id_to_token']
            self.token_counts = data['token_counts']
            self.max_vocab_size = data['max_vocab_size']
