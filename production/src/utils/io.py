from collections import defaultdict
import random

class NGramModel:
    def __init__(self, n):
        self.n = n
        self.ngrams = defaultdict(list)

    def train(self, tokens):
        for i in range(len(tokens) - self.n):
            ngram = tuple(tokens[i:i + self.n])
            next_token = tokens[i + self.n]
            self.ngrams[ngram].append(next_token)

    def suggest(self, context):
        context_tuple = tuple(context[-(self.n - 1):])
        return self.ngrams.get(context_tuple, [])

# Example usage
code_snippets = [
    "def my_function(param1, param2):",
    "    return param1 + param2",
    "if condition:",
    "    my_function(1, 2)"
]

# Tokenization (simple split for demonstration)
tokens = [token for snippet in code_snippets for token in snippet.split()]

# Create and train the n-gram model
ngram_model = NGramModel(n=2)
ngram_model.train(tokens)

# Get suggestions based on the last token
context = ["my_function", "1,"]
suggestions = ngram_model.suggest(context)
print("Suggestions:", suggestions)