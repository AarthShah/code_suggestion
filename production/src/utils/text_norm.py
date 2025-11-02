from collections import defaultdict
import re

def preprocess_code(code_snippets):
    # Simple preprocessing: remove comments and normalize whitespace
    cleaned_snippets = []
    for snippet in code_snippets:
        # Remove comments (this is a simple regex and may not cover all cases)
        snippet = re.sub(r'#.*', '', snippet)
        cleaned_snippets.append(' '.join(snippet.split()))
    return cleaned_snippets

def generate_bigrams(snippets):
    bigram_model = defaultdict(int)
    for snippet in snippets:
        tokens = snippet.split()
        for i in range(len(tokens) - 1):
            bigram = (tokens[i], tokens[i + 1])
            bigram_model[bigram] += 1
    return bigram_model

# Example usage
code_snippets = [
    "for i in range(10):",
    "print(i)",
    "if i % 2 == 0:",
    "print('Even')",
    "else:",
    "print('Odd')"
]

cleaned_snippets = preprocess_code(code_snippets)
bigram_model = generate_bigrams(cleaned_snippets)

# Display the bigram frequencies
for bigram, freq in bigram_model.items():
    print(f"{bigram}: {freq}")