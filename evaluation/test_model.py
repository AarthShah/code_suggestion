"""
Test the BEST trained model
"""
from src.ngram.ast_tokenizer import ASTTokenizer, VocabularyManager
from src.ngram.enhanced_model import EnhancedNGramModel
from src.ngram.lstm_model import LSTMTrainer
from src.ngram.hybrid_model import HybridCodeCompleter

print("="*70)
print("🏆 Testing BEST Model")
print("="*70)

# Load vocabulary
print("\nLoading vocabulary...")
vocab = VocabularyManager()
vocab.load('data/processed/vocabulary_best.pkl')
print(f"✓ Vocabulary: {vocab.vocab_size:,} tokens")

# Load N-gram model
print("\nLoading N-gram model (n=15)...")
ngram_model = EnhancedNGramModel(n=15)
ngram_model.load('data/processed/ngram_best_model.pkl')
stats = ngram_model.stats
print(f"✓ N-gram model loaded:")
print(f"  N: {stats['n']}")
print(f"  Contexts: {stats['unique_contexts']:,}")
print(f"  N-grams: {stats['total_ngrams']:,}")

# Load LSTM model
print("\nLoading LSTM model (256 hidden units)...")
lstm_model = LSTMTrainer.load_model('data/processed/lstm_best_model.pth')
print(f"✓ LSTM model loaded (256 hidden units, 2 layers)")

# Create tokenizer and hybrid
tokenizer = ASTTokenizer()
hybrid = HybridCodeCompleter(
    ngram_model=ngram_model,
    lstm_model=lstm_model,
    vocabulary=vocab,
    tokenizer=tokenizer,
    ngram_weight=0.6,
    lstm_weight=0.4
)

print("\n" + "="*70)
print("🧪 Comprehensive Test Suite")
print("="*70)

test_cases = [
    ("def add ( a", "Function parameter (old problematic case)"),
    ("def add ( a , b", "Function parameters"),
    ("def add ( a , b )", "After parameters"),
    ("def add ( a , b ) :", "Function body"),
    
    ("for i in range", "For loop start"),
    ("for i in range (", "Range parameter"),
    ("for i in range ( n", "Range closing"),
    ("for i in range ( n ) :", "Loop body"),
    
    ("if x ==", "Comparison"),
    ("if x == 0", "After condition"),
    ("if x == 0 :", "Condition body"),
    
    ("while True :", "Infinite loop"),
    ("while i <", "While condition"),
    
    ("try :", "Try block"),
    ("except", "Exception handler"),
    
    ("class MyClass", "Class definition"),
    ("class MyClass :", "Class body"),
    
    ("import", "Import statement"),
    ("from", "From import"),
    
    ("return a +", "Return expression"),
    ("return", "Return statement"),
    
    ("arr [ i", "Array indexing"),
    ("arr [ i ]", "After index"),
    
    ("print (", "Print function"),
    ("len (", "Length function"),
    
    ("x = [ i for", "List comprehension"),
    ("lambda x :", "Lambda function"),
]

for test_code, description in test_cases:
    print(f"\n{description}")
    print(f"  Input: '{test_code}'")
    
    # Get suggestions (hybrid)
    suggestions = hybrid.suggest(test_code, top_k=5, use_lstm=True)
    
    if suggestions:
        print("  Suggestions:")
        for i, (token, prob) in enumerate(suggestions, 1):
            bar = '█' * int(prob * 25)
            print(f"    {i}. {token:15s} {bar:25s} {prob:.1%}")
    else:
        print("  (No suggestions)")

print("\n" + "="*70)
print("💬 Interactive Mode")
print("="*70)
print("\nCommands:")
print("  Type code to get suggestions")
print("  'n' = N-gram only mode")
print("  'h' = Hybrid mode (default)")
print("  'c' = Complete code (auto-complete)")
print("  'quit' = Exit")
print()

mode = "hybrid"

while True:
    try:
        user_input = input(f"[{mode}] >>> ").strip()
        
        if not user_input:
            continue
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            break
        
        if user_input.lower() == 'n':
            mode = "ngram"
            print("  → N-gram only mode")
            continue
        
        if user_input.lower() == 'h':
            mode = "hybrid"
            print("  → Hybrid mode")
            continue
        
        if user_input.lower() == 'c':
            print("  Enter code to complete:")
            code = input("  >>> ").strip()
            if code:
                completion = hybrid.complete_code(code, max_tokens=10, use_lstm=(mode=="hybrid"))
                print(f"\n  Completed: {completion}\n")
            continue
        
        # Get suggestions
        use_lstm = (mode == "hybrid")
        suggestions = hybrid.suggest(user_input, top_k=5, use_lstm=use_lstm)
        
        if suggestions:
            print("\n  Suggestions:")
            for token, prob in suggestions:
                print(f"    {token:15s} {prob:.1%}")
            
            # Auto-complete
            completion = hybrid.complete_code(user_input, max_tokens=5, use_lstm=use_lstm)
            print(f"\n  Auto-complete: {completion}")
        else:
            print("  (No suggestions - try more context)")
        print()
    
    except KeyboardInterrupt:
        break
    except Exception as e:
        print(f"  Error: {e}")

print("\n✅ Testing complete!")
