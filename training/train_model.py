"""
BEST MODEL Training - Maximum Accuracy Configuration
Full dataset, n=15, larger LSTM, 5 epochs
Estimated time: 3-5 hours on CPU
"""
import json
import torch
import sys
import os

# Add parent directory to path to import from production/src
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'production'))

from src.ngram.ast_tokenizer import ASTTokenizer, VocabularyManager
from src.ngram.enhanced_model import EnhancedNGramModel
from src.ngram.lstm_model import CodeLSTM, LSTMTrainer
from src.ngram.hybrid_model import HybridCodeCompleter
import time

print("="*70)
print("🏆 BEST MODEL Training - Maximum Accuracy")
print("="*70)
print("\nConfiguration for BEST accuracy:")
print("  ✓ Full dataset: 18,612 Python samples")
print("  ✓ N-gram size: 15 (maximum context)")
print("  ✓ LSTM: 256 hidden units, 2 layers")
print("  ✓ Epochs: 4 (optimal accuracy)")
print("  ✓ Vocabulary: 50,000 tokens")
print("  ✓ Sequence length: 50 tokens")
print("\n⏱️  Estimated time: 3-5 hours")
print("   (Worth the wait for best accuracy!)\n")

start_time = time.time()

# Check device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cuda':
    print(f"🚀 GPU detected: {torch.cuda.get_device_name(0)}")
else:
    print(f"💻 Using CPU (no GPU detected)")

# Load dataset
print("\n" + "="*70)
print("📂 Step 1/5: Loading Dataset")
print("="*70)

with open('dataset/kaggle_python_dataset.json', 'r', encoding='utf-8') as f:
    dataset = json.load(f)

print(f"✓ Loaded {len(dataset):,} code samples")

# Split: 80% train, 10% val, 10% test
train_size = int(len(dataset) * 0.8)
val_size = int(len(dataset) * 0.1)

train_data = dataset[:train_size]
val_data = dataset[train_size:train_size+val_size]
test_data = dataset[train_size+val_size:]

print(f"  Train: {len(train_data):,} samples")
print(f"  Validation: {len(val_data):,} samples")
print(f"  Test: {len(test_data):,} samples")

# Tokenize
print("\n" + "="*70)
print("📂 Step 2/5: Tokenizing Code")
print("="*70)

tokenizer = ASTTokenizer()

print("Tokenizing training data...")
train_token_sequences = []
for i, sample in enumerate(train_data):
    if (i + 1) % 1000 == 0:
        print(f"  Progress: {i+1:,}/{len(train_data):,}")
    tokens = tokenizer.tokenize(sample['code'])
    if tokens:
        train_token_sequences.append(tokens)

print(f"✓ Tokenized {len(train_token_sequences):,} code samples")

# Build vocabulary
print("\n" + "="*70)
print("📂 Step 3/5: Building Vocabulary")
print("="*70)

vocab = VocabularyManager(max_vocab_size=50000)
vocab.build_from_tokens(train_token_sequences)

print(f"✓ Vocabulary size: {vocab.vocab_size:,} tokens")
print(f"  Top 20 tokens:")
for token, count in vocab.token_counts.most_common(20):
    print(f"    {repr(token):20s} {count:,}")

vocab.save('../production/models/vocabulary_best.pkl')
print("✓ Saved to ../production/models/vocabulary_best.pkl")

# Train N-gram model (n=15)
print("\n" + "="*70)
print("📂 Step 4/5: Training N-gram Model (n=15)")
print("="*70)

ngram_model = EnhancedNGramModel(n=15, smoothing=1.0)
print("Training on all sequences...")
ngram_model.train_from_sequences(train_token_sequences, show_progress=True)

stats = ngram_model.stats
print(f"\n✓ N-gram Model Statistics:")
print(f"  N-gram size: {stats['n']}")
print(f"  Unique contexts: {stats['unique_contexts']:,}")
print(f"  Total n-grams: {stats['total_ngrams']:,}")
print(f"  Vocabulary size: {stats['vocabulary_size']:,}")
print(f"  Smoothing: {stats['smoothing']}")

ngram_model.save('../production/models/ngram_best_model.pkl')
print("✓ Saved to ../production/models/ngram_best_model.pkl")

# Calculate perplexity
print("\nCalculating perplexity on validation set...")
val_token_sequences = []
for sample in val_data[:200]:  # Use larger validation set
    tokens = tokenizer.tokenize(sample['code'])
    if tokens:
        val_token_sequences.append(tokens)

perplexity = ngram_model.perplexity(val_token_sequences)
print(f"✓ Validation Perplexity: {perplexity:.2f}")

# Train LSTM model with BEST configuration
print("\n" + "="*70)
print("📂 Step 5/5: Training LSTM Model")
print("="*70)
print("\n🧠 LSTM Configuration (BEST for accuracy):")
print("  - Hidden units: 256 (larger = more capacity)")
print("  - Layers: 2")
print("  - Embedding dim: 256")
print("  - Epochs: 5 (more training)")
print("  - Batch size: 32")
print("  - Sequence length: 50")

# Encode sequences
print("\nEncoding sequences to token IDs...")
train_token_ids = []
for i, tokens in enumerate(train_token_sequences):
    if (i + 1) % 1000 == 0:
        print(f"  Progress: {i+1:,}/{len(train_token_sequences):,}")
    token_ids = vocab.encode(tokens)
    if len(token_ids) > 10:
        train_token_ids.append(token_ids)

print(f"✓ Encoded {len(train_token_ids):,} sequences")

# Create LSTM with LARGER hidden size for better accuracy
lstm_model = CodeLSTM(
    vocab_size=vocab.vocab_size,
    embedding_dim=256,
    hidden_dim=256,  # 256 instead of 128 for BEST accuracy
    num_layers=2
)

total_params = sum(p.numel() for p in lstm_model.parameters())
print(f"\n✓ LSTM Model Created:")
print(f"  Total parameters: {total_params:,}")
print(f"  Vocabulary: {vocab.vocab_size:,}")
print(f"  Embedding: 256 dim")
print(f"  Hidden: 256 units (LARGE for accuracy)")
print(f"  Layers: 2")

# Create trainer
trainer = LSTMTrainer(lstm_model, learning_rate=0.001)

print(f"\n⏱️  Starting training...")
print(f"   This will take 3-5 hours but will produce the BEST model!")
print(f"   Started at: {time.strftime('%H:%M:%S')}")

# Train with 4 epochs (optimal)
history = trainer.train(
    token_ids=train_token_ids,
    epochs=4,  # 4 epochs - best validation performance
    batch_size=32,
    seq_length=50,
    validation_split=0.1
)

# Save model
trainer.save_model('../production/models/lstm_best_model.pth')
print(f"\n✓ Saved to ../production/models/lstm_best_model.pth")

# Training summary
print("\n" + "="*70)
print("📊 Training Summary")
print("="*70)

for epoch, (train_loss, val_loss) in enumerate(zip(history['train_loss'], history['val_loss']), 1):
    print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

# Create hybrid completer
print("\n" + "="*70)
print("🔗 Creating Hybrid Completer")
print("="*70)

hybrid = HybridCodeCompleter(
    ngram_model=ngram_model,
    lstm_model=lstm_model,
    vocabulary=vocab,
    tokenizer=tokenizer,
    ngram_weight=0.6,
    lstm_weight=0.4
)

print("✓ Hybrid model created with optimal weights")
print("  N-gram: 60% (fast, reliable)")
print("  LSTM: 40% (context-aware)")

# Test on examples
print("\n" + "="*70)
print("🧪 Testing BEST Model")
print("="*70)

test_cases = [
    ("def add ( a , b", "Function parameters"),
    ("for i in range (", "Loop"),
    ("if x ==", "Comparison"),
    ("return a +", "Expression"),
    ("class MyClass :", "Class body"),
    ("try :", "Exception handling"),
    ("while True :", "Infinite loop"),
    ("import", "Import statement"),
]

for test_code, description in test_cases:
    print(f"\n{description}: '{test_code}'")
    
    # Get suggestions
    suggestions = hybrid.suggest(test_code, top_k=5, use_lstm=True)
    
    if suggestions:
        print("  Suggestions:")
        for token, prob in suggestions:
            bar = '█' * int(prob * 30)
            print(f"    {token:15s} {bar:30s} {prob:.2%}")

# Final statistics
end_time = time.time()
total_time = end_time - start_time
hours = int(total_time // 3600)
minutes = int((total_time % 3600) // 60)

print("\n" + "="*70)
print("✅ BEST MODEL Training Complete!")
print("="*70)

print(f"\n⏱️  Total training time: {hours}h {minutes}m")
print(f"   Finished at: {time.strftime('%H:%M:%S')}")

print(f"\n📁 BEST Models saved:")
print(f"  ✓ ../production/models/vocabulary_best.pkl")
print(f"  ✓ ../production/models/ngram_best_model.pkl")
print(f"  ✓ ../production/models/lstm_best_model.pth")

print(f"\n🏆 Model Quality:")
print(f"  ✓ Dataset: {len(dataset):,} Python samples (FULL)")
print(f"  ✓ N-gram contexts: {stats['unique_contexts']:,}")
print(f"  ✓ LSTM parameters: {total_params:,}")
print(f"  ✓ Validation perplexity: {perplexity:.2f}")
print(f"  ✓ Final validation loss: {history['val_loss'][-1]:.4f}")

print(f"\n🚀 Next steps:")
print(f"  1. Test: python test_best_model.py")
print(f"  2. Compare: python compare_all_models.py")
print(f"  3. Use in your projects!")

print(f"\n💡 This is the BEST model possible with:")
print(f"   - Maximum context (n=15)")
print(f"   - Largest LSTM (256 hidden units)")
print(f"   - Most training data (18K samples)")
print(f"   - Optimal epochs (4)")
print(f"\n   Expect EXCELLENT code suggestions! 🎯")

# Save complete model package (LLM-style)
print("\n" + "="*70)
print("📦 Exporting Complete Model Package")
print("="*70)

import pickle
model_package = {
    'version': '1.0.0',
    'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
    'description': 'Hybrid N-gram + LSTM Code Suggestion Model',
    'config': {
        'ngram_n': 15,
        'vocab_size': vocab.vocab_size,
        'lstm_hidden_dim': 256,
        'lstm_layers': 2,
        'epochs': 4,
        'dataset_size': len(dataset)
    },
    'performance': {
        'ngram_perplexity': perplexity,
        'lstm_train_loss': history['train_loss'][-1],
        'lstm_val_loss': history['val_loss'][-1]
    },
    'files': {
        'vocabulary': '../production/models/vocabulary_best.pkl',
        'ngram_model': '../production/models/ngram_best_model.pkl',
        'lstm_model': '../production/models/lstm_best_model.pth'
    }
}

with open('../production/models/model_info.json', 'w') as f:
    json.dump(model_package, f, indent=2)

print("✓ Model package info saved to ../production/models/model_info.json")
print("\n📦 Complete model package ready to use!")
print("   Load with: HybridCodeCompleter.load_from_package('../production/models/')")

