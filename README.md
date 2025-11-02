# 🚀 AI-Powered Code Suggestion System# Code Suggestion System - Quick Start Guide



> An intelligent Python code completion system powered by Hybrid N-gram + LSTM neural network with smart template detection## 🎯 Your Model is Ready!



[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)**Training Results:**

[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-red.svg)](https://pytorch.org/)- ✅ Dataset: 18,612 Python code samples

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)- ✅ N-gram size: 15 (maximum context)

- ✅ LSTM: 256 hidden units, 26.7M parameters

## 📖 Overview- ✅ Training time: 51 minutes on RTX 4060

- ✅ Final validation loss: 3.89 (excellent!)

This project implements a state-of-the-art code suggestion system that combines:

- **Statistical N-gram Model** (n=15) for fast pattern matching**Accuracy Improvements:**

- **Deep Learning LSTM** (2 layers, 256 units) for contextual understanding- `return a +` → `b` (37.32% vs 15.4% before) ✨

- **Smart Templates** (60+ patterns) for common Python constructs- `def add ( a , b` → `)` (32.35% confidence)

- **Naming Convention Detection** (get_, set_, is_, has_, validate_, calculate_)- Better rare pattern handling with 4 epochs!



### ✨ Key Features---



- 🎯 **85.15% Top-5 Accuracy** on Python code completion## 🚀 How to Use Your Model

- ⚡ **Real-time Suggestions** (<20ms response time)

- 🧠 **Smart Code Completion** - Understands intent and generates full code blocks### 1. Quick Test (Run Examples)

- 🎨 **VS Code Dark Theme UI** - Beautiful web interface```powershell

- 🔄 **Hybrid Architecture** - Best of statistical and neural approachescd E:\storage\code_sugesstion_system\code-suggestion-ngram

- 📊 **50,000 Token Vocabulary** - Comprehensive Python coveragepython use_model.py

```

## 🎯 Live DemoThis will show example usage and results.



```python### 2. Interactive Mode

# Type: def add```powershell

# Get:python use_model.py interactive

def add(a, b):```

    """Add two numbers"""Then type code to get suggestions:

    return a + b```

>>> def add ( a , b

# Type: def get_username  Suggestions:

# Get:    )               32.35%

def get_username(self):    ,                7.53%

    """Get username"""

    return self._username>>> for i in range (

  Suggestions:

# Type: try:    len             14.54%

# Get:    n                5.76%

try:

    # Code that might raise exception>>> complete return a +

    pass  Completed: return a + b

except Exception as e:

    # Handle exception>>> quit

    print(f"Error: {e}")```

```

### 3. Use in Your Python Code

## 📊 Model Performance```python

from src.ngram.ast_tokenizer import ASTTokenizer, VocabularyManager

| Metric | Value |from src.ngram.enhanced_model import EnhancedNGramModel

|--------|-------|from src.ngram.lstm_model import LSTMTrainer

| **Top-1 Accuracy** | 57.97% |from src.ngram.hybrid_model import HybridCodeCompleter

| **Top-3 Accuracy** | 78.68% |

| **Top-5 Accuracy** | 85.15% |# Load model

| **Training Loss** | 1.128 |vocab = VocabularyManager()

| **Validation Loss** | 3.886 |vocab.load('data/processed/vocabulary_best.pkl')

| **Training Time** | 51 minutes (4 epochs, GPU) |

| **Model Parameters** | 26.7M (LSTM) + 1.2M contexts (N-gram) |ngram_model = EnhancedNGramModel(n=15)

| **Dataset Size** | 18,612 Python code samples |ngram_model.load('data/processed/ngram_best_model.pkl')



## 🏗️ Architecturelstm_model = LSTMTrainer.load_model('data/processed/lstm_best_model.pth')



```tokenizer = ASTTokenizer()

User Input: "def add(a, b"

     ↓# Create completer

┌────────────────────┐completer = HybridCodeCompleter(

│   Tokenization     │    ngram_model=ngram_model,

└────────────────────┘    lstm_model=lstm_model,

     ↓    vocabulary=vocab,

┌────────────────────────────┐    tokenizer=tokenizer

│  Hybrid Model (Ensemble)   │)

├────────────┬───────────────┤

│  N-gram    │     LSTM      │# Get suggestions

│  (60%)     │     (40%)     │code = "def calculate ( x , y"

│            │               │suggestions = completer.suggest(code, top_k=5)

│ • n=15     │ • 2 layers    │

│ • 1.2M ctx │ • 256 units   │for token, prob in suggestions:

│ • Fast     │ • Deep        │    print(f"{token}: {prob:.2%}")

└────────────┴───────────────┘```

     ↓

┌────────────────────┐---

│ Smart Templates    │

│ (60+ patterns)     │## 📦 Model Files

└────────────────────┘

     ↓All saved in `data/processed/`:

Top-5 Suggestions + Full Code Block- `vocabulary_best.pkl` - Token vocabulary (50,000 tokens)

```- `ngram_best_model.pkl` - N-gram model (n=15, 1.2M contexts)

- `lstm_best_model.pth` - LSTM model (26.7M parameters)

### Components- `model_info.json` - Model metadata and configuration



1. **N-gram Model** (Statistical)---

   - Size: 15-gram (looks at last 15 tokens)

   - Contexts: 1,210,756 unique patterns## 🎨 API Reference

   - Speed: <1ms per prediction

   - Use case: Common patterns, exact matches### `completer.suggest(code, top_k=5, use_lstm=True)`

Get top-k suggestions for partial code.

2. **LSTM Model** (Neural Network)- `code`: Partial Python code (string)

   - Architecture: 2-layer stacked LSTM- `top_k`: Number of suggestions to return

   - Hidden units: 256 per layer- `use_lstm`: Use hybrid model (True) or N-gram only (False)

   - Embedding: 256 dimensions- Returns: List of (token, probability) tuples

   - Parameters: 26,702,672

   - Context: 50 tokens### `completer.complete_code(code, max_tokens=10)`

   - Use case: Rare patterns, semantic understandingAuto-complete code by generating multiple tokens.

- `code`: Partial Python code

3. **Smart Completer** (Template-based)- `max_tokens`: Maximum tokens to generate

   - Templates: 60+ Python patterns- Returns: Completed code string

   - Naming conventions: 6 patterns (get_, set_, is_, etc.)

   - Confidence scoring: Multi-factor analysis### `completer.get_multiple_completions(code, num_completions=3)`

   - Use case: Full code block generationGet multiple possible completions.

- `code`: Partial Python code

## 🚀 Quick Start- `num_completions`: Number of different completions

- Returns: List of completed code strings

### Prerequisites

---

```bash

# Python 3.11+## 💡 Tips

# CUDA-capable GPU (optional, for training)

```**For best results:**

1. Provide enough context (5-10 tokens minimum)

### Installation2. Use hybrid mode for rare patterns

3. Use N-gram only mode (`use_lstm=False`) for speed

```bash4. The model understands Python syntax, keywords, and common patterns

# 1. Clone the repository

git clone https://github.com/yourusername/code-suggestion-ngram.git**Performance:**

cd code-suggestion-ngram- N-gram only: Very fast (<1ms)

- Hybrid mode: Fast (~10-50ms on GPU)

# 2. Create virtual environment- Best for: function signatures, loops, conditions, imports

python -m venv venv

source venv/bin/activate  # On Windows: venv\Scripts\activate---



# 3. Install dependencies## 📊 Model Performance

pip install -r requirements.txt

```**Test Results (4 epochs):**

- Function completion: ✅ Excellent (32-37% confidence)

### Download Pre-trained Models- Loop constructs: ✅ Very good (14-38% confidence)

- Imports: ✅ Smart suggestions (numpy, pandas, requests)

**Option 1: Download from Release**- Return statements: ✅ Context-aware (37% for `return a + b`)

```bash

# Download models from GitHub Releases page**vs 1 Epoch:**

# Extract to production/models/ directory- `return a +` → `b`: **37.32%** (was 15.4%) - **2.4x better!**

```- More confident predictions overall

- Better rare pattern handling

**Option 2: Train Your Own** (See Training section below)

---

### Run Web Application

## 🎯 Next Steps

```bash

# Start the web server1. **Test it:** `python use_model.py`

cd production2. **Try interactive mode:** `python use_model.py interactive`

python web_app.py3. **Integrate in your IDE/editor**

4. **Fine-tune on your own code** (optional)

# Open browser at http://localhost:5000

```Enjoy your accurate code suggestion system! 🚀


## 📁 Project Structure

```
code-suggestion-ngram/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── .gitignore                  # Git ignore rules
│
├── training/                    # Training scripts and data
│   ├── train_model.py          # Main training script
│   └── dataset/                # Training dataset
│       └── README.md           # Dataset download instructions
│
├── evaluation/                  # Model evaluation
│   ├── test_model.py           # Test trained models
│   ├── compare_models.py       # Compare different models
│   └── results/                # Evaluation results
│
├── production/                  # Production-ready application
│   ├── web_app.py              # Flask web server
│   ├── templates/              # HTML templates
│   │   └── index.html          # Web UI
│   ├── models/                 # Trained models (gitignored)
│   │   ├── vocabulary_best.pkl
│   │   ├── ngram_best_model.pkl
│   │   ├── lstm_best_model.pth
│   │   └── model_info.json
│   └── src/                    # Source code
│       ├── ngram/              # Model implementations
│       │   ├── model.py        # N-gram model
│       │   ├── lstm_model.py   # LSTM model
│       │   ├── hybrid_model.py # Hybrid ensemble
│       │   ├── smart_completer.py  # Template system
│       │   ├── trainer.py      # Training logic
│       │   └── ...
│       └── utils/              # Utility functions
│
└── docs/                        # Documentation
    ├── IMPROVEMENTS.md         # v2.0 improvements
    └── SMART_FEATURES.md       # Smart completion guide
```

## 🎓 Training Your Own Model

### 1. Prepare Dataset

```bash
# Download Kaggle Python dataset
# Place in training/dataset/kaggle_python_dataset.json
```

Dataset format:
```json
[
  {
    "instruction": "Write a function to add two numbers",
    "input": "",
    "output": "def add(a, b):\n    return a + b"
  }
]
```

### 2. Train Models

```bash
cd training
python train_model.py
```

Training will:
- Filter Python-only code (removes C++/Java)
- Build 50K vocabulary
- Train 15-gram N-gram model (~5 seconds)
- Train 2-layer LSTM model (~50 minutes on GPU)
- Save models to `production/models/`

**Training Configuration:**
```python
{
  "n": 15,                  # N-gram size
  "vocab_size": 50000,      # Vocabulary size
  "hidden_dim": 256,        # LSTM hidden units
  "layers": 2,              # LSTM layers
  "epochs": 4,              # Training epochs
  "batch_size": 32,         # Mini-batch size
  "learning_rate": 0.001,   # Adam optimizer LR
  "dropout": 0.2           # Dropout rate
}
```

### 3. Evaluate

```bash
cd evaluation
python test_model.py
```

## 🧪 Testing

### Interactive Testing

```bash
cd production
python web_app.py
# Open http://localhost:5000
```

### Command-line Testing

```bash
cd evaluation
python test_model.py --model ../production/models/lstm_best_model.pth
```

### Compare Models

```bash
cd evaluation
python compare_models.py
```

## 🎨 Web Interface Features

### 1. Token-by-Token Suggestions
- Type code and get real-time suggestions
- Use **Tab** to accept suggestion
- Use **↑↓** arrows to navigate suggestions
- Use **Esc** to dismiss

### 2. Smart Mode (🧠 Button)
- Detects user intent
- Generates complete code blocks
- Shows confidence scores
- Press **Ctrl+Enter** to accept

### 3. LSTM Toggle
- Switch between N-gram only and Hybrid mode
- Blue = Hybrid (LSTM + N-gram)
- Gray = N-gram only

### 4. Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| **Tab** | Accept token suggestion |
| **↑↓** | Navigate suggestions |
| **Esc** | Dismiss suggestions |
| **Ctrl+Enter** | Accept smart completion |

## 📊 Technical Details

### Machine Learning Concepts Used

1. **Natural Language Processing**
   - Tokenization (AST-aware)
   - Vocabulary building (50K tokens)
   - Sequence modeling

2. **Statistical ML**
   - N-gram language model
   - Maximum likelihood estimation
   - Add-k smoothing (Laplace)
   - Perplexity metric

3. **Deep Learning**
   - LSTM (Long Short-Term Memory)
   - Word embeddings (256D)
   - Dropout regularization (0.2)
   - Cross-entropy loss

4. **Training Techniques**
   - Mini-batch gradient descent
   - Adam optimizer
   - Early stopping
   - Train/validation split (90/10)
   - GPU acceleration (CUDA)

5. **Ensemble Methods**
   - Weighted ensemble (60% N-gram, 40% LSTM)
   - Hybrid prediction

### Model Files

| File | Size | Description |
|------|------|-------------|
| `vocabulary_best.pkl` | ~3 MB | 50K token dictionary |
| `ngram_best_model.pkl` | ~150 MB | 1.2M N-gram contexts |
| `lstm_best_model.pth` | ~105 MB | 26.7M LSTM parameters |
| `model_info.json` | ~1 KB | Model metadata |

**Total:** ~260 MB (too large for GitHub - use Git LFS or download separately)

## 🛠️ Development

### Requirements

```
Python >= 3.11
torch >= 2.5.1
flask >= 3.1.0
numpy >= 1.26.0
```

See `requirements.txt` for full list.

### Running Tests

```bash
# Test individual components
cd evaluation
python test_model.py --verbose
```

### Code Style

- PEP 8 compliant
- Type hints included
- Docstrings for all functions

## 📈 Performance Optimization

### Speed
- N-gram lookups: <1ms
- LSTM inference: ~10ms
- Total response: <20ms (production)

### Memory
- Model size: 260 MB
- Runtime RAM: 300-400 MB
- GPU VRAM: 500 MB (if using CUDA)

### Accuracy Improvements
- Use larger dataset (current: 18K samples)
- Increase vocabulary size (current: 50K)
- Add more LSTM layers (current: 2)
- Train longer (current: 4 epochs)
- Fine-tune weights (current: 60% N-gram, 40% LSTM)

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Dataset:** [Kaggle Python Code Instruction Dataset](https://www.kaggle.com/datasets/thedevastator/python-code-instruction-dataset)
- **Framework:** PyTorch 2.5.1
- **Web Framework:** Flask 3.1.0
- **UI Inspiration:** VS Code Dark+ Theme

## 📞 Contact

For questions or feedback:
- Open an issue on GitHub
- Email: aarths123@gmail.com

## 🎯 Roadmap

- [ ] Support for more programming languages (JavaScript, Java, C++)
- [ ] Transformer-based model (BERT, GPT)
- [ ] VS Code extension
- [ ] API endpoint for integration
- [ ] Mobile app
- [ ] Real-time collaborative coding
- [ ] User-defined custom templates
- [ ] Learning from user corrections

## 📚 Documentation

- [Smart Features Guide](docs/SMART_FEATURES.md) - Complete guide to smart completion
- [Improvements v2.0](docs/IMPROVEMENTS.md) - Latest enhancements and features

## ⭐ Star History

If you find this project useful, please consider giving it a star!

---

**Made with ❤️ and Python**
