"""
Flask web application for code suggestions
Provides real-time code completion with VS Code-like interface
"""
from flask import Flask, render_template, request, jsonify
import os
import torch
from src.ngram.ast_tokenizer import ASTTokenizer, VocabularyManager
from src.ngram.enhanced_model import EnhancedNGramModel
from src.ngram.lstm_model import LSTMTrainer
from src.ngram.hybrid_model import HybridCodeCompleter
from src.ngram.smart_completer import SmartCodeCompleter

app = Flask(__name__)

# Global model instances
completer = None
smart_completer = None
vocab = None
ngram_model = None
lstm_model = None
tokenizer = None

# Model configurations
MODELS = {
    'best': {
        'vocab': 'data/processed/vocabulary_best.pkl',
        'ngram': 'data/processed/ngram_best_model.pkl',
        'lstm': 'data/processed/lstm_best_model.pth',
        'description': 'Best Model (50K tokens, 1.2M contexts, LSTM 26.7M params)'
    }
}

current_model = 'best'


def load_model(model_name='best'):
    """Load the specified model"""
    global completer, smart_completer, vocab, ngram_model, lstm_model, tokenizer
    
    print(f"Loading model: {model_name}...")
    config = MODELS[model_name]
    
    # Load vocabulary
    vocab = VocabularyManager()
    vocab.load(config['vocab'])
    print(f"✓ Vocabulary loaded: {vocab.vocab_size:,} tokens")
    
    # Load N-gram model
    ngram_model = EnhancedNGramModel(n=15)
    ngram_model.load(config['ngram'])
    print(f"✓ N-gram model loaded (n={ngram_model.n})")
    
    # Load LSTM model
    lstm_model = LSTMTrainer.load_model(config['lstm'])
    print(f"✓ LSTM model loaded")
    
    # Create tokenizer
    tokenizer = ASTTokenizer()
    
    # Create hybrid completer
    completer = HybridCodeCompleter(
        ngram_model=ngram_model,
        lstm_model=lstm_model,
        vocabulary=vocab,
        tokenizer=tokenizer,
        ngram_weight=0.6,
        lstm_weight=0.4
    )
    
    # Create smart completer
    smart_completer = SmartCodeCompleter(completer)
    
    print(f"✓ Model '{model_name}' ready!\n")


@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html', models=MODELS, current_model=current_model)


@app.route('/api/suggest', methods=['POST'])
def suggest():
    """Get code suggestions"""
    try:
        data = request.json
        code = data.get('code', '')
        top_k = data.get('top_k', 5)
        use_lstm = data.get('use_lstm', True)
        
        if not code:
            return jsonify({'suggestions': []})
        
        # Get suggestions
        suggestions = completer.suggest(code, top_k=top_k, use_lstm=use_lstm)
        
        # Format response
        result = [
            {
                'token': token,
                'probability': prob,
                'percentage': f"{prob * 100:.2f}%"
            }
            for token, prob in suggestions
        ]
        
        return jsonify({'suggestions': result})
    
    except Exception as e:
        print(f"Error in suggest: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/smart_suggest', methods=['POST'])
def smart_suggest():
    """Get smart code suggestions with full code completion"""
    try:
        data = request.json
        code = data.get('code', '')
        
        if not code:
            return jsonify({'suggestions': [], 'full_completion': None})
        
        # Get smart completion
        result = smart_completer.get_smart_completion(code)
        
        # Format token suggestions
        formatted_suggestions = [
            {
                'token': token,
                'probability': prob,
                'percentage': f"{prob * 100:.2f}%"
            }
            for token, prob in result['suggestions']
        ]
        
        return jsonify({
            'suggestions': formatted_suggestions,
            'full_completion': result['full_completion'],
            'intent': result['intent'],
            'confidence': result['confidence'],
            'description': result['description']
        })
    
    except Exception as e:
        print(f"Error in smart_suggest: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/complete', methods=['POST'])
def complete():
    """Auto-complete code"""
    try:
        data = request.json
        code = data.get('code', '')
        max_tokens = data.get('max_tokens', 10)
        use_lstm = data.get('use_lstm', True)
        
        if not code:
            return jsonify({'completed': code})
        
        # Complete code
        completed = completer.complete_code(code, max_tokens=max_tokens, use_lstm=use_lstm)
        
        return jsonify({'completed': completed})
    
    except Exception as e:
        print(f"Error in complete: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/model/info', methods=['GET'])
def model_info():
    """Get current model information"""
    try:
        info = {
            'current_model': current_model,
            'description': MODELS[current_model]['description'],
            'vocab_size': vocab.vocab_size if vocab else 0,
            'ngram_n': ngram_model.n if ngram_model else 0,
            'ngram_contexts': len(ngram_model.ngrams) if ngram_model else 0,
            'available_models': list(MODELS.keys())
        }
        return jsonify(info)
    
    except Exception as e:
        print(f"Error in model_info: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/model/switch', methods=['POST'])
def switch_model():
    """Switch to a different model"""
    global current_model
    
    try:
        data = request.json
        model_name = data.get('model', 'best')
        
        if model_name not in MODELS:
            return jsonify({'error': f'Model "{model_name}" not found'}), 404
        
        load_model(model_name)
        current_model = model_name
        
        return jsonify({
            'success': True,
            'current_model': current_model,
            'description': MODELS[current_model]['description']
        })
    
    except Exception as e:
        print(f"Error in switch_model: {e}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    # Load default model on startup
    load_model('best')
    
    print("\n" + "="*70)
    print("🚀 Code Suggestion Web App")
    print("="*70)
    print("\n✓ Server starting...")
    print("✓ Open your browser and go to: http://localhost:5000")
    print("\nPress Ctrl+C to stop\n")
    
    # Run Flask app
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)
