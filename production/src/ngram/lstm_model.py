"""
Small LSTM model for code completion
Handles long-range dependencies that n-grams miss
"""
import torch
import torch.nn as nn
from typing import List, Tuple
import numpy as np
from tqdm import tqdm


class CodeLSTM(nn.Module):
    """Small LSTM model for code completion"""
    
    def __init__(self, vocab_size: int, embedding_dim: int = 256, hidden_dim: int = 256, num_layers: int = 2):
        """
        Initialize LSTM model
        
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of token embeddings
            hidden_dim: Dimension of LSTM hidden state (128-256 recommended)
            num_layers: Number of LSTM layers
        """
        super(CodeLSTM, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # LSTM layer
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        
        # Output layer
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x, hidden=None):
        """
        Forward pass
        
        Args:
            x: Input tensor (batch_size, seq_len)
            hidden: Hidden state tuple (h, c)
            
        Returns:
            Output logits and hidden state
        """
        # Embedding
        embedded = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        embedded = self.dropout(embedded)
        
        # LSTM
        if hidden is not None:
            output, hidden = self.lstm(embedded, hidden)
        else:
            output, hidden = self.lstm(embedded)
        
        # Output layer
        output = self.dropout(output)
        logits = self.fc(output)  # (batch_size, seq_len, vocab_size)
        
        return logits, hidden
    
    def predict(self, context_ids: List[int], top_k: int = 5) -> List[Tuple[int, float]]:
        """
        Predict next token given context
        
        Args:
            context_ids: List of token IDs
            top_k: Number of predictions to return
            
        Returns:
            List of (token_id, probability) tuples
        """
        self.eval()
        with torch.no_grad():
            # Ensure tensor is on same device as model
            try:
                model_device = next(self.parameters()).device
            except StopIteration:
                model_device = torch.device('cpu')

            # Convert to tensor on correct device
            x = torch.tensor([context_ids], dtype=torch.long, device=model_device)

            # Forward pass
            logits, _ = self(x)

            # Get predictions for last position
            last_logits = logits[0, -1, :]  # (vocab_size,)

            # Apply softmax
            probs = torch.softmax(last_logits, dim=0)

            # Get top-k
            top_probs, top_indices = torch.topk(probs, top_k)

            # Convert to list
            results = []
            for idx, prob in zip(top_indices.tolist(), top_probs.tolist()):
                results.append((idx, prob))

            return results


class LSTMTrainer:
    """Trainer for LSTM model"""
    
    def __init__(self, model: CodeLSTM, learning_rate: float = 0.001, device = None):
        """
        Initialize trainer
        
        Args:
            model: CodeLSTM model
            learning_rate: Learning rate
            device: Device to train on ('cuda' or 'cpu')
        """
        self.model = model
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        print(f"🔥 Using device: {self.device.upper()}")
        if self.device == 'cuda':
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    def prepare_sequences(self, token_ids: List[List[int]], seq_length: int = 50):
        """
        Prepare training sequences
        
        Args:
            token_ids: List of token ID sequences
            seq_length: Length of each training sequence
            
        Returns:
            X (inputs) and Y (targets)
        """
        X, Y = [], []
        
        for sequence in token_ids:
            if len(sequence) < seq_length + 1:
                continue
            
            for i in range(len(sequence) - seq_length):
                X.append(sequence[i:i+seq_length])
                Y.append(sequence[i+1:i+seq_length+1])
        
        return np.array(X), np.array(Y)
    
    def train_epoch(self, X, Y, batch_size: int = 32):
        """
        Train for one epoch
        
        Args:
            X: Input sequences
            Y: Target sequences
            batch_size: Batch size
            
        Returns:
            Average loss
        """
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        # Shuffle data
        indices = np.random.permutation(len(X))
        X = X[indices]
        Y = Y[indices]
        
        # Use larger batch size for GPU
        if self.device == 'cuda':
            batch_size = min(batch_size * 4, 256)  # 4x larger batches on GPU
        
        # Training loop with progress bar
        num_iterations = (len(X) + batch_size - 1) // batch_size
        
        for i in tqdm(range(0, len(X), batch_size), desc="  Training", total=num_iterations):
            batch_X = X[i:i+batch_size]
            batch_Y = Y[i:i+batch_size]
            
            # Convert to tensors
            batch_X = torch.tensor(batch_X, dtype=torch.long, device=self.device)
            batch_Y = torch.tensor(batch_Y, dtype=torch.long, device=self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            logits, _ = self.model(batch_X)
            
            # Calculate loss
            loss = self.criterion(logits.view(-1, self.model.vocab_size), batch_Y.view(-1))
            
            # Backward pass
            loss.backward()
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            # Update weights
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0
    
    def train(self, token_ids: List[List[int]], epochs: int = 5, batch_size: int = 32, 
              seq_length: int = 50, validation_split: float = 0.1):
        """
        Train model
        
        Args:
            token_ids: List of token ID sequences
            epochs: Number of epochs (3-5 recommended)
            batch_size: Batch size
            seq_length: Sequence length
            validation_split: Validation data split
            
        Returns:
            Training history
        """
        print(f"Preparing training data (seq_length={seq_length})...")
        X, Y = self.prepare_sequences(token_ids, seq_length)
        
        print(f"Total training sequences: {len(X)}")
        
        # Split into train and validation
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        Y_train, Y_val = Y[:split_idx], Y[split_idx:]
        
        print(f"Train: {len(X_train)}, Validation: {len(X_val)}")
        
        history = {'train_loss': [], 'val_loss': []}
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            
            # Train
            train_loss = self.train_epoch(X_train, Y_train, batch_size)
            history['train_loss'].append(train_loss)
            
            # Validate
            val_loss = self.validate(X_val, Y_val, batch_size)
            history['val_loss'].append(val_loss)
            
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
        
        return history
    
    def validate(self, X, Y, batch_size: int = 32):
        """
        Validate model
        
        Args:
            X: Input sequences
            Y: Target sequences
            batch_size: Batch size
            
        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        # Use larger batch size for GPU
        if self.device == 'cuda':
            batch_size = min(batch_size * 4, 256)
        
        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                batch_X = X[i:i+batch_size]
                batch_Y = Y[i:i+batch_size]
                
                # Convert to tensors
                batch_X = torch.tensor(batch_X, dtype=torch.long, device=self.device)
                batch_Y = torch.tensor(batch_Y, dtype=torch.long, device=self.device)
                
                # Forward pass
                logits, _ = self.model(batch_X)
                
                # Calculate loss
                loss = self.criterion(logits.view(-1, self.model.vocab_size), batch_Y.view(-1))
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0
    
    def save_model(self, filepath: str):
        """Save model to file"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'vocab_size': self.model.vocab_size,
            'embedding_dim': self.model.embedding_dim,
            'hidden_dim': self.model.hidden_dim,
            'num_layers': self.model.num_layers
        }, filepath)
    
    @staticmethod
    def load_model(filepath: str, device = None):
        """Load model from file"""
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        checkpoint = torch.load(filepath, map_location=device)
        
        model = CodeLSTM(
            vocab_size=checkpoint['vocab_size'],
            embedding_dim=checkpoint['embedding_dim'],
            hidden_dim=checkpoint['hidden_dim'],
            num_layers=checkpoint['num_layers']
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        
        return model
