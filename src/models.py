"""
LSTM models for temporal prediction with network features.

Implements:
1. Standard LSTM with 2 layers
2. Attention-LSTM with temporal attention mechanism
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class StandardLSTM(nn.Module):
    """
    Standard 2-layer LSTM for temporal sequence prediction.

    Architecture:
    - Input: [batch, seq_len, input_dim]
    - LSTM: 2 layers with dropout
    - Output: [batch, 1] logits (no sigmoid)
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Prediction head
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, input_dim]

        Returns:
            logits: [batch, 1]
        """
        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Use last hidden state
        last_hidden = h_n[-1]  # [batch, hidden_dim]

        # Predict
        logits = self.fc(last_hidden)

        return logits


class AttentionLSTM(nn.Module):
    """
    LSTM with attention mechanism over temporal dimension.

    Architecture:
    - Input: [batch, seq_len, input_dim]
    - LSTM: 2 layers with dropout
    - Attention: Learn which timesteps matter most
    - Output: [batch, 1] logits (no sigmoid)
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

        # Prediction head
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, input_dim]

        Returns:
            logits: [batch, 1]
        """
        # LSTM
        lstm_out, _ = self.lstm(x)  # [batch, seq_len, hidden_dim]

        # Compute attention scores
        attention_scores = self.attention(lstm_out)  # [batch, seq_len, 1]
        attention_weights = F.softmax(attention_scores, dim=1)  # [batch, seq_len, 1]

        # Apply attention
        context = torch.sum(attention_weights * lstm_out, dim=1)  # [batch, hidden_dim]

        # Predict
        logits = self.fc(context)

        return logits


def train_lstm(
    model: nn.Module,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    num_epochs: int = 50,
    batch_size: int = 64,
    lr: float = 0.001,
    pos_weight: float = 1.0,
    device: str = 'cpu',
    verbose: bool = True
):
    """
    Train LSTM model with best practices.

    Args:
        model: LSTM model
        X_train, y_train: Training data
        X_test, y_test: Test data
        num_epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
        pos_weight: Weight for positive class (handles imbalance)
        device: 'cpu' or 'cuda'
        verbose: Print training progress

    Returns:
        Dictionary with training history and best model state
    """
    model = model.to(device)

    # Loss function with class weighting
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(device))

    # Optimizer with weight decay (L2 regularization)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    # Convert to tensors
    X_train = torch.FloatTensor(X_train).to(device)
    y_train = torch.FloatTensor(y_train).unsqueeze(1).to(device)
    X_test = torch.FloatTensor(X_test).to(device)
    y_test = torch.FloatTensor(y_test).unsqueeze(1).to(device)

    # Training history
    history = {
        'train_loss': [],
        'test_loss': [],
        'test_auprc': [],
        'test_auroc': []
    }

    best_test_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    patience = 10

    if verbose:
        print("\n" + "="*80)
        print("Training LSTM")
        print("="*80)
        print(f"Training batches: {len(X_train) // batch_size + 1}")
        print(f"Test batches: {len(X_test) // batch_size + 1}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"Input dim: {X_train.shape[2]}, Hidden dim: {model.hidden_dim}, Layers: {model.num_layers}")
        print(f"Positive class weight: {pos_weight:.3f} (balances {y_train.mean()*100:.1f}% positive rate)")
        print("\n" + "="*80)
        print("Training Progress")
        print("="*80)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        # Mini-batch training
        indices = torch.randperm(len(X_train))
        for i in range(0, len(X_train), batch_size):
            batch_indices = indices[i:i+batch_size]
            X_batch = X_train[batch_indices]
            y_batch = y_train[batch_indices]

            # Forward pass
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            # Backward pass
            loss.backward()

            # Gradient clipping (stabilize training)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            train_loss += loss.item()

        train_loss /= (len(X_train) // batch_size + 1)

        # Evaluate on test set
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test)
            test_loss = criterion(test_outputs, y_test).item()

            # Compute metrics
            from sklearn.metrics import average_precision_score, roc_auc_score
            y_test_np = y_test.cpu().numpy()
            y_scores = torch.sigmoid(test_outputs).cpu().numpy()

            test_auprc = average_precision_score(y_test_np, y_scores)
            test_auroc = roc_auc_score(y_test_np, y_scores)

        # Update history
        history['train_loss'].append(train_loss)
        history['test_loss'].append(test_loss)
        history['test_auprc'].append(test_auprc)
        history['test_auroc'].append(test_auroc)

        # Learning rate scheduling
        scheduler.step(test_loss)

        # Early stopping
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        # Print progress
        if verbose and (epoch % 10 == 0 or epoch == num_epochs - 1):
            print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | Test AUPRC: {test_auprc:.4f} | Test AUROC: {test_auroc:.4f}")

        # Early stopping check
        if patience_counter >= patience:
            if verbose:
                print(f"\nEarly stopping at epoch {epoch} (no improvement for {patience} epochs)")
            break

    # Load best model
    model.load_state_dict(best_model_state)

    if verbose:
        print("\n" + "="*80)
        print("TRAINING COMPLETE")
        print("="*80)
        best_epoch = history['test_auprc'].index(max(history['test_auprc']))
        print(f"Best Test AUPRC: {max(history['test_auprc']):.4f} (at epoch {best_epoch})")

    return {
        'model': model,
        'history': history,
        'best_model_state': best_model_state
    }


def evaluate_lstm(
    model: nn.Module,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    device: str = 'cpu'
) -> dict:
    """
    Evaluate LSTM model and return comprehensive metrics.

    Args:
        model: Trained LSTM model
        X_train, y_train: Training data (for generalization gap)
        X_test, y_test: Test data
        device: 'cpu' or 'cuda'

    Returns:
        Dictionary with all evaluation metrics
    """
    from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss

    model.eval()
    model = model.to(device)

    # Convert to tensors
    X_train = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.FloatTensor(y_train).to(device)
    X_test = torch.FloatTensor(X_test).to(device)
    y_test_tensor = torch.FloatTensor(y_test).to(device)

    with torch.no_grad():
        # Train predictions
        train_logits = model(X_train)
        train_probs = torch.sigmoid(train_logits).cpu().numpy()

        # Test predictions
        test_logits = model(X_test)
        test_probs = torch.sigmoid(test_logits).cpu().numpy()

    # Compute metrics
    train_auprc = average_precision_score(y_train, train_probs)
    train_auroc = roc_auc_score(y_train, train_probs)
    train_brier = brier_score_loss(y_train, train_probs)

    test_auprc = average_precision_score(y_test, test_probs)
    test_auroc = roc_auc_score(y_test, test_probs)
    test_brier = brier_score_loss(y_test, test_probs)

    # Generalization gap
    gap = train_auprc - test_auprc

    return {
        'train_auprc': train_auprc,
        'train_auroc': train_auroc,
        'train_brier': train_brier,
        'test_auprc': test_auprc,
        'test_auroc': test_auroc,
        'test_brier': test_brier,
        'gap': gap
    }
