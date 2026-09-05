"""
Experiment runner for LSTM with network features.
"""

import numpy as np
import torch
from .data_prep import (
    load_acled_data,
    create_monthly_aggregation_with_networks,
    create_temporal_sequences,
    temporal_split,
    normalize_sequences
)
from .models import StandardLSTM, AttentionLSTM, train_lstm, evaluate_lstm


def run_single_experiment(
    data_path: str,
    seed: int = 0,
    use_attention: bool = False,
    sequence_length: int = 6,
    hidden_dim: int = 64,
    num_layers: int = 2,
    dropout: float = 0.2,
    num_epochs: int = 50,
    batch_size: int = 64,
    split_ratio: float = 0.7,
    device: str = 'cpu',
    verbose: bool = True
) -> dict:
    """
    Run a single LSTM experiment.

    Args:
        data_path: Path to ACLED CSV file
        seed: Random seed for reproducibility
        use_attention: Use attention mechanism
        sequence_length: Number of months in lookback window
        hidden_dim: LSTM hidden dimension
        num_layers: Number of LSTM layers
        dropout: Dropout rate
        num_epochs: Number of training epochs
        batch_size: Batch size
        split_ratio: Train/test temporal split ratio
        device: 'cpu' or 'cuda'
        verbose: Print progress

    Returns:
        Dictionary with metrics and model
    """
    # Set seeds
    np.random.seed(seed)
    torch.manual_seed(seed)

    if verbose:
        print("="*80)
        print(f"RUNNING LSTM EXPERIMENT WITH NETWORK FEATURES")
        print("="*80)
        print(f"Seed: {seed}")
        print(f"Model: {'Attention-LSTM' if use_attention else 'Standard LSTM'}")
        print(f"Split ratio: {split_ratio:.0%}")

    # Load data
    df = load_acled_data(data_path)

    # Create monthly aggregation with network features
    monthly_df = create_monthly_aggregation_with_networks(df)

    # Create temporal sequences
    X, y, region_ids, regions, month_targets = create_temporal_sequences(
        monthly_df,
        sequence_length=sequence_length
    )

    # Temporal split
    X_train, y_train, X_test, y_test = temporal_split(
        X, y, region_ids, month_targets, split_ratio=split_ratio
    )

    # Normalize
    X_train, X_test, scaler = normalize_sequences(X_train, X_test)

    # Create model
    input_dim = X_train.shape[2]

    if use_attention:
        model = AttentionLSTM(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout
        )
    else:
        model = StandardLSTM(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout
        )

    # Compute class weight
    pos_weight = (1 - y_train.mean()) / y_train.mean()

    # Train
    train_result = train_lstm(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        num_epochs=num_epochs,
        batch_size=batch_size,
        pos_weight=pos_weight,
        device=device,
        verbose=verbose
    )

    # Evaluate
    metrics = evaluate_lstm(
        model=train_result['model'],
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        device=device
    )

    if verbose:
        print("\nFinal Generalization Metrics:")
        print(f"Train AUPRC: {metrics['train_auprc']:.4f} | AUROC: {metrics['train_auroc']:.4f} | Brier: {metrics['train_brier']:.4f}")
        print(f"Test  AUPRC: {metrics['test_auprc']:.4f} | AUROC: {metrics['test_auroc']:.4f} | Brier: {metrics['test_brier']:.4f}")
        print(f"Generalization Gap: {metrics['gap']:.4f}")

    return {
        'metrics': metrics,
        'model': train_result['model'],
        'history': train_result['history'],
        'scaler': scaler
    }


def run_comparison(
    data_path: str,
    num_seeds: int = 3,
    sequence_length: int = 6,
    hidden_dim: int = 64,
    num_layers: int = 2,
    dropout: float = 0.2,
    num_epochs: int = 50,
    batch_size: int = 64,
    split_ratio: float = 0.7,
    device: str = 'cpu',
    verbose: bool = True
) -> dict:
    """
    Compare Standard LSTM vs. Attention-LSTM with multiple seeds.

    Args:
        data_path: Path to ACLED CSV file
        num_seeds: Number of random seeds to run
        Other args: Same as run_single_experiment

    Returns:
        Dictionary with comparison results
    """
    print("="*80)
    print("COMPARISON: Standard LSTM vs. Attention-LSTM with Network Features")
    print("="*80)
    print(f"Seeds: {num_seeds}")
    print(f"Sequence length: {sequence_length}")

    results = {
        'standard_lstm': [],
        'attention_lstm': []
    }

    for seed in range(num_seeds):
        print("\n" + "="*80)
        print(f"SEED {seed+1}/{num_seeds}")
        print("="*80)

        # Standard LSTM
        print("\n" + "="*80)
        print("RUNNING STANDARD LSTM")
        print("="*80)
        result_standard = run_single_experiment(
            data_path=data_path,
            seed=seed,
            use_attention=False,
            sequence_length=sequence_length,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            num_epochs=num_epochs,
            batch_size=batch_size,
            split_ratio=split_ratio,
            device=device,
            verbose=verbose
        )
        results['standard_lstm'].append(result_standard['metrics'])

        # Attention-LSTM
        print("\n" + "="*80)
        print("RUNNING ATTENTION-LSTM")
        print("="*80)
        result_attention = run_single_experiment(
            data_path=data_path,
            seed=seed,
            use_attention=True,
            sequence_length=sequence_length,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            num_epochs=num_epochs,
            batch_size=batch_size,
            split_ratio=split_ratio,
            device=device,
            verbose=verbose
        )
        results['attention_lstm'].append(result_attention['metrics'])

    # Print comparison
    print("\n" + "="*80)
    print("FINAL RESULTS (mean ± std)")
    print("="*80)

    for model_name, model_results in results.items():
        test_auprcs = [r['test_auprc'] for r in model_results]
        train_auprcs = [r['train_auprc'] for r in model_results]
        gaps = [r['gap'] for r in model_results]
        briers = [r['test_brier'] for r in model_results]

        print(f"\n{model_name.replace('_', ' ').title()}:")
        print(f"  Test AUPRC:  {np.mean(test_auprcs):.4f} ± {np.std(test_auprcs):.4f}")
        print(f"  Train AUPRC: {np.mean(train_auprcs):.4f} ± {np.std(train_auprcs):.4f}")
        print(f"  Gap:         {np.mean(gaps):.4f} ± {np.std(gaps):.4f}")
        print(f"  Brier:       {np.mean(briers):.4f} ± {np.std(briers):.4f}")

    # Determine winner
    standard_mean = np.mean([r['test_auprc'] for r in results['standard_lstm']])
    attention_mean = np.mean([r['test_auprc'] for r in results['attention_lstm']])

    print("\n" + "="*80)
    winner = "Attention-LSTM" if attention_mean > standard_mean else "Standard LSTM"
    winner_score = max(attention_mean, standard_mean)
    print(f"🏆 Winner: {winner}")
    print(f"   Score: {winner_score:.4f}")
    print("="*80)

    return results


def print_comparison_table(results: dict):
    """
    Print a formatted comparison table.

    Args:
        results: Dictionary from run_comparison
    """
    print("\n" + "="*80)
    print("COMPARISON TABLE")
    print("="*80)

    print(f"\n{'Model':<25} {'Test AUPRC':<20} {'Train AUPRC':<12} {'Gap':<10} {'Brier':<10}")
    print("-" * 85)

    for model_name, model_results in results.items():
        test_auprcs = [r['test_auprc'] for r in model_results]
        train_auprcs = [r['train_auprc'] for r in model_results]
        gaps = [r['gap'] for r in model_results]
        briers = [r['test_brier'] for r in model_results]

        display_name = model_name.replace('_', ' ').title()

        print(f"{display_name:<25} {np.mean(test_auprcs):.4f} ± {np.std(test_auprcs):.4f}    "
              f"{np.mean(train_auprcs):.4f}      "
              f"{np.mean(gaps):>6.4f}    "
              f"{np.mean(briers):.4f}")

        # Progress bar
        bar_length = int(np.mean(test_auprcs) * 50)
        print(f"{'':25} {'█' * bar_length}")

    print("-" * 85)
