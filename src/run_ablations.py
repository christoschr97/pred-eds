"""
Ablation study runner for LSTM with network features.

Tests what happens when we remove different feature groups.
"""

import numpy as np
import torch
from typing import List, Dict
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_prep import (
    load_acled_data,
    create_monthly_aggregation_with_networks,
    create_temporal_sequences,
    temporal_split,
    normalize_sequences
)
from models import AttentionLSTM, train_lstm, evaluate_lstm


def create_sequences_with_feature_subset(
    monthly_df,
    sequence_length: int,
    feature_subset: List[str]
):
    """
    Create temporal sequences with only specified features.

    Similar to create_temporal_sequences but allows feature selection.
    """
    print(f"\nCreating temporal sequences with {len(feature_subset)} features...")
    print(f"  Features: {', '.join(feature_subset)}")

    # Get unique regions
    regions = sorted(monthly_df.groupby(['country', 'admin1']).groups.keys())
    region_to_id = {r: i for i, r in enumerate(regions)}

    print(f"  Regions: {len(regions)}")

    # Prepare features
    monthly_df = monthly_df.copy()
    monthly_df['region_id'] = monthly_df.apply(
        lambda x: region_to_id[(x['country'], x['admin1'])], axis=1
    )

    # Sort by region and time
    monthly_df = monthly_df.sort_values(['region_id', 'month']).reset_index(drop=True)

    sequences_X = []
    sequences_y = []
    sequences_region = []
    sequences_target_month = []

    # For each region, create sliding windows
    for region_id in range(len(regions)):
        region_data = monthly_df[monthly_df['region_id'] == region_id].copy()

        # Skip if not enough data
        if len(region_data) < sequence_length + 1:
            continue

        # Create feature matrix with only selected features
        features = np.zeros((len(region_data), len(feature_subset)))
        for i, col in enumerate(feature_subset):
            if col in region_data.columns:
                features[:, i] = region_data[col].values
            else:
                print(f"  Warning: Missing feature column {col}")

        # Replace any NaN/Inf
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        # Get targets and months
        targets = region_data['target'].values
        months = region_data['month'].values

        # Create sliding windows
        for i in range(len(region_data) - sequence_length):
            X_seq = features[i:i+sequence_length]
            y_target = targets[i+sequence_length]
            target_month = months[i+sequence_length]

            sequences_X.append(X_seq)
            sequences_y.append(y_target)
            sequences_region.append(region_id)
            sequences_target_month.append(target_month)

    X = np.array(sequences_X)
    y = np.array(sequences_y)
    region_ids = np.array(sequences_region)
    month_targets = np.array(sequences_target_month)

    print(f"  Created {len(X)} sequences")
    print(f"  Shape: {X.shape}")
    print(f"  Positive rate: {y.mean()*100:.1f}%")

    return X, y, region_ids, regions, month_targets


def run_ablation_experiment(
    data_path: str,
    feature_subset: List[str],
    experiment_name: str,
    seed: int = 0,
    sequence_length: int = 6,
    num_epochs: int = 50,
    device: str = 'cpu',
    verbose: bool = True
) -> dict:
    """
    Run experiment with specific feature subset.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    if verbose:
        print("\n" + "="*80)
        print(f"ABLATION: {experiment_name}")
        print("="*80)
        print(f"Seed: {seed}")
        print(f"Features: {len(feature_subset)}")

    # Load data
    df = load_acled_data(data_path)
    monthly_df = create_monthly_aggregation_with_networks(df)

    # Create sequences with feature subset
    X, y, region_ids, regions, month_targets = create_sequences_with_feature_subset(
        monthly_df,
        sequence_length=sequence_length,
        feature_subset=feature_subset
    )

    # Temporal split
    X_train, y_train, X_test, y_test = temporal_split(
        X, y, region_ids, month_targets, split_ratio=0.7
    )

    # Normalize
    X_train, X_test, scaler = normalize_sequences(X_train, X_test)

    # Create model
    input_dim = X_train.shape[2]
    model = AttentionLSTM(
        input_dim=input_dim,
        hidden_dim=64,
        num_layers=2,
        dropout=0.2
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
        batch_size=64,
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
        print(f"\n{experiment_name} Test AUPRC: {metrics['test_auprc']:.4f}")

    return metrics


def run_all_ablations(
    data_path: str,
    num_seeds: int = 3,
    num_epochs: int = 50,
    device: str = 'cpu'
) -> Dict[str, List[dict]]:
    """
    Run all ablation studies.
    """
    print("="*80)
    print("ABLATION STUDIES: Feature Importance Analysis")
    print("="*80)
    print(f"Seeds: {num_seeds}")
    print(f"Epochs: {num_epochs}")

    # Define feature groups
    event_features = ['arrests', 'violence', 'disappearances', 'fatalities']
    network_features = [
        'num_actors', 'num_edges', 'density', 'avg_degree', 'max_degree',
        'num_components', 'largest_component_size', 'avg_clustering',
        'transitivity', 'avg_edge_weight', 'max_edge_weight', 'degree_centralization'
    ]
    all_features = event_features + network_features

    # Ablation configurations
    ablations = {
        'full': {
            'features': all_features,
            'name': 'Full Model (All Features)'
        },
        'no_network': {
            'features': event_features,
            'name': 'Events Only (No Network)'
        },
        'no_events': {
            'features': network_features,
            'name': 'Network Only (No Events)'
        },
        'no_disappearances': {
            'features': [f for f in all_features if f != 'disappearances'],
            'name': 'No Past Disappearances (Early Warning)'
        },
        'no_fatalities': {
            'features': [f for f in all_features if f != 'fatalities'],
            'name': 'No Fatalities'
        },
        'structure_only': {
            'features': event_features + ['num_actors', 'num_edges', 'density',
                                         'num_components', 'largest_component_size'],
            'name': 'Events + Structure Features'
        },
        'centrality_only': {
            'features': event_features + ['avg_degree', 'max_degree', 'degree_centralization'],
            'name': 'Events + Centrality Features'
        },
        'clustering_only': {
            'features': event_features + ['avg_clustering', 'transitivity'],
            'name': 'Events + Clustering Features'
        },
        'weights_only': {
            'features': event_features + ['avg_edge_weight', 'max_edge_weight'],
            'name': 'Events + Weight Features'
        }
    }

    results = {}

    for ablation_key, config in ablations.items():
        print(f"\n{'='*80}")
        print(f"ABLATION: {config['name']}")
        print(f"{'='*80}")

        ablation_results = []

        for seed in range(num_seeds):
            print(f"\n--- Seed {seed+1}/{num_seeds} ---")

            metrics = run_ablation_experiment(
                data_path=data_path,
                feature_subset=config['features'],
                experiment_name=config['name'],
                seed=seed,
                num_epochs=num_epochs,
                device=device,
                verbose=True
            )

            ablation_results.append(metrics)

        results[ablation_key] = ablation_results

    return results, ablations


def print_ablation_results(results: Dict[str, List[dict]], ablations: dict):
    """
    Print formatted ablation results.
    """
    print("\n" + "="*80)
    print("ABLATION STUDY RESULTS (mean ± std)")
    print("="*80)

    # Sort by test AUPRC
    sorted_keys = sorted(
        results.keys(),
        key=lambda k: np.mean([r['test_auprc'] for r in results[k]]),
        reverse=True
    )

    print(f"\n{'Experiment':<45} {'Test AUPRC':<18} {'Δ from Full':<12} {'#Features':<10}")
    print("-" * 95)

    full_mean = np.mean([r['test_auprc'] for r in results['full']])

    for key in sorted_keys:
        ablation_results = results[key]
        test_auprcs = [r['test_auprc'] for r in ablation_results]

        mean_auprc = np.mean(test_auprcs)
        std_auprc = np.std(test_auprcs)
        delta = mean_auprc - full_mean
        num_features = len(ablations[key]['features'])

        delta_str = f"{delta:+.4f}" if key != 'full' else "baseline"

        print(f"{ablations[key]['name']:<45} {mean_auprc:.4f} ± {std_auprc:.4f}    {delta_str:<12} {num_features:<10}")

        # Progress bar
        bar_length = int(mean_auprc * 50)
        print(f"{'':45} {'█' * bar_length}")

    print("-" * 95)

    # Key findings
    print("\n" + "="*80)
    print("KEY FINDINGS")
    print("="*80)

    no_network_mean = np.mean([r['test_auprc'] for r in results['no_network']])
    no_disapp_mean = np.mean([r['test_auprc'] for r in results['no_disappearances']])
    network_only_mean = np.mean([r['test_auprc'] for r in results['no_events']])

    print(f"\n1. Network Features Contribution:")
    print(f"   Full model: {full_mean:.4f}")
    print(f"   Events only (no network): {no_network_mean:.4f}")
    print(f"   → Network adds: {full_mean - no_network_mean:+.4f} ({(full_mean - no_network_mean)/no_network_mean*100:+.1f}%)")

    print(f"\n2. Past Disappearances Dependency:")
    print(f"   With past disappearances: {full_mean:.4f}")
    print(f"   Without past disappearances: {no_disapp_mean:.4f}")
    print(f"   → Disappearances add: {full_mean - no_disapp_mean:+.4f} ({(full_mean - no_disapp_mean)/full_mean*100:.1f}% of performance)")

    print(f"\n3. Network vs. Events:")
    print(f"   Events only: {no_network_mean:.4f}")
    print(f"   Network only: {network_only_mean:.4f}")
    print(f"   → Events are {no_network_mean - network_only_mean:.4f} points better")

    # Detailed network feature analysis
    print(f"\n4. Network Feature Categories:")
    structure_mean = np.mean([r['test_auprc'] for r in results['structure_only']])
    centrality_mean = np.mean([r['test_auprc'] for r in results['centrality_only']])
    clustering_mean = np.mean([r['test_auprc'] for r in results['clustering_only']])
    weights_mean = np.mean([r['test_auprc'] for r in results['weights_only']])

    print(f"   Structure features: {structure_mean:.4f} (Δ={structure_mean - no_network_mean:+.4f})")
    print(f"   Centrality features: {centrality_mean:.4f} (Δ={centrality_mean - no_network_mean:+.4f})")
    print(f"   Clustering features: {clustering_mean:.4f} (Δ={clustering_mean - no_network_mean:+.4f})")
    print(f"   Weight features: {weights_mean:.4f} (Δ={weights_mean - no_network_mean:+.4f})")

    best_category = max(
        [('Structure', structure_mean), ('Centrality', centrality_mean),
         ('Clustering', clustering_mean), ('Weights', weights_mean)],
        key=lambda x: x[1]
    )
    print(f"   → Best category: {best_category[0]} features")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Run ablation studies')
    parser.add_argument('--data', type=str,
                       default='./data/ACLED Data_2025-12-31_Nigeria_Mexico_Myanmar.csv',
                       help='Path to ACLED CSV file')
    parser.add_argument('--seeds', type=int, default=3,
                       help='Number of random seeds')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'],
                       help='Device to use')

    args = parser.parse_args()

    # Run ablations
    results, ablations = run_all_ablations(
        data_path=args.data,
        num_seeds=args.seeds,
        num_epochs=args.epochs,
        device=args.device
    )

    # Print results
    print_ablation_results(results, ablations)

    print("\n" + "="*80)
    print("ABLATION STUDY COMPLETE")
    print("="*80)
