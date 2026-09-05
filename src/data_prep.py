"""
Data preparation with network features for LSTM experiments.

Creates temporal sequences with:
- Event features (arrests, violence, disappearances, fatalities)
- Network structure features (12 graph metrics)
- NO population feature (noise reduction)
"""

import pandas as pd
import numpy as np
import networkx as nx
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List
import warnings
warnings.filterwarnings('ignore')


def load_acled_data(filepath: str, start_date: str = '2018-01-01') -> pd.DataFrame:
    """
    Load and preprocess ACLED data.

    Args:
        filepath: Path to ACLED CSV file
        start_date: Filter events after this date

    Returns:
        DataFrame with event indicators
    """
    print(f"Loading ACLED data from {filepath}...")
    df = pd.read_csv(filepath)
    df['event_date'] = pd.to_datetime(df['event_date'])
    df = df[df['event_date'] >= start_date]

    # Create indicators
    df['is_disappearance'] = (df['sub_event_type'] == 'Abduction/forced disappearance').astype(int)
    df['is_arrest'] = (df['sub_event_type'] == 'Arrests').astype(int)
    kinetic = ['Battles', 'Explosions/Remote violence', 'Violence against civilians']
    df['is_violence'] = df['event_type'].isin(kinetic).astype(int)

    # Fatalities (ensure numeric)
    df['fatalities'] = pd.to_numeric(df.get('fatalities', 0), errors='coerce').fillna(0)

    print(f"  Total events: {len(df):,}")
    print(f"  Date range: {df['event_date'].min()} to {df['event_date'].max()}")
    print(f"  Disappearances: {df['is_disappearance'].sum():,}")
    print(f"  Arrests: {df['is_arrest'].sum():,}")
    print(f"  Violence: {df['is_violence'].sum():,}")
    print(f"  Total fatalities: {df['fatalities'].sum():,.0f}")

    return df


def compute_network_features(actors: list, events_df: pd.DataFrame) -> dict:
    """
    Compute network structure features from actor co-occurrences.

    For a given region-month, build a network where:
    - Nodes = actors present
    - Edges = co-occurrence in same events
    - Weights = number of co-occurrences

    Args:
        actors: List of actor names present in this region-month
        events_df: DataFrame of events for this region-month (with actor columns)

    Returns:
        Dictionary of network features
    """
    # Default features (for empty networks)
    features = {
        'num_actors': 0,
        'num_edges': 0,
        'density': 0,
        'avg_degree': 0,
        'max_degree': 0,
        'num_components': 0,
        'largest_component_size': 0,
        'avg_clustering': 0,
        'transitivity': 0,
        'avg_edge_weight': 0,
        'max_edge_weight': 0,
        'degree_centralization': 0
    }

    # Handle empty or single actor cases
    if len(actors) == 0:
        return features

    if len(actors) == 1:
        features['num_actors'] = 1
        features['num_components'] = 1
        features['largest_component_size'] = 1
        return features

    # Build co-occurrence network
    G = nx.Graph()
    G.add_nodes_from(actors)

    # Count co-occurrences (actors appearing in same event)
    co_occurrence = {}
    for _, event in events_df.iterrows():
        # Get all actors in this event
        event_actors = []
        for col in ['actor1', 'actor2']:
            if col in event and pd.notna(event[col]) and event[col] in actors:
                event_actors.append(event[col])

        # Add edges for all pairs
        event_actors = list(set(event_actors))  # Unique
        for i, a1 in enumerate(event_actors):
            for a2 in event_actors[i+1:]:
                pair = tuple(sorted([a1, a2]))
                co_occurrence[pair] = co_occurrence.get(pair, 0) + 1

    # Add weighted edges
    for (a1, a2), weight in co_occurrence.items():
        G.add_edge(a1, a2, weight=weight)

    # Compute features
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()

    features['num_actors'] = num_nodes
    features['num_edges'] = num_edges

    # Density
    if num_nodes > 1:
        features['density'] = nx.density(G)

    # Degree statistics
    if num_edges > 0:
        degrees = [d for n, d in G.degree()]
        features['avg_degree'] = np.mean(degrees)
        features['max_degree'] = np.max(degrees)

        # Degree centralization: (max_degree - avg_degree) / theoretical_max
        # Theoretical max for a star graph: n-1
        if num_nodes > 2:
            theoretical_max = num_nodes - 1
            features['degree_centralization'] = (features['max_degree'] - features['avg_degree']) / theoretical_max

    # Components
    components = list(nx.connected_components(G))
    features['num_components'] = len(components)
    features['largest_component_size'] = len(max(components, key=len)) if components else 0

    # Clustering
    if num_edges > 0:
        try:
            clustering = nx.clustering(G)
            features['avg_clustering'] = np.mean(list(clustering.values()))
            features['transitivity'] = nx.transitivity(G)
        except:
            pass  # Some graphs can't compute clustering

    # Edge weights
    if num_edges > 0:
        weights = [G[u][v]['weight'] for u, v in G.edges()]
        features['avg_edge_weight'] = np.mean(weights)
        features['max_edge_weight'] = np.max(weights)

    return features


def create_monthly_aggregation_with_networks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate events to monthly level with network features.

    Args:
        df: Raw ACLED data with event indicators

    Returns:
        Monthly aggregated data with event counts, network features, and target
    """
    print("\nCreating monthly aggregation with network features...")

    df['month'] = df['event_date'].dt.to_period('M')

    # Store events for network computation
    monthly_data = []

    for (country, admin1, month), group in df.groupby(['country', 'admin1', 'month']):
        # Event counts
        row = {
            'country': country,
            'admin1': admin1,
            'month': month,
            'disappearances': group['is_disappearance'].sum(),
            'arrests': group['is_arrest'].sum(),
            'violence': group['is_violence'].sum(),
            'fatalities': group['fatalities'].sum(),
        }

        # Get actors for network
        actors = []
        for col in ['actor1', 'actor2']:
            if col in group.columns:
                actors.extend(group[col].dropna().unique().tolist())
        actors = list(set(actors))  # Unique actors

        # Compute network features
        network_features = compute_network_features(actors, group)
        row.update(network_features)

        monthly_data.append(row)

    monthly = pd.DataFrame(monthly_data)

    # Target: Binary - will there be disappearances NEXT month?
    monthly['target'] = (
        monthly.groupby(['country', 'admin1'])['disappearances']
        .shift(-1) > 0
    ).astype(float)

    # Drop rows with no target (last month per region)
    monthly_clean = monthly.dropna(subset=['target']).copy()

    print(f"  Shape: {monthly_clean.shape}")
    print(f"  Unique regions: {monthly_clean.groupby(['country', 'admin1']).ngroups}")
    print(f"  Unique months: {monthly_clean['month'].nunique()}")
    print(f"  Positive rate: {monthly_clean['target'].mean()*100:.1f}%")

    # Print network feature statistics
    print("\n  Network feature statistics:")
    network_cols = ['num_actors', 'num_edges', 'density', 'avg_degree', 'max_degree',
                    'avg_clustering', 'transitivity', 'degree_centralization']
    for col in network_cols:
        if col in monthly_clean.columns:
            print(f"    {col:25s}: mean={monthly_clean[col].mean():6.2f}, max={monthly_clean[col].max():6.0f}")

    return monthly_clean


def create_temporal_sequences(
    monthly_df: pd.DataFrame,
    sequence_length: int = 6
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List, np.ndarray]:
    """
    Create temporal sequences for LSTM training.

    Features per month (16 total):
    - Event features: arrests, violence, disappearances, fatalities (4)
    - Network features: 12 graph metrics
    - NO population (removed as noise)

    Args:
        monthly_df: Monthly aggregated data with network features
        sequence_length: Number of months in lookback window

    Returns:
        X: [num_sequences, sequence_length, 16] (UN-normalized)
        y: [num_sequences]
        region_ids: [num_sequences]
        regions: List of (country, admin1) tuples
        month_targets: [num_sequences] (which month each sequence predicts)
    """
    print(f"\nCreating temporal sequences (length={sequence_length})...")

    # Feature columns (16 total)
    feature_cols = [
        # Event features (4)
        'arrests', 'violence', 'disappearances', 'fatalities',
        # Network features (12)
        'num_actors', 'num_edges', 'density', 'avg_degree', 'max_degree',
        'num_components', 'largest_component_size', 'avg_clustering',
        'transitivity', 'avg_edge_weight', 'max_edge_weight', 'degree_centralization'
    ]

    print(f"  Features ({len(feature_cols)}): {', '.join(feature_cols)}")

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

        # Create feature matrix
        features = np.zeros((len(region_data), len(feature_cols)))
        for i, col in enumerate(feature_cols):
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
            # Input: months i to i+sequence_length-1
            X_seq = features[i:i+sequence_length]

            # Target: month i+sequence_length
            y_target = targets[i+sequence_length]
            target_month = months[i+sequence_length]

            sequences_X.append(X_seq)
            sequences_y.append(y_target)
            sequences_region.append(region_id)
            sequences_target_month.append(target_month)

    X = np.array(sequences_X)  # [num_sequences, seq_len, features]
    y = np.array(sequences_y)  # [num_sequences]
    region_ids = np.array(sequences_region)
    month_targets = np.array(sequences_target_month)

    print(f"  Created {len(X)} sequences")
    print(f"  Shape: {X.shape}")
    print(f"  Positive rate: {y.mean()*100:.1f}%")
    print(f"  ⚠️  Data is UN-normalized (normalize after train/test split)")

    return X, y, region_ids, regions, month_targets


def temporal_split(
    X: np.ndarray,
    y: np.ndarray,
    region_ids: np.ndarray,
    month_targets: np.ndarray,
    split_ratio: float = 0.7
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split by TIME (actual date cutoff) to respect temporal ordering.

    Args:
        X: Sequences [num_sequences, seq_len, features]
        y: Targets [num_sequences]
        region_ids: Region identifiers [num_sequences]
        month_targets: Which month each sequence predicts [num_sequences]
        split_ratio: Fraction of TIME PERIOD for training

    Returns:
        X_train, y_train, X_test, y_test (UN-normalized)
    """
    # Find unique months and cutoff
    unique_months = np.unique(month_targets)
    unique_months_sorted = np.sort(unique_months)

    cutoff_idx = int(len(unique_months_sorted) * split_ratio)
    cutoff_month = unique_months_sorted[cutoff_idx]

    print(f"\nTemporal split by DATE ({split_ratio:.0%} of time period for train):")
    print(f"  Total unique months: {len(unique_months_sorted)}")
    print(f"  Cutoff month index: {cutoff_idx} / {len(unique_months_sorted)}")
    print(f"  Cutoff month: {cutoff_month}")

    # Split sequences based on which month they predict
    train_mask = month_targets < cutoff_month
    test_mask = month_targets >= cutoff_month

    X_train = X[train_mask]
    y_train = y[train_mask]
    region_train = region_ids[train_mask]
    months_train = month_targets[train_mask]

    X_test = X[test_mask]
    y_test = y[test_mask]
    region_test = region_ids[test_mask]
    months_test = month_targets[test_mask]

    print(f"\n  Train:")
    print(f"    Sequences: {len(X_train)}")
    print(f"    Positive rate: {y_train.mean()*100:.1f}%")
    print(f"    Unique regions: {len(np.unique(region_train))}")
    print(f"    Month range: {months_train.min()} to {months_train.max()}")

    print(f"  Test:")
    print(f"    Sequences: {len(X_test)}")
    print(f"    Positive rate: {y_test.mean()*100:.1f}%")
    print(f"    Unique regions: {len(np.unique(region_test))}")
    print(f"    Month range: {months_test.min()} to {months_test.max()}")

    gap = abs(y_train.mean() - y_test.mean()) * 100
    print(f"\n  Distribution gap: {gap:.1f} percentage points")

    # VERIFICATION: Check for temporal leakage
    train_latest = months_train.max()
    test_earliest = months_test.min()
    print(f"\n  ✅ Verification: No temporal overlap")
    print(f"     Latest train month: {train_latest}")
    print(f"     Earliest test month: {test_earliest}")
    print(f"     Gap: {test_earliest > train_latest}")

    return X_train, y_train, X_test, y_test


def normalize_sequences(
    X_train: np.ndarray,
    X_test: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    Normalize sequences AFTER train/test split.

    Fits scaler only on training data, then applies to both train and test.

    Args:
        X_train: Training sequences [num_train, seq_len, features]
        X_test: Test sequences [num_test, seq_len, features]

    Returns:
        X_train_normalized, X_test_normalized, scaler
    """
    print(f"\nNormalizing features (fit on train only)...")

    # Flatten for normalization
    X_train_flat = X_train.reshape(-1, X_train.shape[2])
    X_test_flat = X_test.reshape(-1, X_test.shape[2])

    # Fit scaler ONLY on train
    scaler = StandardScaler()
    X_train_normalized = scaler.fit_transform(X_train_flat).reshape(X_train.shape)

    # Apply to test (using train statistics)
    X_test_normalized = scaler.transform(X_test_flat).reshape(X_test.shape)

    print(f"  Train - Mean: {X_train_normalized.mean():.6f}, Std: {X_train_normalized.std():.6f}")
    print(f"  Test  - Mean: {X_test_normalized.mean():.6f}, Std: {X_test_normalized.std():.6f}")
    print(f"  ✅ No information leakage (test stats not used in normalization)")

    return X_train_normalized, X_test_normalized, scaler
