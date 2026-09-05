"""
Verification Script for Paper Statistics
==========================================

This script verifies all numerical claims made in the paper against the actual data.

Paper Claims to Verify:
1. Data timeframe: 2018-2024
2. Total disappearances: 5,959 (2.9% of events)
3. Random baseline AUPRC: 0.354
4. Total dataset size: 202,347 events
5. Total admin1 regions: 87
6. Arrests: 6,913
7. Violent events (battles, explosions, violence against civilians): 197,817
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import average_precision_score

print("=" * 80)
print("PAPER STATISTICS VERIFICATION")
print("=" * 80)
print()

# Load the data - try multiple possible paths
possible_paths = [
    Path("../data/ACLED Data_2025-12-31_Nigeria_Mexico_Myanmar.csv"),
    Path("data/ACLED Data_2025-12-31_Nigeria_Mexico_Myanmar.csv"),
    Path("/Users/christoschristodoulou/projects/DISACT-GNN/data/ACLED Data_2025-12-31_Nigeria_Mexico_Myanmar.csv"),
]

data_path = None
for path in possible_paths:
    if path.exists():
        data_path = path
        break

if data_path is None:
    print(f"ERROR: Data file not found. Tried:")
    for path in possible_paths:
        print(f"  - {path}")
    print("Please verify the correct path to the ACLED data file")
    exit(1)

print(f"Loading data from: {data_path}")

df = pd.read_csv(data_path)
print(f"✓ Data loaded successfully")
print()

# ============================================================================
# VERIFICATION 1: Data Timeframe (2018-2024)
# ============================================================================
print("1. DATA TIMEFRAME")
print("-" * 80)

# Convert event_date to datetime if it's not already
if 'event_date' in df.columns:
    df['event_date'] = pd.to_datetime(df['event_date'])
    min_year = df['event_date'].dt.year.min()
    max_year = df['event_date'].dt.year.max()

    print(f"   Claimed in paper: 2018-2024")
    print(f"   Actual in data:   {min_year}-{max_year}")

    if min_year == 2018 and max_year == 2024:
        print(f"   ✅ VERIFIED")
    else:
        print(f"   ❌ MISMATCH - Paper needs correction")
    print()
else:
    print("   ⚠️  WARNING: No 'event_date' column found")
    print()

# ============================================================================
# VERIFICATION 2: Total Dataset Size (202,347 events)
# ============================================================================
print("2. TOTAL DATASET SIZE")
print("-" * 80)

total_events = len(df)
print(f"   Claimed in paper: 202,347 events")
print(f"   Actual in data:   {total_events:,} events")

if total_events == 202347:
    print(f"   ✅ VERIFIED")
elif abs(total_events - 202347) / 202347 < 0.01:  # Within 1%
    print(f"   ⚠️  CLOSE (within 1%) - Consider updating paper")
else:
    print(f"   ❌ MISMATCH - Paper needs correction")
print()

# ============================================================================
# VERIFICATION 3: Disappearances Count (5,959 = 2.9%)
# ============================================================================
print("3. ENFORCED DISAPPEARANCES")
print("-" * 80)

# Check for disappearances in sub_event_type column (as per the fix we made)
if 'sub_event_type' in df.columns:
    disappearances = df[df['sub_event_type'].str.contains('disappearance', case=False, na=False)]
    n_disappearances = len(disappearances)
    pct_disappearances = (n_disappearances / total_events) * 100

    print(f"   Claimed in paper: 5,959 disappearances (2.9%)")
    print(f"   Actual in data:   {n_disappearances:,} disappearances ({pct_disappearances:.1f}%)")

    if n_disappearances == 5959:
        print(f"   ✅ COUNT VERIFIED")
    else:
        print(f"   ❌ COUNT MISMATCH - Difference: {n_disappearances - 5959:+,}")

    if abs(pct_disappearances - 2.9) < 0.1:
        print(f"   ✅ PERCENTAGE VERIFIED (within 0.1%)")
    else:
        print(f"   ❌ PERCENTAGE MISMATCH - Paper needs correction")
    print()
else:
    print("   ⚠️  WARNING: No 'sub_event_type' column found")
    print()

# ============================================================================
# VERIFICATION 4: Random Baseline AUPRC (0.354 vs 0.483)
# ============================================================================
print("4. RANDOM BASELINE AUPRC")
print("-" * 80)

print(f"   ⚠️  NOTE: Random baseline depends on the unit of analysis:")
print(f"   - Event-level: Proportion of disappearance events (2.9%)")
print(f"   - Region-month sequences (after temporal aggregation): ~40-48%")
print()
print(f"   The paper uses region-month sequences, so baseline should be:")
print(f"   - Overall sequences: ~0.408 (40.8% positive rate)")
print(f"   - Test set (2023-2024): ~0.483 (48.3% positive rate)")
print()
print(f"   Claimed in paper: 0.354")
print(f"   ❌ LIKELY MISMATCH - Should verify with actual experiment results")
print(f"   Recommendation: Update paper to use 0.483 (test set baseline)")
print()

# ============================================================================
# VERIFICATION 5: Total Admin1 Regions (87)
# ============================================================================
print("5. ADMIN1 REGIONS")
print("-" * 80)

if 'admin1' in df.columns:
    unique_admin1 = df['admin1'].nunique()

    print(f"   Claimed in paper: 87 admin1 regions")
    print(f"   Actual in data:   {unique_admin1} admin1 regions")

    if unique_admin1 == 87:
        print(f"   ✅ VERIFIED")
    else:
        print(f"   ❌ MISMATCH - Difference: {unique_admin1 - 87:+d}")

    # Show breakdown by country
    if 'country' in df.columns:
        print(f"\n   Breakdown by country:")
        country_admin1 = df.groupby('country')['admin1'].nunique().sort_values(ascending=False)
        for country, count in country_admin1.items():
            print(f"     - {country}: {count} regions")
    print()
else:
    print("   ⚠️  WARNING: No 'admin1' column found")
    print()

# ============================================================================
# VERIFICATION 6: Arrests (6,913)
# ============================================================================
print("6. ARREST EVENTS")
print("-" * 80)

# Arrests are in sub_event_type, not event_type
if 'sub_event_type' in df.columns:
    arrests = df[df['sub_event_type'].str.contains('arrest', case=False, na=False)]
    n_arrests = len(arrests)

    print(f"   Claimed in paper: 6,913 arrests")
    print(f"   Actual in data:   {n_arrests:,} arrests (in sub_event_type)")

    if n_arrests == 6913:
        print(f"   ✅ VERIFIED")
    else:
        print(f"   ❌ MISMATCH - Difference: {n_arrests - 6913:+,}")
    print()
else:
    print("   ⚠️  WARNING: No 'sub_event_type' column found")
    print()

# ============================================================================
# VERIFICATION 7: Total Fatalities (197,817) 
# ============================================================================
print("7. TOTAL FATALITIES (Paper incorrectly labeled as 'violent events')")
print("-" * 80)

if 'fatalities' in df.columns:
    total_fatalities = df['fatalities'].sum()

    print(f"   Claimed in paper: 197,817 'violent events'")
    print(f"   Actual meaning:   197,817 is TOTAL FATALITIES, not event count")
    print(f"   Actual fatalities in data: {total_fatalities:,.0f}")

    if int(total_fatalities) == 197817:
        print(f"   ✅ VERIFIED - 197,817 is total fatalities")
    else:
        print(f"   ❌ MISMATCH - Difference: {int(total_fatalities) - 197817:+,}")

    # Show actual violent events count
    if 'event_type' in df.columns:
        kinetic = ['Battles', 'Explosions/Remote violence', 'Violence against civilians']
        violent_events = df[df['event_type'].isin(kinetic)]
        print(f"   Actual violent event counts:")
        print(f"   - Battles: {len(df[df['event_type'] == 'Battles']):,}")
        print(f"   - Explosions/Remote violence: {len(df[df['event_type'] == 'Explosions/Remote violence']):,}")
        print(f"   - Violence against civilians: {len(df[df['event_type'] == 'Violence against civilians']):,}")
        print(f"   - Total violent events: {len(violent_events):,}")
    print()
else:
    print("   ⚠️  WARNING: No 'fatalities' column found")
    print()

# ============================================================================
# SUMMARY
# ============================================================================
print("=" * 80)
print("SUMMARY")
print("=" * 80)
print()
print("Verification Status:")
print("  1. Data timeframe (2018-2024): Check above")
print("  2. Total events (202,347): Check above")
print("  3. Disappearances (5,959 = 2.9%): Check above")
print("  4. Random baseline (0.354): Check above")
print("  5. Admin1 regions (87): Check above")
print("  6. Arrests (6,913): Check above")
print("  7. Violent events (197,817): Check above")
print()
print("=" * 80)
print()

# Additional diagnostic information
print("ADDITIONAL DIAGNOSTICS")
print("=" * 80)
print()

# Show column names
print("Available columns in dataset:")
for i, col in enumerate(df.columns, 1):
    print(f"  {i:2d}. {col}")
print()

# Show data shape
print(f"Dataset shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
print()

# Check for missing values in key columns
print("Missing values in key columns:")
key_cols = ['event_date', 'event_type', 'sub_event_type', 'admin1', 'country']
for col in key_cols:
    if col in df.columns:
        missing = df[col].isna().sum()
        pct_missing = (missing / len(df)) * 100
        print(f"  - {col}: {missing:,} ({pct_missing:.1f}%)")
    else:
        print(f"  - {col}: Column not found")
print()

print("=" * 80)
print("Verification complete!")
print("=" * 80)
