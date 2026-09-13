"""
Experiment 3 — Persistence Baseline
=====================================
Reviewer response: standard conflict-forecasting sanity check (ViEWS benchmark).
Reviewer 2 explicitly flags its absence.

Method:
- For each region-month in test set: ŷ(t+1) = y(t)
  i.e. "disappearances happened last month → predict they'll happen again"
- y(t) is extracted from the LAST timestep of the input sequence,
  feature index 2 = 'disappearances' (UN-normalized raw count)
- Binarised: ŷ = (disappearances_at_t > 0).astype(int)
- For Jan 2023 (first test month) this naturally uses Dec 2022 label ✓
- No model training — pure post-processing on the test set

Output: AUPRC, AUROC, Brier score — new row in Table 3
Save to: results/exp3_persistence.json
"""

import sys
import os
import json
import numpy as np

# ── path plumbing ─────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_prep import (
    DATA_FILE,
    load_acled_data,
    create_monthly_aggregation_with_networks,
    create_temporal_sequences,
    temporal_split,
)

from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss

# ── config ────────────────────────────────────────────────────────────────────
DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data",
    DATA_FILE,
)

SEQUENCE_LENGTH      = 6
SPLIT_RATIO          = 0.7
DISAPPEARANCES_IDX   = 2   # position of 'disappearances' in the 16-feature vector

RESULTS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "results",
    "exp3_persistence.json",
)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "="*70)
    print("  EXPERIMENT 3 — Persistence Baseline")
    print("  ŷ(t+1) = y(t)  [standard ViEWS sanity check]")
    print("="*70)

    print(f"\nData path: {DATA_PATH}")
    df         = load_acled_data(DATA_PATH)
    monthly_df = create_monthly_aggregation_with_networks(df)

    # ── build sequences — capture month_targets BEFORE split ──────────────────
    X, y, region_ids, regions, month_targets = create_temporal_sequences(
        monthly_df, sequence_length=SEQUENCE_LENGTH
    )
    # X shape: [N, 6, 16]  — UN-normalized at this point

    # ── replicate the temporal split cutoff to get test mask ──────────────────
    unique_months       = np.unique(month_targets)
    unique_months_sorted = np.sort(unique_months)
    cutoff_idx          = int(len(unique_months_sorted) * SPLIT_RATIO)
    cutoff_month        = unique_months_sorted[cutoff_idx]

    test_mask           = month_targets >= cutoff_month
    month_targets_test  = month_targets[test_mask]

    print(f"\n  Cutoff month : {cutoff_month}")
    print(f"  Test sequences: {test_mask.sum()}")

    # ── standard temporal split (for y_test ground truth) ─────────────────────
    X_train_raw, y_train, X_test_raw, y_test = temporal_split(
        X, y, region_ids, month_targets, split_ratio=SPLIT_RATIO
    )
    # X_test_raw is UN-normalized — required for persistence signal

    # ── persistence prediction ─────────────────────────────────────────────────
    # X_test_raw[:, -1, DISAPPEARANCES_IDX] = disappearances count at month t
    # (the last of the 6 lookback months, i.e. the most recent observed month)
    disappearances_at_t = X_test_raw[:, -1, DISAPPEARANCES_IDX]   # raw counts
    y_pred_binary       = (disappearances_at_t > 0).astype(float)  # binarise

    print(f"\n  Persistence signal stats:")
    print(f"    Sequences where disappearances > 0 at t : {y_pred_binary.sum():.0f} / {len(y_pred_binary)}")
    print(f"    Predicted positive rate                 : {y_pred_binary.mean()*100:.1f}%")
    print(f"    Actual positive rate (y_test)           : {y_test.mean()*100:.1f}%")

    # ── metrics ────────────────────────────────────────────────────────────────
    # NOTE: AUPRC / AUROC use the binary predictions as scores (no probabilities).
    # This is the standard persistence baseline — the "score" IS the binary prediction.
    auprc = average_precision_score(y_test, y_pred_binary)
    auroc = roc_auc_score(y_test, y_pred_binary)
    brier = brier_score_loss(y_test, y_pred_binary)

    # Breakdown by month range
    n_test_months = len(np.unique(month_targets_test))

    print(f"\n{'='*70}")
    print(f"  PERSISTENCE BASELINE RESULTS")
    print(f"{'='*70}")
    print(f"  Test AUPRC : {auprc:.4f}   ← Table 3 number")
    print(f"  Test AUROC : {auroc:.4f}")
    print(f"  Test Brier : {brier:.4f}")
    print(f"  Test months: {month_targets_test.min()} → {month_targets_test.max()} ({n_test_months} months)")

    print(f"\n  Reference points:")
    print(f"    Random Baseline AUPRC  : 0.483")
    print(f"    LSTM (paper)    AUPRC  : 0.741")
    print(f"    Persistence     AUPRC  : {auprc:.3f}  ← where does it sit?")

    # ── print Table 3 summary ──────────────────────────────────────────────────
    print("\n" + "="*70)
    print("  TABLE 3 — Updated with Persistence row")
    print("="*70)
    hdr = f"{'Model':<50} {'Test AUPRC':>12} {'Train AUPRC':>12} {'Gap':>8} {'Brier':>8}"
    print(f"\n{hdr}")
    print("-" * 95)

    ref_rows = [
        ("Attention-LSTM (Full, 16 feat) [paper]",  0.741, 0.705, -0.036, 0.208),
        ("Standard LSTM  (Full, 16 feat) [paper]",  0.740, 0.701, -0.039, 0.206),
        ("Tuned LR — Events Only         [exp1]",   0.743, 0.683, +0.060, 0.206),
        ("Tuned LR — Full Model          [exp1]",   0.738, 0.689, +0.049, 0.206),
        ("XGBoost — Events Only          [exp2]",   0.733, 0.741, -0.008, 0.205),
        ("XGBoost — Full Model           [exp2]",   0.723, 0.704, +0.018, 0.212),
        ("Logistic Regression (untuned)  [paper]",  0.717, 0.626, -0.091, 0.212),
        ("Random Baseline                [paper]",  0.483,  None,   None,  None),
    ]
    for name, ta, tr, g, b in ref_rows:
        tr_s = f"{tr:.3f}" if tr is not None else "     —"
        g_s  = f"{g:+.3f}" if g  is not None else "     —"
        b_s  = f"{b:.3f}" if b  is not None else "     —"
        print(f"{name:<50} {ta:>12.3f} {tr_s:>12} {g_s:>8} {b_s:>8}")

    print("-" * 95)
    print(
        f"{'Persistence Baseline            [exp3]':<50} "
        f"{auprc:>12.3f} "
        f"{'     —':>12} "
        f"{'     —':>8} "
        f"{brier:>8.3f}"
    )
    print("-" * 95)

    # ── save JSON ──────────────────────────────────────────────────────────────
    output = {
        "experiment": "Exp3 — Persistence Baseline",
        "description": (
            "ŷ(t+1) = (disappearances_at_t > 0). "
            "No model training. Uses last timestep of UN-normalized input sequences. "
            "Same temporal split as paper (train≤2022-12, test≥2023-01)."
        ),
        "method": "y_pred = (X_test_raw[:, -1, 2] > 0).astype(int)",
        "results": {
            "test_auprc":          round(float(auprc), 4),
            "test_auroc":          round(float(auroc), 4),
            "test_brier":          round(float(brier), 4),
            "n_test_sequences":    int(len(y_test)),
            "n_test_months":       int(n_test_months),
            "predicted_pos_rate":  round(float(y_pred_binary.mean()), 4),
            "actual_pos_rate":     round(float(y_test.mean()), 4),
        },
    }

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n✅  Results saved to: {RESULTS_PATH}")
    print("\n" + "="*70)
    print("  DONE — Experiment 3 complete")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
