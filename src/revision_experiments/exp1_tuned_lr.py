"""
Experiment 1 — Tuned Logistic Regression Baseline
===================================================
Reviewer response: current LR baseline is untuned (default sklearn), making
the AUPRC gap over LSTM uninterpretable. This tunes it properly.

Setup:
- Input: 6-month lookback window FLATTENED to a single vector
  (same as existing LR: 6 timesteps × F features)
- Same temporal train/test split as paper (train ≤ 2022-12, test ≥ 2023-01)
- Same class weighting: {0: 1.0, 1: 1.683}
- GridSearchCV with TimeSeriesSplit(n_splits=5), scoring=average_precision
- Grid: C=[0.001,0.01,0.1,1,10,100], penalty=[l1,l2], solver=[liblinear,saga]

Run on:
  - Full Model    : all 16 features → 6 × 16 = 96 input dims
  - Events Only   : 4 event features → 6 × 4  = 24 input dims

Output: results/exp1_tuned_lr.json  (ready to slot into Table 3)
"""

import sys
import os
import json
import time
import numpy as np

# ── path plumbing ─────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_prep import (
    load_acled_data,
    create_monthly_aggregation_with_networks,
    create_temporal_sequences,
    temporal_split,
    normalize_sequences,
)

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss

# ── config ────────────────────────────────────────────────────────────────────
DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data",
    "ACLED Data_2025-12-31_Nigeria_Mexico_Myanmar.csv",
)

SEQUENCE_LENGTH = 6
SPLIT_RATIO     = 0.7
CLASS_WEIGHT    = {0: 1.0, 1: 1.683}   # matches paper §3.6 exactly

PARAM_GRID = {
    "C":       [0.001, 0.01, 0.1, 1, 10, 100],
    "penalty": ["l1", "l2"],
    "solver":  ["liblinear", "saga"],
}

ALL_FEATURES = [
    "arrests", "violence", "disappearances", "fatalities",
    "num_actors", "num_edges", "density", "avg_degree",
    "max_degree", "num_components", "largest_component_size",
    "avg_clustering", "transitivity",
    "avg_edge_weight", "max_edge_weight",
    "degree_centralization",
]
EVENT_FEATURES = ["arrests", "violence", "disappearances", "fatalities"]

RESULTS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "results",
    "exp1_tuned_lr.json",
)


# ── helpers ───────────────────────────────────────────────────────────────────

def flatten(X: np.ndarray) -> np.ndarray:
    """[N, seq_len, features] → [N, seq_len * features]."""
    return X.reshape(X.shape[0], -1)


def select_features(X: np.ndarray, feature_names: list, subset: list) -> np.ndarray:
    """Pick feature columns from X by name."""
    idx = [feature_names.index(f) for f in subset]
    return X[:, :, idx]


def run_tuned_lr(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_label: str,
) -> dict:
    """
    Fit a tuned Logistic Regression with GridSearchCV (TimeSeriesSplit)
    on already-normalised, already-flattened data and return metrics.
    """
    print(f"\n{'='*70}")
    print(f"  Tuned LR — {feature_label}")
    print(f"  Train: {X_train.shape}  |  Test: {X_test.shape}")
    print(f"  Train positive rate: {y_train.mean()*100:.1f}%")
    print(f"  Test  positive rate: {y_test.mean()*100:.1f}%")
    print(f"{'='*70}")

    tscv = TimeSeriesSplit(n_splits=5)

    base = LogisticRegression(
        class_weight=CLASS_WEIGHT,
        max_iter=2000,
        random_state=42,
    )

    grid = GridSearchCV(
        estimator=base,
        param_grid=PARAM_GRID,
        cv=tscv,
        scoring="average_precision",
        n_jobs=-1,
        verbose=1,
        refit=True,
    )

    t0 = time.time()
    grid.fit(X_train, y_train)
    elapsed = time.time() - t0

    best_params   = grid.best_params_
    best_cv_auprc = grid.best_score_

    print(f"\n  Best params   : {best_params}")
    print(f"  Best CV AUPRC : {best_cv_auprc:.4f}")
    print(f"  Tuning time   : {elapsed:.1f}s")

    # ── final evaluation on held-out test set ─────────────────────────────────
    best_model = grid.best_estimator_

    y_train_prob = best_model.predict_proba(X_train)[:, 1]
    y_test_prob  = best_model.predict_proba(X_test)[:, 1]

    train_auprc = average_precision_score(y_train, y_train_prob)
    test_auprc  = average_precision_score(y_test,  y_test_prob)
    test_auroc  = roc_auc_score(y_test, y_test_prob)
    test_brier  = brier_score_loss(y_test, y_test_prob)
    gap         = test_auprc - train_auprc

    print(f"\n  Train AUPRC : {train_auprc:.4f}")
    print(f"  Test  AUPRC : {test_auprc:.4f}   ← Table 3 number")
    print(f"  Test  AUROC : {test_auroc:.4f}")
    print(f"  Test  Brier : {test_brier:.4f}")
    print(f"  Gap (test-train) : {gap:+.4f}")

    return {
        "feature_set":    feature_label,
        "best_params":    best_params,
        "cv_auprc":       round(best_cv_auprc, 4),
        "train_auprc":    round(train_auprc, 4),
        "test_auprc":     round(test_auprc,  4),
        "test_auroc":     round(test_auroc,  4),
        "test_brier":     round(test_brier,  4),
        "gap":            round(gap, 4),
        "tuning_seconds": round(elapsed, 1),
    }


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "="*70)
    print("  EXPERIMENT 1 — Tuned Logistic Regression Baseline")
    print("  Reviewer: current LR is untuned — gap over LSTM uninterpretable")
    print("="*70)

    print(f"\nData path: {DATA_PATH}")
    df         = load_acled_data(DATA_PATH)
    monthly_df = create_monthly_aggregation_with_networks(df)

    # ── build full 16-feature sequences ───────────────────────────────────────
    X, y, region_ids, regions, month_targets = create_temporal_sequences(
        monthly_df, sequence_length=SEQUENCE_LENGTH
    )

    # temporal split
    X_train_raw, y_train, X_test_raw, y_test = temporal_split(
        X, y, region_ids, month_targets, split_ratio=SPLIT_RATIO
    )

    # normalise — fit on train only
    X_train_norm, X_test_norm, _ = normalize_sequences(X_train_raw, X_test_raw)

    results = []

    # ── A: Full Model (16 features) ────────────────────────────────────────────
    X_train_full = flatten(X_train_norm)
    X_test_full  = flatten(X_test_norm)

    res_full = run_tuned_lr(
        X_train_full, y_train,
        X_test_full,  y_test,
        feature_label="Full Model (16 feat, 96-dim flattened)",
    )
    results.append(res_full)

    # ── B: Events Only (4 features) ───────────────────────────────────────────
    X_train_ev = flatten(select_features(X_train_norm, ALL_FEATURES, EVENT_FEATURES))
    X_test_ev  = flatten(select_features(X_test_norm,  ALL_FEATURES, EVENT_FEATURES))

    res_ev = run_tuned_lr(
        X_train_ev, y_train,
        X_test_ev,  y_test,
        feature_label="Events Only (4 feat, 24-dim flattened)",
    )
    results.append(res_ev)

    # ── print Table 3 summary ─────────────────────────────────────────────────
    print("\n" + "="*70)
    print("  TABLE 3 ADDITION — Tuned LR rows")
    print("="*70)
    hdr = f"{'Model':<50} {'Test AUPRC':>12} {'Train AUPRC':>12} {'Gap':>8} {'Brier':>8}"
    print(f"\n{hdr}")
    print("-" * 95)

    ref_rows = [
        ("Attention-LSTM (Full, 16 feat) [paper]",  0.741, 0.705, -0.036, 0.208),
        ("Standard LSTM  (Full, 16 feat) [paper]",  0.740, 0.701, -0.039, 0.206),
        ("XGBoost — Full Model           [exp2]",   0.723, 0.704, +0.018, 0.212),
        ("XGBoost — Events Only          [exp2]",   0.733, 0.741, -0.008, 0.205),
        ("Logistic Regression (untuned)  [paper]",  0.717, 0.626, -0.091, 0.212),
        ("Random Baseline                [paper]",  0.483,  None,   None,  None),
    ]
    for name, ta, tr, g, b in ref_rows:
        tr_s = f"{tr:.3f}" if tr is not None else "     —"
        g_s  = f"{g:+.3f}" if g  is not None else "     —"
        b_s  = f"{b:.3f}" if b  is not None else "     —"
        print(f"{name:<50} {ta:>12.3f} {tr_s:>12} {g_s:>8} {b_s:>8}")

    print("-" * 95)
    for r in results:
        label = f"Tuned LR — {r['feature_set']}"
        g_s   = f"{r['gap']:+.3f}"
        print(
            f"{label:<50} "
            f"{r['test_auprc']:>12.3f} "
            f"{r['train_auprc']:>12.3f} "
            f"{g_s:>8} "
            f"{r['test_brier']:>8.3f}"
        )
    print("-" * 95)

    # ── save JSON ─────────────────────────────────────────────────────────────
    output = {
        "experiment": "Exp1 — Tuned Logistic Regression Baseline",
        "description": (
            "LogisticRegression with GridSearchCV (TimeSeriesSplit n=5). "
            "Flattened 6-month window input. class_weight={0:1.0, 1:1.683}. "
            "Same temporal split as paper (train≤2022-12, test≥2023-01)."
        ),
        "param_grid": PARAM_GRID,
        "results": results,
    }

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n✅  Results saved to: {RESULTS_PATH}")
    print("\n" + "="*70)
    print("  DONE — Experiment 1 complete")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
