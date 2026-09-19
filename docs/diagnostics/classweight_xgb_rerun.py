"""
Experiment 2 — Gradient Boosted Machine Baseline (XGBoost)
===========================================================
Reviewer response: tests whether LSTM temporal modelling adds value over
a strong non-temporal baseline.

Setup (matches LR baseline in Table 3 exactly):
- Input: 6-month lookback window FLATTENED to a single vector
  (same as LR: 6 timesteps × F features)
- Same temporal train/test split (2018-Jul → 2022-Dec train,
  2023-Jan → 2024-Dec test, ~70/30 by time)
- Same class weighting rule as paper §3.6: w+ = (1 - p) / p, recomputed
  from the training split (1.683 on the three-country sample, 2.213 on five)
- GridSearchCV with TimeSeriesSplit(n_splits=5)

Run on:
  - Full Model    : all 16 features  → 6 × 16 = 96 input dims
  - Events Only   : 4 event features → 6 × 4  = 24 input dims

Output: results/exp2_gbm.json  (ready to slot into Table 3)
"""

import sys
import os
import json
import time
import numpy as np

# ── path plumbing ───────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_prep import (
    DATA_FILE,
    load_acled_data,
    create_monthly_aggregation_with_networks,
    create_temporal_sequences,
    temporal_split,
    normalize_sequences,
)

from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss

# ── config ───────────────────────────────────────────────────────────────────
DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data",
    DATA_FILE,
)

SEQUENCE_LENGTH = 6
SPLIT_RATIO = 0.7
SCALE_POS_WEIGHT = 1.683          # = (1 - 0.373) / 0.373  (matches paper §3.6)
SEEDS = [42, 123, 456]            # same three seeds used in paper

PARAM_GRID = {
    "n_estimators":  [100, 300, 500],
    "max_depth":     [3, 5, 7],
    "learning_rate": [0.01, 0.05, 0.1],
    "subsample":     [0.8, 1.0],
}

# Feature names in the order they appear in the lstm_new_features pipeline
ALL_FEATURES = [
    "arrests", "violence", "disappearances", "fatalities",          # event (4)
    "num_actors", "num_edges", "density", "avg_degree",             # structure (5)
    "max_degree", "num_components", "largest_component_size",
    "avg_clustering", "transitivity",                                # clustering (2)
    "avg_edge_weight", "max_edge_weight",                           # weights (2)
    "degree_centralization",                                         # centrality (1)
]
EVENT_FEATURES = ["arrests", "violence", "disappearances", "fatalities"]

RESULTS_PATH = "/tmp/classweight_diag/exp2_gbm_RERUN.json"


# ── helpers ──────────────────────────────────────────────────────────────────

def flatten(X: np.ndarray) -> np.ndarray:
    """Flatten [N, seq_len, features] → [N, seq_len * features]."""
    return X.reshape(X.shape[0], -1)


def select_features(X: np.ndarray, feature_names: list, subset: list) -> np.ndarray:
    """
    Pick columns from X (shape [N, seq_len, F]) that correspond to `subset`.
    `feature_names` is the ordered list of all F feature names.
    """
    idx = [feature_names.index(f) for f in subset]
    return X[:, :, idx]


def run_gbm_experiment(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_label: str,
) -> dict:
    """
    Fit a LightGBM classifier with GridSearchCV (TimeSeriesSplit) on
    already-normalised, already-flattened data and return metrics.
    """
    print(f"\n{'='*70}")
    print(f"  GBM — {feature_label}")
    print(f"  Train: {X_train.shape}  |  Test: {X_test.shape}")
    print(f"  Train positive rate: {y_train.mean()*100:.1f}%")
    print(f"  Test  positive rate: {y_test.mean()*100:.1f}%")
    print(f"{'='*70}")

    tscv = TimeSeriesSplit(n_splits=5)

    base = XGBClassifier(
        scale_pos_weight=SCALE_POS_WEIGHT,
        random_state=42,
        n_jobs=-1,
        eval_metric="aucpr",
        verbosity=0,
        use_label_encoder=False,
    )

    grid = GridSearchCV(
        estimator=base,
        param_grid=PARAM_GRID,
        cv=tscv,
        scoring="average_precision",   # AUPRC — primary metric in paper
        n_jobs=1,  # n_jobs=-1 hits a joblib/loky semaphore permission error
        verbose=1,
        refit=True,
    )

    t0 = time.time()
    grid.fit(X_train, y_train)
    elapsed = time.time() - t0

    best_params = grid.best_params_
    best_cv_auprc = grid.best_score_

    print(f"\n  Best params : {best_params}")
    print(f"  Best CV AUPRC : {best_cv_auprc:.4f}")
    print(f"  Tuning time : {elapsed:.1f}s")

    # ── final evaluation on held-out test set ─────────────────────────────
    best_model = grid.best_estimator_
    y_train_prob = best_model.predict_proba(X_train)[:, 1]
    y_test_prob  = best_model.predict_proba(X_test)[:, 1]

    train_auprc = average_precision_score(y_train, y_train_prob)
    test_auprc  = average_precision_score(y_test,  y_test_prob)
    test_auroc  = roc_auc_score(y_test, y_test_prob)
    test_brier  = brier_score_loss(y_test, y_test_prob)
    gap         = test_auprc - train_auprc   # negative = over-fit

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
        "test_auprc":     round(test_auprc, 4),
        "test_auroc":     round(test_auroc, 4),
        "test_brier":     round(test_brier, 4),
        "gap":            round(gap, 4),
        "tuning_seconds": round(elapsed, 1),
    }


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "="*70)
    print("  EXPERIMENT 2 — GBM Baseline")
    print("  Reviewer: tests whether LSTM temporal modelling adds value")
    print("  over a strong non-temporal baseline (LightGBM)")
    print("="*70)

    # ── load data once (network features are expensive to compute) ──────────
    print(f"\nData path: {DATA_PATH}")
    df = load_acled_data(DATA_PATH)
    monthly_df = create_monthly_aggregation_with_networks(df)

    # ── build full 16-feature sequences ─────────────────────────────────────
    X, y, region_ids, regions, month_targets = create_temporal_sequences(
        monthly_df, sequence_length=SEQUENCE_LENGTH
    )
    # X shape: [N, 6, 16]

    # temporal split
    X_train_raw, y_train, X_test_raw, y_test = temporal_split(
        X, y, region_ids, month_targets, split_ratio=SPLIT_RATIO
    )

    # normalise — fit on train only (same as paper)
    X_train_norm, X_test_norm, _ = normalize_sequences(X_train_raw, X_test_raw)

    # Class weight follows paper §3.6: w+ = (1 - p) / p on the TRAINING split.
    # The literal 1.683 above was p = 0.373, the three-country training
    # prevalence. On any other sample that constant is wrong, so recompute it
    # here from y_train — this is what run_ablations.py and exp4/exp5/exp7 do.
    global SCALE_POS_WEIGHT
    SCALE_POS_WEIGHT = float((1 - y_train.mean()) / y_train.mean())
    print(f"scale_pos_weight = {SCALE_POS_WEIGHT:.3f} "
          f"(training prevalence {y_train.mean()*100:.1f}%)")

    results = []

    # ── Experiment A: Full Model (16 features) ───────────────────────────────
    X_train_full = flatten(X_train_norm)   # [N_train, 96]
    X_test_full  = flatten(X_test_norm)    # [N_test,  96]

    res_full = run_gbm_experiment(
        X_train_full, y_train,
        X_test_full,  y_test,
        feature_label="Full Model (16 feat, 96-dim flattened)",
    )
    results.append(res_full)

    # ── Experiment B: Events Only (4 features) ───────────────────────────────
    X_train_ev = flatten(select_features(X_train_norm, ALL_FEATURES, EVENT_FEATURES))
    X_test_ev  = flatten(select_features(X_test_norm,  ALL_FEATURES, EVENT_FEATURES))

    res_ev = run_gbm_experiment(
        X_train_ev, y_train,
        X_test_ev,  y_test,
        feature_label="Events Only (4 feat, 24-dim flattened)",
    )
    results.append(res_ev)

    # ── print Table 3 summary ────────────────────────────────────────────────
    print("\n" + "="*70)
    print("  TABLE 3 ADDITION — GBM rows (single run, same split as paper)")
    print("="*70)
    print(f"\n{'Model':<45} {'Test AUPRC':>12} {'Train AUPRC':>12} {'Gap':>8} {'Brier':>8}")
    print("-" * 90)

    # Reference rows from paper
    ref_rows = [
        ("Attention-LSTM (Full, 16 feat) [paper]", 0.741, 0.705, -0.036, 0.208),
        ("Standard LSTM  (Full, 16 feat) [paper]", 0.740, 0.701, -0.039, 0.206),
        ("Logistic Regression            [paper]", 0.717, 0.626, -0.091, 0.212),
        ("Random Baseline                [paper]", 0.483,  None,   None,  None),
    ]
    for name, ta, tr, g, b in ref_rows:
        tr_s  = f"{tr:.3f}" if tr   is not None else "   —  "
        g_s   = f"{g:+.3f}" if g   is not None else "   —  "
        b_s   = f"{b:.3f}" if b   is not None else "   —  "
        print(f"{name:<45} {ta:>12.3f} {tr_s:>12} {g_s:>8} {b_s:>8}")

    print("-" * 90)
    for r in results:
        g_s   = f"{r['gap']:+.3f}"
        label = f"XGBoost — {r['feature_set']}"
        print(
            f"{label:<45} "
            f"{r['test_auprc']:>12.3f} "
            f"{r['train_auprc']:>12.3f} "
            f"{g_s:>8} "
            f"{r['test_brier']:>8.3f}"
        )
    print("-" * 90)

    # ── save JSON ────────────────────────────────────────────────────────────
    output = {
        "experiment": "Exp2 — XGBoost Baseline",
        "description": (
            "XGBoost with GridSearchCV (TimeSeriesSplit n=5). "
            "Flattened 6-month window input. "
            f"scale_pos_weight={SCALE_POS_WEIGHT:.3f} recomputed from the "
            "training split. "
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
    print("  DONE — Experiment 2 complete")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
