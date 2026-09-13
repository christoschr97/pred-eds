"""
Experiment 6 — Supplementary Poisson Regression on Monthly Counts
====================================================================
Reviewer 3 asked whether the binary framing is the right modeling choice
given the recent shift toward count/regression-based conflict forecasting
(e.g. the ViEWS fatality-count challenge). This experiment does not replace
the binary early-warning classifier; it adds a bounded supplementary
analysis regressing the actual monthly disappearance COUNT in the target
region-month (rather than its binarized occurrence) on the same feature
set, to show the classifier's binary framing is not concealing a materially
different count-level signal.

Setup:
- Input: 6-month lookback window FLATTENED to a single vector
  (identical construction to exp1_tuned_lr.py)
- Same temporal train/test split as paper (train ≤ 2022-12, test ≥ 2023-01)
- Target: raw disappearance COUNT in the target region-month (not the
  binarized >0 threshold used by the main classifier). Recomputed
  independently from the raw ACLED event table (see build_count_lookup)
  rather than reusing data_prep.py's internal 'target' column, because
  create_monthly_aggregation_with_networks() drops the last calendar month
  per region before returning (its target is undefined for that month) --
  exactly the month whose count is needed as the outcome for the
  second-to-last row's sequence. A cross-check below confirms
  (count > 0) reproduces the same binary target used in the paper.
- Model: sklearn PoissonRegressor (L2-penalized GLM with log link),
  alpha tuned via GridSearchCV(TimeSeriesSplit(5), scoring=
  'neg_mean_poisson_deviance') -- the count-regression analogue of the
  AUPRC-scored tuning already used for the LR baseline in exp1.

Metrics reported (per co-author-agreed scope, citing the metrics
framework in Correndo et al. 2022, JOSS, 10.21105/joss.04655):
  - MAE, RMSE
  - Mean Poisson deviance
  - McFadden pseudo-R^2 (vs. an intercept-only null model fit on train)
  - Lin's concordance correlation coefficient (CCC)

Run on:
  - Full Model    : all 16 features -> 6 x 16 = 96 input dims
  - Events Only   : 4 event features -> 6 x 4  = 24 input dims

Output: results/exp6_poisson_regression.json
"""

import sys
import os
import json
import time
import numpy as np
import pandas as pd

# ── path plumbing ─────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_prep import (
    DATA_FILE,
    load_acled_data,
    create_monthly_aggregation_with_networks,
    create_temporal_sequences,
    temporal_split,
    normalize_sequences,
)

from sklearn.linear_model import PoissonRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_poisson_deviance
from scipy.stats import poisson as poisson_dist

# ── config ────────────────────────────────────────────────────────────────────
DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data",
    DATA_FILE,
)

SEQUENCE_LENGTH = 6
SPLIT_RATIO     = 0.7

PARAM_GRID = {
    "alpha": [0.001, 0.01, 0.1, 1, 10, 100],
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
    "exp6_poisson_regression.json",
)


# ── helpers ───────────────────────────────────────────────────────────────────

def flatten(X: np.ndarray) -> np.ndarray:
    """[N, seq_len, features] -> [N, seq_len * features]."""
    return X.reshape(X.shape[0], -1)


def select_features(X: np.ndarray, feature_names: list, subset: list) -> np.ndarray:
    """Pick feature columns from X by name."""
    idx = [feature_names.index(f) for f in subset]
    return X[:, :, idx]


def build_count_lookup(df: pd.DataFrame) -> dict:
    """
    (country, admin1, month) -> disappearance count IN that month.

    Keyed on the month itself, with no shift. The lookup is queried with a
    sequence's `target_month`, and in data_prep.py that field already names
    the month the binary label refers to: the aggregation's 'target' column
    is shifted forward one row at construction, so row j of `targets` holds
    the outcome for row j+1, and create_temporal_sequences pairs
    targets[i+sequence_length-1] with months[i+sequence_length]. Label and
    target_month therefore denote the same month, and the count for that
    month is read directly.

    Applying shift(-1) here as well would advance one further row and put
    the count target a month ahead of the binary target. The sanity check
    in main() catches exactly that: it compares (count > 0) against the
    classifier's own label and aborts below a 99.9% match rate.

    Region timelines can have calendar gaps -- monthly aggregation only
    creates rows for (region, month) combinations with at least one
    recorded event, so a month with zero total events is absent rather
    than a zero-row. Gaps are harmless under direct keying: every
    target_month comes from an aggregated row, and that row is present in
    the grouping below by construction. Unmatched keys default to a count
    of 0 in counts_for_sequences, which cannot arise for target months
    drawn from the same aggregation.
    """
    monthly_counts = (
        df.groupby(["country", "admin1", df["event_date"].dt.to_period("M")])
        ["is_disappearance"].sum()
        .rename("disappearances")
        .reset_index()
        .rename(columns={"event_date": "month"})
        .sort_values(["country", "admin1", "month"])
    )
    return {
        (row.country, row.admin1, row.month): row.disappearances
        for row in monthly_counts.itertuples(index=False)
    }


def counts_for_sequences(
    region_ids: np.ndarray,
    month_targets: np.ndarray,
    regions: list,
    count_lookup: dict,
) -> np.ndarray:
    """
    Map each sequence's (region, target_month) to its count-level target.
    Keyed directly on target_month, the month the paper's classifier
    label refers to; count_lookup holds that month's own count with no
    further shift. See build_count_lookup.
    """
    counts = np.zeros(len(region_ids), dtype=float)
    for i, (rid, tmonth) in enumerate(zip(region_ids, month_targets)):
        country, admin1 = regions[rid]
        counts[i] = count_lookup.get((country, admin1, tmonth), 0)
    return counts


def mcfadden_pseudo_r2(y_true: np.ndarray, mu_model: np.ndarray, mu_null: np.ndarray) -> float:
    """McFadden pseudo-R^2 = 1 - LL_model / LL_null (Poisson log-likelihoods)."""
    ll_model = poisson_dist.logpmf(y_true, np.clip(mu_model, 1e-8, None)).sum()
    ll_null  = poisson_dist.logpmf(y_true, np.clip(mu_null,  1e-8, None)).sum()
    return 1.0 - (ll_model / ll_null)


def lins_ccc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Lin's concordance correlation coefficient."""
    mean_true, mean_pred = y_true.mean(), y_pred.mean()
    var_true,  var_pred  = y_true.var(),  y_pred.var()
    covariance = np.mean((y_true - mean_true) * (y_pred - mean_pred))
    return (2 * covariance) / (var_true + var_pred + (mean_true - mean_pred) ** 2)


def run_poisson(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_label: str,
) -> dict:
    """
    Fit an L2-penalized Poisson GLM with GridSearchCV (TimeSeriesSplit) on
    already-normalised, already-flattened data and return metrics.
    """
    print(f"\n{'='*70}")
    print(f"  Poisson Regression — {feature_label}")
    print(f"  Train: {X_train.shape}  |  Test: {X_test.shape}")
    print(f"  Train mean count: {y_train.mean():.4f}  |  Test mean count: {y_test.mean():.4f}")
    print(f"{'='*70}")

    tscv = TimeSeriesSplit(n_splits=5)

    base = PoissonRegressor(max_iter=1000)

    grid = GridSearchCV(
        estimator=base,
        param_grid=PARAM_GRID,
        cv=tscv,
        scoring="neg_mean_poisson_deviance",
        n_jobs=1,  # n_jobs=-1 hits a joblib/loky semaphore permission error
        verbose=1,  # in this sandboxed environment; single-process is fine
        refit=True,  # for this grid size (6 candidates x 5 folds).
    )

    t0 = time.time()
    grid.fit(X_train, y_train)
    elapsed = time.time() - t0

    best_params  = grid.best_params_
    best_cv_dev  = -grid.best_score_

    print(f"\n  Best params        : {best_params}")
    print(f"  Best CV deviance   : {best_cv_dev:.4f}")
    print(f"  Tuning time        : {elapsed:.1f}s")

    # ── final evaluation on held-out test set ─────────────────────────────────
    best_model = grid.best_estimator_

    mu_test  = np.clip(best_model.predict(X_test),  1e-8, None)
    mu_train = np.clip(best_model.predict(X_train), 1e-8, None)

    # Null model: intercept-only Poisson (constant rate = train mean)
    null_rate     = y_train.mean()
    mu_null_test  = np.full_like(y_test,  null_rate, dtype=float)
    mu_null_train = np.full_like(y_train, null_rate, dtype=float)

    test_mae      = mean_absolute_error(y_test, mu_test)
    test_rmse      = np.sqrt(mean_squared_error(y_test, mu_test))
    test_deviance = mean_poisson_deviance(y_test, mu_test)
    test_pseudo_r2 = mcfadden_pseudo_r2(y_test, mu_test, mu_null_test)
    test_ccc      = lins_ccc(y_test, mu_test)

    train_mae     = mean_absolute_error(y_train, mu_train)
    train_deviance = mean_poisson_deviance(y_train, mu_train)
    train_pseudo_r2 = mcfadden_pseudo_r2(y_train, mu_train, mu_null_train)

    print(f"\n  Test  MAE         : {test_mae:.4f}")
    print(f"  Test  RMSE        : {test_rmse:.4f}")
    print(f"  Test  Deviance    : {test_deviance:.4f}")
    print(f"  Test  McFadden R2 : {test_pseudo_r2:.4f}")
    print(f"  Test  Lin's CCC   : {test_ccc:.4f}")
    print(f"  Train MAE         : {train_mae:.4f}")
    print(f"  Train Deviance    : {train_deviance:.4f}")
    print(f"  Train McFadden R2 : {train_pseudo_r2:.4f}")

    return {
        "feature_set":       feature_label,
        "best_params":       best_params,
        "cv_deviance":       round(best_cv_dev, 4),
        "test_mae":          round(test_mae, 4),
        "test_rmse":         round(test_rmse, 4),
        "test_deviance":     round(test_deviance, 4),
        "test_mcfadden_r2":  round(test_pseudo_r2, 4),
        "test_lins_ccc":     round(test_ccc, 4),
        "train_mae":         round(train_mae, 4),
        "train_deviance":    round(train_deviance, 4),
        "train_mcfadden_r2": round(train_pseudo_r2, 4),
        "tuning_seconds":    round(elapsed, 1),
    }


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "="*70)
    print("  EXPERIMENT 6 — Supplementary Poisson Regression on Monthly Counts")
    print("  Reviewer 3: does binary framing conceal count-level signal?")
    print("="*70)

    print(f"\nData path: {DATA_PATH}")
    df         = load_acled_data(DATA_PATH)
    monthly_df = create_monthly_aggregation_with_networks(df)

    # ── build full 16-feature sequences (identical to exp1/paper pipeline) ───
    X, y_binary, region_ids, regions, month_targets = create_temporal_sequences(
        monthly_df, sequence_length=SEQUENCE_LENGTH
    )

    # ── build raw count targets for the same sequences ────────────────────────
    count_lookup = build_count_lookup(df)
    y_count = counts_for_sequences(region_ids, month_targets, regions, count_lookup)

    # sanity check: (count > 0) must reproduce the binary target used by the
    # paper's classifier -- if this fails, the count lookup is misaligned.
    reconstructed_binary = (y_count > 0).astype(float)
    mismatch = (reconstructed_binary != y_binary).sum()
    match_rate = 1.0 - mismatch / len(y_binary)
    print(f"\n  Sanity check: (count>0) vs. paper's binary target — "
          f"match rate {match_rate*100:.2f}% ({mismatch} mismatches / {len(y_binary)})")
    assert match_rate > 0.999, "Count-target lookup misaligned with binary target — aborting."

    # temporal split (same cutoff as classifier; y_count replaces y_binary)
    X_train_raw, y_train, X_test_raw, y_test = temporal_split(
        X, y_count, region_ids, month_targets, split_ratio=SPLIT_RATIO
    )

    # normalise — fit on train only
    X_train_norm, X_test_norm, _ = normalize_sequences(X_train_raw, X_test_raw)

    results = []

    # ── A: Full Model (16 features) ────────────────────────────────────────────
    X_train_full = flatten(X_train_norm)
    X_test_full  = flatten(X_test_norm)

    res_full = run_poisson(
        X_train_full, y_train,
        X_test_full,  y_test,
        feature_label="Full Model (16 feat, 96-dim flattened)",
    )
    results.append(res_full)

    # ── B: Events Only (4 features) ───────────────────────────────────────────
    X_train_ev = flatten(select_features(X_train_norm, ALL_FEATURES, EVENT_FEATURES))
    X_test_ev  = flatten(select_features(X_test_norm,  ALL_FEATURES, EVENT_FEATURES))

    res_ev = run_poisson(
        X_train_ev, y_train,
        X_test_ev,  y_test,
        feature_label="Events Only (4 feat, 24-dim flattened)",
    )
    results.append(res_ev)

    # ── print summary table ───────────────────────────────────────────────────
    print("\n" + "="*70)
    print("  SUPPLEMENTARY TABLE — Poisson regression on monthly counts")
    print("="*70)
    hdr = f"{'Model':<45} {'MAE':>8} {'RMSE':>8} {'Deviance':>10} {'McFadden R2':>12} {'CCC':>8}"
    print(f"\n{hdr}")
    print("-" * 95)
    for r in results:
        label = f"Poisson GLM — {r['feature_set']}"
        print(
            f"{label:<45} "
            f"{r['test_mae']:>8.3f} "
            f"{r['test_rmse']:>8.3f} "
            f"{r['test_deviance']:>10.3f} "
            f"{r['test_mcfadden_r2']:>12.3f} "
            f"{r['test_lins_ccc']:>8.3f}"
        )
    print("-" * 95)

    # ── save JSON ─────────────────────────────────────────────────────────────
    output = {
        "experiment": "Exp6 — Supplementary Poisson Regression on Monthly Counts",
        "description": (
            "PoissonRegressor (sklearn, L2-penalized GLM, log link) with "
            "GridSearchCV (TimeSeriesSplit n=5, scoring=neg_mean_poisson_deviance). "
            "Flattened 6-month window input, identical construction to exp1. "
            "Target: raw disappearance count in the target region-month "
            "(paper's classifier uses the binarized >0 threshold of this same "
            "quantity). Same temporal split as paper (train<=2022-12, test>=2023-01)."
        ),
        "sanity_check_binary_match_rate": round(match_rate, 4),
        "param_grid": PARAM_GRID,
        "results": results,
    }

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n✅  Results saved to: {RESULTS_PATH}")
    print("\n" + "="*70)
    print("  DONE — Experiment 6 complete")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
