"""
Experiment 10 -- Paired bootstrap / permutation test on the LR-vs-LSTM AUPRC
gap, run on the EVENTS-ONLY feature set.

Why this exists
---------------
exp7 answered Reviewer 3's question about the Attention-LSTM's edge over
tuned logistic regression, but it ran on the Full Model (16 features). The
paper's recommended specification is the Events Only model (4 features:
arrests, violence, disappearances, fatalities), which is the configuration
the network-null finding points to. On the five-country sample the
events-only Attention-LSTM reaches 0.8140 test AUPRC against tuned LR's
0.8115 -- a margin of +0.0025 that exp7 never tested.

This script is exp7's design applied to that feature set. Nothing about the
test is changed: same seeds, same LSTM configuration, same GridSearchCV
grid, same B and P, same paired resampling and label-swap null. The only
difference is the feature subset the sequences are built from.

Method (identical to exp7 unless noted)
---------------------------------------
1. Train the Attention-LSTM on the events-only feature set with the
   configuration used for the paper's recommended model (AttentionLSTM,
   hidden_dim=64, num_layers=2, dropout=0.2, 50 epochs, batch_size=64,
   seeds 0/1/2, split_ratio=0.7) -- identical to run_ablations.py's
   'events_only' ablation -- and capture raw test-set predicted
   probabilities per seed.
2. Train tuned logistic regression on the same events-only sequences with
   exp1's GridSearchCV configuration and capture its test probabilities.
3. Average the 3 seeds' probabilities into a single ensemble prediction,
   as exp7 does.
4. Paired bootstrap: resample the shared test set with replacement
   B=10,000 times, compute the AUPRC difference on each resample, report
   the mean, the 2.5/97.5 percentile CI, and P(difference <= 0).
5. Paired permutation test: swap which model's prediction is used per test
   point with probability 0.5 across P=10,000 permutations, and report the
   two-sided p-value of the observed difference against that null.

exp7 reported its headline difference against the ensemble AUPRC while
recording the seed mean alongside it, which made the two hard to reconcile.
This script reports BOTH differences explicitly and labels them.

Run on the same temporal split as the rest of the paper (train through
2022-12, test from 2023-01) using the pipeline in data_prep.py. No new
data, no new splits, no change to model or evaluation code.
"""

import json
import os
import sys
import time

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_prep import (
    DATA_FILE,
    load_acled_data,
    create_monthly_aggregation_with_networks,
    temporal_split,
    normalize_sequences,
)
from run_ablations import create_sequences_with_feature_subset
from models import AttentionLSTM, train_lstm, evaluate_lstm

DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data",
    DATA_FILE,
)
RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "results",
)
FIGURES_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "paper",
    "figures",
)

EVENT_FEATURES = ["arrests", "violence", "disappearances", "fatalities"]
SEQUENCE_LENGTH = 6
SPLIT_RATIO = 0.7
SEEDS = [0, 1, 2]
NUM_EPOCHS = 50
N_BOOTSTRAP = 10000
N_PERMUTATION = 10000
RNG_SEED = 12345  # for the resampling/permutation draws themselves


def get_events_only_data():
    """Identical construction to run_ablations.py's 'events_only' config."""
    df = load_acled_data(DATA_PATH)
    monthly_df = create_monthly_aggregation_with_networks(df)
    X, y, region_ids, regions, month_targets = create_sequences_with_feature_subset(
        monthly_df,
        sequence_length=SEQUENCE_LENGTH,
        feature_subset=EVENT_FEATURES,
    )
    X_train, y_train, X_test, y_test = temporal_split(
        X, y, region_ids, month_targets, split_ratio=SPLIT_RATIO
    )
    X_train_n, X_test_n, scaler = normalize_sequences(X_train, X_test)
    return X_train_n, y_train, X_test_n, y_test


def train_lstm_seed(X_train, y_train, X_test, y_test, seed):
    """Train one Attention-LSTM seed, matching run_ablations.py's config."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    input_dim = X_train.shape[2]
    model = AttentionLSTM(input_dim=input_dim, hidden_dim=64, num_layers=2, dropout=0.2)
    pos_weight = (1 - y_train.mean()) / y_train.mean()

    train_result = train_lstm(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        num_epochs=NUM_EPOCHS,
        batch_size=64,
        pos_weight=pos_weight,
        device="cpu",
        verbose=False,
    )
    metrics = evaluate_lstm(
        model=train_result["model"],
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        device="cpu",
    )

    model = train_result["model"]
    model.eval()
    with torch.no_grad():
        test_logits = model(torch.FloatTensor(X_test))
        test_probs = torch.sigmoid(test_logits).cpu().numpy().ravel()

    return test_probs, metrics["test_auprc"]


def train_tuned_lr(X_train, y_train, X_test, y_test):
    """Identical GridSearchCV configuration to exp1_tuned_lr.py."""
    n_train = X_train.shape[0]
    X_train_flat = X_train.reshape(n_train, -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)

    pos_weight = (1 - y_train.mean()) / y_train.mean()
    class_weight = {0: 1.0, 1: pos_weight}

    param_grid = {
        "C": [0.001, 0.01, 0.1, 1, 10, 100],
        "penalty": ["l1", "l2"],
        "solver": ["liblinear", "saga"],
    }
    tscv = TimeSeriesSplit(n_splits=5)
    base = LogisticRegression(class_weight=class_weight, max_iter=5000, random_state=42)
    grid = GridSearchCV(
        estimator=base,
        param_grid=param_grid,
        cv=tscv,
        scoring="average_precision",
        n_jobs=1,
        refit=True,
    )
    grid.fit(X_train_flat, y_train)
    best = grid.best_estimator_
    test_probs = best.predict_proba(X_test_flat)[:, 1]
    test_auprc = average_precision_score(y_test, test_probs)
    return test_probs, test_auprc, grid.best_params_


def paired_bootstrap(y_test, p_lstm, p_lr, n_boot, rng):
    n = len(y_test)
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        y_b = y_test[idx]
        if y_b.min() == y_b.max():
            diffs[b] = np.nan
            continue
        auprc_lstm = average_precision_score(y_b, p_lstm[idx])
        auprc_lr = average_precision_score(y_b, p_lr[idx])
        diffs[b] = auprc_lstm - auprc_lr
    return diffs


def paired_permutation(y_test, p_lstm, p_lr, n_perm, rng):
    n = len(y_test)
    observed = average_precision_score(y_test, p_lstm) - average_precision_score(y_test, p_lr)
    null_diffs = np.empty(n_perm)
    for p in range(n_perm):
        swap = rng.random(n) < 0.5
        p_a = np.where(swap, p_lr, p_lstm)
        p_b = np.where(swap, p_lstm, p_lr)
        null_diffs[p] = average_precision_score(y_test, p_a) - average_precision_score(y_test, p_b)
    return observed, null_diffs


def main():
    print("\n" + "=" * 70)
    print("  EXPERIMENT 10 -- Paired Bootstrap/Permutation: LSTM vs. Tuned LR")
    print("  EVENTS-ONLY feature set (the paper's recommended model)")
    print("=" * 70 + "\n")

    print(f"Data path: {DATA_PATH}")
    print(f"Feature set ({len(EVENT_FEATURES)}): {', '.join(EVENT_FEATURES)}\n")
    X_train, y_train, X_test, y_test = get_events_only_data()
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"Train positive rate: {y_train.mean()*100:.1f}%  "
          f"Test positive rate: {y_test.mean()*100:.1f}%\n")

    # --- Train LSTM seeds ---
    lstm_probs = []
    lstm_auprcs = []
    for seed in SEEDS:
        t0 = time.time()
        probs, auprc = train_lstm_seed(X_train, y_train, X_test, y_test, seed)
        lstm_probs.append(probs)
        lstm_auprcs.append(auprc)
        print(f"  Seed {seed}: test AUPRC = {auprc:.4f}  ({time.time()-t0:.1f}s)")

    lstm_auprcs = np.array(lstm_auprcs)
    seed_mean_auprc = float(lstm_auprcs.mean())
    print(f"\n  Attention-LSTM seed mean: {seed_mean_auprc:.4f} "
          f"+/- {lstm_auprcs.std():.4f}")

    p_lstm_ensemble = np.mean(np.stack(lstm_probs, axis=0), axis=0)
    ensemble_auprc = average_precision_score(y_test, p_lstm_ensemble)
    print(f"  Attention-LSTM ensemble (mean of 3 seeds' probs): {ensemble_auprc:.4f}")

    # --- Train tuned LR ---
    print("\n  Tuning logistic regression (GridSearchCV, TimeSeriesSplit n=5)...")
    p_lr, lr_auprc, lr_best_params = train_tuned_lr(X_train, y_train, X_test, y_test)
    print(f"  Tuned LR: test AUPRC = {lr_auprc:.4f}")
    print(f"  Best params: {lr_best_params}")

    # Both differences, explicitly labelled -- the bootstrap and permutation
    # tests below are run on the ensemble prediction, so the ensemble
    # difference is the one the p-values refer to.
    diff_ensemble = ensemble_auprc - lr_auprc
    diff_seed_mean = seed_mean_auprc - lr_auprc
    print(f"\n  Difference, ensemble  - LR: {diff_ensemble:+.4f}   <- tested below")
    print(f"  Difference, seed mean - LR: {diff_seed_mean:+.4f}")

    # --- Paired bootstrap ---
    print(f"\n  Running paired bootstrap (B={N_BOOTSTRAP})...")
    rng = np.random.default_rng(RNG_SEED)
    boot_diffs = paired_bootstrap(y_test, p_lstm_ensemble, p_lr, N_BOOTSTRAP, rng)
    boot_diffs = boot_diffs[~np.isnan(boot_diffs)]
    ci_lo, ci_hi = np.percentile(boot_diffs, [2.5, 97.5])
    boot_p_le0 = float(np.mean(boot_diffs <= 0))
    print(f"    Mean diff: {boot_diffs.mean():+.4f}")
    print(f"    95% CI: [{ci_lo:+.4f}, {ci_hi:+.4f}]")
    print(f"    P(diff <= 0): {boot_p_le0:.4f}")

    # --- Paired permutation test ---
    print(f"\n  Running paired permutation test (P={N_PERMUTATION})...")
    rng2 = np.random.default_rng(RNG_SEED + 1)
    obs_diff_perm, null_diffs = paired_permutation(
        y_test, p_lstm_ensemble, p_lr, N_PERMUTATION, rng2
    )
    perm_p_two_sided = float(np.mean(np.abs(null_diffs) >= np.abs(obs_diff_perm)))
    print(f"    Observed diff: {obs_diff_perm:+.4f}")
    print(f"    Null distribution: mean={null_diffs.mean():+.4f}, sd={null_diffs.std():.4f}")
    print(f"    Two-sided p-value: {perm_p_two_sided:.4f}")

    # --- Verdict ---
    ci_excludes_zero = (ci_lo > 0) or (ci_hi < 0)
    print("\n  " + "-" * 66)
    print(f"  95% CI {'EXCLUDES' if ci_excludes_zero else 'INCLUDES'} zero "
          f"-> the events-only LSTM/LR difference is "
          f"{'distinguishable' if ci_excludes_zero else 'not distinguishable'} "
          f"from noise.")
    print("  " + "-" * 66)

    # --- Save results ---
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    results = {
        "experiment": "Exp10 -- Paired Bootstrap/Permutation: LSTM vs. Tuned LR, events-only",
        "description": (
            "exp7's design applied to the EVENTS-ONLY feature set (4 features: "
            f"{', '.join(EVENT_FEATURES)}), which is the paper's recommended "
            "specification. Attention-LSTM (3 seeds, ensemble-averaged test "
            "probabilities) vs. tuned logistic regression on the identical "
            f"test set (n={len(y_test)}, 2023-01 onward). Paired bootstrap "
            f"(B={N_BOOTSTRAP}, resampling test sequences with replacement) "
            f"and paired permutation test (P={N_PERMUTATION}, randomly "
            "swapping which model's prediction is used per test sequence) on "
            "the AUPRC difference. The reported p-values refer to the "
            "ensemble difference."
        ),
        "feature_set": EVENT_FEATURES,
        "n_test": int(len(y_test)),
        "test_positive_rate": float(y_test.mean()),
        "lstm_seed_auprcs": lstm_auprcs.tolist(),
        "lstm_mean_auprc": seed_mean_auprc,
        "lstm_std_auprc": float(lstm_auprcs.std()),
        "lstm_ensemble_auprc": float(ensemble_auprc),
        "lr_auprc": float(lr_auprc),
        "lr_best_params": lr_best_params,
        "diff_ensemble_minus_lr": float(diff_ensemble),
        "diff_seed_mean_minus_lr": float(diff_seed_mean),
        "bootstrap": {
            "n_boot": int(len(boot_diffs)),
            "mean_diff": float(boot_diffs.mean()),
            "ci95_lo": float(ci_lo),
            "ci95_hi": float(ci_hi),
            "p_diff_le_0": boot_p_le0,
            "ci_excludes_zero": bool(ci_excludes_zero),
        },
        "permutation": {
            "n_perm": N_PERMUTATION,
            "null_mean": float(null_diffs.mean()),
            "null_sd": float(null_diffs.std()),
            "two_sided_p": perm_p_two_sided,
        },
    }

    out_path = os.path.join(RESULTS_DIR, "exp10_bootstrap_events_only.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved results: {out_path}")

    # --- Figure: bootstrap distribution ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(boot_diffs, bins=60, color="#4C72B0", alpha=0.85,
            edgecolor="white", linewidth=0.3)
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.axvline(boot_diffs.mean(), color="#C44E52", linestyle="-", linewidth=1.5,
               label=f"Mean diff = {boot_diffs.mean():+.4f}")
    ax.axvspan(ci_lo, ci_hi, color="#C44E52", alpha=0.12,
               label=f"95% CI [{ci_lo:+.3f}, {ci_hi:+.3f}]")
    ax.set_xlabel("Bootstrap AUPRC difference (LSTM ensemble $-$ tuned LR)")
    ax.set_ylabel("Count")
    ax.set_title(f"Events-only model, paired bootstrap, B={len(boot_diffs)}")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, "exp10_bootstrap_events_only.png")
    fig.savefig(fig_path, dpi=200)
    print(f"  Saved figure: {fig_path}")

    print("\n" + "=" * 70)
    print("  DONE -- Experiment 10 complete")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
