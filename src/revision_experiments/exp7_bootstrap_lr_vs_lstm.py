"""
Experiment 7 -- Paired bootstrap / permutation test on the LR-vs-LSTM AUPRC gap.

Reviewer 3 asked whether the Attention-LSTM's edge over tuned logistic
regression (0.741 vs. 0.738 test AUPRC, Table 3) is a real generalization
advantage or noise. This script gives that question a formal statistical
answer.

Method
------
1. Train the Attention-LSTM Full Model with the exact configuration used
   for Table 3 (AttentionLSTM, hidden_dim=64, num_layers=2, dropout=0.2,
   50 epochs, batch_size=64, seeds 0/1/2, split_ratio=0.7) -- identical to
   run_ablations.py's 'full' ablation -- and capture raw test-set predicted
   probabilities for each seed.
2. Train tuned logistic regression on the same Full Model feature set with
   the exact GridSearchCV configuration used in exp1_tuned_lr.py, and
   capture its raw test-set predicted probabilities (LR is deterministic
   given the tuned hyperparameters, so a single run suffices, matching
   Table 3's single-run LR entry).
3. Average the 3 LSTM seeds' predicted probabilities into a single
   ensemble prediction per test sequence (the natural single-model
   summary of "the LSTM's" prediction for a paired comparison against LR).
4. Paired bootstrap: resample the shared test set (n=2035) WITH
   replacement B=10,000 times; on each resample compute AUPRC for the
   LSTM ensemble and for LR, and record the difference. Report the mean
   difference and the 2.5/97.5 percentile 95% CI, plus the fraction of
   resamples with difference <= 0 (a bootstrap-based one-sided p-value
   for "LSTM outperforms LR").
5. Paired permutation test: under the null that the two models' per-region
   predictions are exchangeable, randomly swap which model's prediction
   is used for each test point (independently, with prob 0.5) across
   P=10,000 permutations, recompute the AUPRC difference each time to
   build a null distribution, and report the two-sided p-value of the
   observed difference against that null.

Both tests are run on the SAME shared test set as Table 3 (train through
2022-12, test from 2023-01), using the identical data pipeline in
data_prep.py -- no new data, no new splits, no change to existing model
or evaluation code.
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
    create_temporal_sequences,
    temporal_split,
    normalize_sequences,
)
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

SEEDS = [0, 1, 2]
NUM_EPOCHS = 50
N_BOOTSTRAP = 10000
N_PERMUTATION = 10000
RNG_SEED = 12345  # for the resampling/permutation draws themselves


def get_full_model_data():
    """Identical construction to run_ablations.py's 'full' config / exp1's Full Model."""
    df = load_acled_data(DATA_PATH)
    monthly_df = create_monthly_aggregation_with_networks(df)
    X, y, region_ids, regions, month_targets = create_temporal_sequences(
        monthly_df, sequence_length=6
    )
    X_train, y_train, X_test, y_test = temporal_split(
        X, y, region_ids, month_targets, split_ratio=0.7
    )
    X_train_n, X_test_n, scaler = normalize_sequences(X_train, X_test)
    return X_train_n, y_train, X_test_n, y_test


def train_lstm_seed(X_train, y_train, X_test, y_test, seed):
    """Train one Attention-LSTM seed, exactly matching run_ablations.py's 'full' config."""
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

    # Recompute test probabilities directly (evaluate_lstm reports scalar
    # metrics only; we need the raw per-sequence probabilities for bootstrap).
    model = train_result["model"]
    model.eval()
    with torch.no_grad():
        test_logits = model(torch.FloatTensor(X_test))
        test_probs = torch.sigmoid(test_logits).cpu().numpy().ravel()

    return test_probs, metrics["test_auprc"]


def train_tuned_lr(X_train, y_train, X_test, y_test):
    """Identical GridSearchCV configuration to exp1_tuned_lr.py's Full Model run."""
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
        # Skip degenerate resamples with a single class (AUPRC undefined)
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
    print("  EXPERIMENT 7 -- Paired Bootstrap/Permutation Test: LSTM vs. Tuned LR")
    print("  Reviewer 3: is the LSTM's edge over tuned LR a real effect?")
    print("=" * 70 + "\n")

    print(f"Data path: {DATA_PATH}")
    X_train, y_train, X_test, y_test = get_full_model_data()
    print(f"Train: {X_train.shape}, Test: {X_test.shape}\n")

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
    print(f"\n  Attention-LSTM: {lstm_auprcs.mean():.4f} +/- {lstm_auprcs.std():.4f} "
          f"(paper reports 0.741 +/- 0.003)")

    p_lstm_ensemble = np.mean(np.stack(lstm_probs, axis=0), axis=0)
    ensemble_auprc = average_precision_score(y_test, p_lstm_ensemble)
    print(f"  LSTM ensemble (mean of 3 seeds' probabilities): AUPRC = {ensemble_auprc:.4f}")

    # --- Train tuned LR ---
    print("\n  Tuning logistic regression (GridSearchCV, TimeSeriesSplit n=5)...")
    p_lr, lr_auprc, lr_best_params = train_tuned_lr(X_train, y_train, X_test, y_test)
    print(f"  Tuned LR: test AUPRC = {lr_auprc:.4f}  (paper reports 0.738)")
    print(f"  Best params: {lr_best_params}")

    observed_diff = ensemble_auprc - lr_auprc
    print(f"\n  Observed AUPRC difference (LSTM ensemble - LR): {observed_diff:+.4f}")

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
    obs_diff_perm, null_diffs = paired_permutation(y_test, p_lstm_ensemble, p_lr, N_PERMUTATION, rng2)
    perm_p_two_sided = float(np.mean(np.abs(null_diffs) >= np.abs(obs_diff_perm)))
    print(f"    Observed diff: {obs_diff_perm:+.4f}")
    print(f"    Null distribution: mean={null_diffs.mean():+.4f}, sd={null_diffs.std():.4f}")
    print(f"    Two-sided p-value: {perm_p_two_sided:.4f}")

    # --- Save results ---
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    results = {
        "experiment": "Exp7 -- Paired Bootstrap/Permutation Test: LSTM vs. Tuned LR",
        "description": (
            "Attention-LSTM Full Model (3 seeds, ensemble-averaged test "
            "probabilities) vs. tuned logistic regression Full Model on the "
            "identical test set (n=2035, 2023-01 onward). Paired bootstrap "
            "(B=10000, resampling test sequences with replacement) and paired "
            "permutation test (P=10000, randomly swapping which model's "
            "prediction is used per test sequence) on the AUPRC difference."
        ),
        "lstm_seed_auprcs": lstm_auprcs.tolist(),
        "lstm_mean_auprc": float(lstm_auprcs.mean()),
        "lstm_std_auprc": float(lstm_auprcs.std()),
        "lstm_ensemble_auprc": float(ensemble_auprc),
        "lr_auprc": float(lr_auprc),
        "lr_best_params": lr_best_params,
        "observed_diff_lstm_minus_lr": float(observed_diff),
        "bootstrap": {
            "n_boot": int(len(boot_diffs)),
            "mean_diff": float(boot_diffs.mean()),
            "ci95_lo": float(ci_lo),
            "ci95_hi": float(ci_hi),
            "p_diff_le_0": boot_p_le0,
        },
        "permutation": {
            "n_perm": N_PERMUTATION,
            "null_mean": float(null_diffs.mean()),
            "null_sd": float(null_diffs.std()),
            "two_sided_p": perm_p_two_sided,
        },
    }

    out_path = os.path.join(RESULTS_DIR, "exp7_bootstrap_lr_vs_lstm.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved results: {out_path}")

    # --- Figure: bootstrap distribution ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(boot_diffs, bins=60, color="#4C72B0", alpha=0.85, edgecolor="white", linewidth=0.3)
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.axvline(boot_diffs.mean(), color="#C44E52", linestyle="-", linewidth=1.5,
               label=f"Mean diff = {boot_diffs.mean():+.4f}")
    ax.axvspan(ci_lo, ci_hi, color="#C44E52", alpha=0.12, label=f"95% CI [{ci_lo:+.3f}, {ci_hi:+.3f}]")
    ax.set_xlabel("Bootstrap AUPRC difference (LSTM ensemble $-$ tuned LR)")
    ax.set_ylabel("Count")
    ax.set_title(f"Paired bootstrap, B={len(boot_diffs)}")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, "exp7_bootstrap_diff.png")
    fig.savefig(fig_path, dpi=200)
    print(f"  Saved figure: {fig_path}")

    print("\n" + "=" * 70)
    print("  DONE -- Experiment 7 complete")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
