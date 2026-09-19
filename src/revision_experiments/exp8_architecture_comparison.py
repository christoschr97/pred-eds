"""
Experiment 8 — Architecture Comparison (regenerates Table 4)
=============================================================
Table 4 of the submitted paper reports Standard LSTM, Attention-LSTM and an
untuned Logistic Regression on the full 16-feature set. None of exp1-exp7
regenerate those rows: exp1 tunes the LR, exp2 is XGBoost, exp3 is
persistence. train.py holds the original comparison but imports its
dependencies package-relative (`from .models import ...`) while src/ has no
__init__.py, so it cannot run in this tree.

This script reproduces the same comparison by calling models.py directly,
exactly as run_ablations.py and exp4 do.

Setup:
- Full 16-feature set, 6-month lookback window
- Strict temporal split (0.7 of the month range for training)
- Attention-LSTM and Standard LSTM: 3 seeds each, 50 epochs, hidden 64,
  2 layers, dropout 0.2 — the paper's configuration
- Untuned Logistic Regression: sklearn defaults on the flattened window
  (this is the "LR (untuned)" row; exp1 supplies the tuned row)
- pos_weight / class_weight computed from the training split, w+ = (1 - p) / p

Gap convention follows models.py and the paper's Table 4:
    gap = train AUPRC - test AUPRC
(exp1 and exp2 print the opposite sign; do not mix them.)

Output: results/exp8_architecture_comparison.json
"""

import sys
import os
import json
import numpy as np

# ── path plumbing ─────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_prep import (
    DATA_FILE,
    load_acled_data,
    create_monthly_aggregation_with_networks,
    create_temporal_sequences,
    temporal_split,
    normalize_sequences,
)
from models import StandardLSTM, AttentionLSTM, train_lstm, evaluate_lstm

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss
import torch

# ── config ────────────────────────────────────────────────────────────────────
DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data",
    DATA_FILE,
)
RESULTS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "results",
    "exp8_architecture_comparison.json",
)

SEQUENCE_LENGTH = 6
SPLIT_RATIO     = 0.7
SEEDS           = [42, 123, 456]
HIDDEN_DIM      = 64
NUM_LAYERS      = 2
DROPOUT         = 0.2
NUM_EPOCHS      = 50
BATCH_SIZE      = 64


def flatten(X: np.ndarray) -> np.ndarray:
    """Flatten [n, seq_len, features] to [n, seq_len * features]."""
    return X.reshape(X.shape[0], -1)


def run_untuned_lr(X_train, y_train, X_test, y_test, class_weight):
    """Default-hyperparameter sklearn LogisticRegression.

    'Untuned' means the sklearn defaults: C=1.0, penalty='l2', solver='lbfgs'.
    Only max_iter is raised, because the default 100 does not converge on a
    96-dimensional flattened window; convergence is not hyperparameter tuning.
    """
    clf = LogisticRegression(class_weight=class_weight, max_iter=5000, random_state=42)
    clf.fit(X_train, y_train)

    p_train = clf.predict_proba(X_train)[:, 1]
    p_test  = clf.predict_proba(X_test)[:, 1]

    train_auprc = average_precision_score(y_train, p_train)
    test_auprc  = average_precision_score(y_test,  p_test)

    return {
        "train_auprc": round(float(train_auprc), 4),
        "test_auprc":  round(float(test_auprc),  4),
        "train_brier": round(float(brier_score_loss(y_train, p_train)), 4),
        "test_brier":  round(float(brier_score_loss(y_test,  p_test)),  4),
        "test_auroc":  round(float(roc_auc_score(y_test, p_test)), 4),
        "gap":         round(float(train_auprc - test_auprc), 4),
    }


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "="*70)
    print("  EXPERIMENT 8 — Architecture Comparison (Table 4)")
    print("="*70)

    print(f"\nData path: {DATA_PATH}")
    df         = load_acled_data(DATA_PATH)
    monthly_df = create_monthly_aggregation_with_networks(df)

    X, y, region_ids, regions, month_targets = create_temporal_sequences(
        monthly_df, sequence_length=SEQUENCE_LENGTH
    )

    X_train_raw, y_train, X_test_raw, y_test = temporal_split(
        X, y, region_ids, month_targets, split_ratio=SPLIT_RATIO
    )

    X_train_norm, X_test_norm, _ = normalize_sequences(X_train_raw, X_test_raw)

    pos_weight = float((1 - y_train.mean()) / y_train.mean())
    print(f"\n  Regions            : {len(regions)}")
    print(f"  Train sequences    : {len(y_train)}  (positive rate {y_train.mean()*100:.1f}%)")
    print(f"  Test  sequences    : {len(y_test)}   (positive rate {y_test.mean()*100:.1f}%)")
    print(f"  Input dim          : {X_train_norm.shape[2]} features x {SEQUENCE_LENGTH} months")
    print(f"  pos_weight w+      : {pos_weight:.3f}")

    results = {"attention_lstm": [], "standard_lstm": []}

    for arch_name, arch_cls in (("attention_lstm", AttentionLSTM),
                                ("standard_lstm",  StandardLSTM)):
        for seed in SEEDS:
            print(f"\n{'='*70}")
            print(f"  {arch_name.upper()} — SEED {seed}")
            print(f"{'='*70}")

            np.random.seed(seed)
            torch.manual_seed(seed)

            model = arch_cls(
                input_dim=X_train_norm.shape[2],
                hidden_dim=HIDDEN_DIM,
                num_layers=NUM_LAYERS,
                dropout=DROPOUT,
            )

            train_result = train_lstm(
                model=model,
                X_train=X_train_norm,
                y_train=y_train,
                X_test=X_test_norm,
                y_test=y_test,
                num_epochs=NUM_EPOCHS,
                batch_size=BATCH_SIZE,
                pos_weight=pos_weight,
                device="cpu",
                verbose=False,
            )

            metrics = evaluate_lstm(
                model=train_result["model"],
                X_train=X_train_norm,
                y_train=y_train,
                X_test=X_test_norm,
                y_test=y_test,
                device="cpu",
            )

            print(f"  Train AUPRC {metrics['train_auprc']:.4f} | "
                  f"Test AUPRC {metrics['test_auprc']:.4f} | "
                  f"gap {metrics['gap']:+.4f} | "
                  f"Brier {metrics['test_brier']:.4f}")

            results[arch_name].append({
                "seed":        seed,
                "train_auprc": round(float(metrics["train_auprc"]), 4),
                "test_auprc":  round(float(metrics["test_auprc"]),  4),
                "test_auroc":  round(float(metrics["test_auroc"]),  4),
                "test_brier":  round(float(metrics["test_brier"]),  4),
                "gap":         round(float(metrics["gap"]),         4),
            })

    # ── untuned Logistic Regression ───────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  LOGISTIC REGRESSION (untuned, sklearn defaults)")
    print(f"{'='*70}")

    lr_metrics = run_untuned_lr(
        flatten(X_train_norm), y_train,
        flatten(X_test_norm),  y_test,
        class_weight={0: 1.0, 1: pos_weight},
    )
    print(f"  Train AUPRC {lr_metrics['train_auprc']:.4f} | "
          f"Test AUPRC {lr_metrics['test_auprc']:.4f} | "
          f"gap {lr_metrics['gap']:+.4f} | "
          f"Brier {lr_metrics['test_brier']:.4f}")

    # ── summary table ─────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("  TABLE 4 — full 16-feature set")
    print("="*70)
    print(f"  {'Model':<26} {'Test AUPRC':>16} {'Train':>8} {'Gap':>8} {'Brier':>8}")

    summary = {}
    for arch_name in ("attention_lstm", "standard_lstm"):
        rs = results[arch_name]
        te = np.array([r["test_auprc"] for r in rs])
        tr = np.array([r["train_auprc"] for r in rs])
        gp = np.array([r["gap"] for r in rs])
        br = np.array([r["test_brier"] for r in rs])
        summary[arch_name] = {
            "test_auprc_mean": round(float(te.mean()), 4),
            "test_auprc_std":  round(float(te.std()),  4),
            "train_auprc_mean": round(float(tr.mean()), 4),
            "gap_mean":         round(float(gp.mean()), 4),
            "test_brier_mean":  round(float(br.mean()), 4),
            "n_seeds": len(rs),
        }
        print(f"  {arch_name:<26} {te.mean():.4f} ± {te.std():.4f}   "
              f"{tr.mean():.4f}  {gp.mean():+.4f}  {br.mean():.4f}")

    summary["logistic_regression_untuned"] = lr_metrics
    print(f"  {'logistic_regression_untuned':<26} {lr_metrics['test_auprc']:.4f}           "
          f"{lr_metrics['train_auprc']:.4f}  {lr_metrics['gap']:+.4f}  {lr_metrics['test_brier']:.4f}")

    payload = {
        "experiment": "exp8_architecture_comparison",
        "note": (
            "Regenerates the Standard LSTM, Attention-LSTM and untuned LR rows of "
            "Table 4 on the full 16-feature set. gap = train AUPRC - test AUPRC "
            "(models.py / paper convention; exp1 and exp2 report the opposite sign). "
            "LSTM rows are mean +/- std over 3 seeds."
        ),
        "config": {
            "sequence_length": SEQUENCE_LENGTH,
            "split_ratio": SPLIT_RATIO,
            "seeds": SEEDS,
            "hidden_dim": HIDDEN_DIM,
            "num_layers": NUM_LAYERS,
            "dropout": DROPOUT,
            "num_epochs": NUM_EPOCHS,
            "batch_size": BATCH_SIZE,
            "pos_weight": round(pos_weight, 4),
        },
        "sample": {
            "n_regions": int(len(regions)),
            "n_train_sequences": int(len(y_train)),
            "n_test_sequences": int(len(y_test)),
            "train_positive_rate": round(float(y_train.mean()), 4),
            "test_positive_rate": round(float(y_test.mean()), 4),
        },
        "per_seed": results,
        "summary": summary,
    }

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n  Saved: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
