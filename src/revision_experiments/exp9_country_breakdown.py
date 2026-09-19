"""
Experiment 9 — Country-Level Performance Breakdown (regenerates Table 8)
========================================================================
The submitted paper reports per-country regions, test sequences, positive
rate and AUPRC for the recommended Events Only model. Nothing in the
codebase computes that breakdown: run_ablations.py reports pooled test
AUPRC only, and exp1-exp8 do not disaggregate. With the sample expanded
from three countries to five, the table cannot be carried over.

This script trains the recommended model once per seed on the pooled
training data — the same model the paper reports — and then scores the
test set separately within each country. It does NOT train per-country
models; the table describes where one pooled model performs well.

Setup:
- Events Only feature set: arrests, violence, disappearances, fatalities
- 6-month lookback window, strict temporal split (0.7 of the month range)
- Attention-LSTM, 3 seeds, 50 epochs, hidden 64, 2 layers, dropout 0.2
- pos_weight computed from the pooled training split, w+ = (1 - p) / p
- Per-country AUPRC is the mean +/- std across seeds

The per-country random baseline is that country's own test positive rate,
which is the number the AUPRC must be read against — pooled prevalence is
not the right comparison for a country whose prevalence differs.

Output: results/exp9_country_breakdown.json
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
    temporal_split,
    normalize_sequences,
)
from models import AttentionLSTM, train_lstm, evaluate_lstm
from run_ablations import create_sequences_with_feature_subset

from sklearn.metrics import average_precision_score, roc_auc_score
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
    "exp9_country_breakdown.json",
)

SEQUENCE_LENGTH = 6
SPLIT_RATIO     = 0.7
SEEDS           = [42, 123, 456]

EVENT_FEATURES = ["arrests", "violence", "disappearances", "fatalities"]


def get_predictions(model: AttentionLSTM, X: np.ndarray, device: str = "cpu") -> np.ndarray:
    """Get sigmoid probabilities from a trained model."""
    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X).to(device)
        logits   = model(X_tensor)
        probs    = torch.sigmoid(logits).cpu().numpy().flatten()
    return probs


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "="*70)
    print("  EXPERIMENT 9 — Country-Level Performance (Table 8)")
    print("="*70)

    print(f"\nData path: {DATA_PATH}")
    df         = load_acled_data(DATA_PATH)
    monthly_df = create_monthly_aggregation_with_networks(df)

    X, y, region_ids, regions, month_targets = create_sequences_with_feature_subset(
        monthly_df,
        sequence_length=SEQUENCE_LENGTH,
        feature_subset=EVENT_FEATURES,
    )

    # ── cutoff month (mirrors temporal_split logic, as exp4 does) ─────────────
    unique_months_sorted = np.sort(np.unique(month_targets))
    cutoff_idx           = int(len(unique_months_sorted) * SPLIT_RATIO)
    cutoff_month         = unique_months_sorted[cutoff_idx]
    print(f"\n  Cutoff month : {cutoff_month}")

    # ── temporal split — preserve test region_ids ─────────────────────────────
    test_mask       = month_targets >= cutoff_month
    region_ids_test = region_ids[test_mask]

    X_train_raw, y_train, X_test_raw, y_test = temporal_split(
        X, y, region_ids, month_targets, split_ratio=SPLIT_RATIO
    )

    X_train_norm, X_test_norm, _ = normalize_sequences(X_train_raw, X_test_raw)

    # ── map test sequences to countries via region_ids ────────────────────────
    country_of_region = np.array([r[0] for r in regions])
    country_test      = country_of_region[region_ids_test]
    countries         = sorted(set(country_of_region.tolist()))

    print("\n  Test sequences by country:")
    for c in countries:
        m = country_test == c
        print(f"    {c:<12} regions {int((country_of_region == c).sum()):>4} | "
              f"test seq {int(m.sum()):>5} | pos rate {y_test[m].mean()*100:>5.1f}%")

    pos_weight = float((1 - y_train.mean()) / y_train.mean())
    print(f"\n  pos_weight w+ : {pos_weight:.3f}")

    # ── train & score per seed ────────────────────────────────────────────────
    per_seed = []

    for seed in SEEDS:
        print(f"\n{'='*70}")
        print(f"  SEED {seed}")
        print(f"{'='*70}")

        np.random.seed(seed)
        torch.manual_seed(seed)

        model = AttentionLSTM(
            input_dim=X_train_norm.shape[2], hidden_dim=64, num_layers=2, dropout=0.2
        )

        train_result = train_lstm(
            model=model,
            X_train=X_train_norm,
            y_train=y_train,
            X_test=X_test_norm,
            y_test=y_test,
            num_epochs=50,
            batch_size=64,
            pos_weight=pos_weight,
            device="cpu",
            verbose=False,
        )

        pooled = evaluate_lstm(
            model=train_result["model"],
            X_train=X_train_norm,
            y_train=y_train,
            X_test=X_test_norm,
            y_test=y_test,
            device="cpu",
        )

        probs = get_predictions(train_result["model"], X_test_norm)

        row = {"seed": seed, "pooled_test_auprc": round(float(pooled["test_auprc"]), 4)}
        for c in countries:
            m = country_test == c
            if m.sum() == 0 or y_test[m].sum() == 0:
                row[c] = None
                continue
            row[c] = round(float(average_precision_score(y_test[m], probs[m])), 4)

        print(f"  pooled {row['pooled_test_auprc']:.4f} | " +
              " | ".join(f"{c[:3]} {row[c]:.4f}" for c in countries if row[c] is not None))
        per_seed.append(row)

    # ── aggregate ─────────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("  TABLE 8 — country-level performance, Events Only model")
    print("="*70)
    print(f"  {'Country':<12} {'Regions':>8} {'Test seq':>9} {'Pos rate':>9} {'AUPRC':>17}")

    table = []
    for c in countries:
        m  = country_test == c
        vs = np.array([r[c] for r in per_seed if r[c] is not None])
        table.append({
            "country":         c,
            "n_regions":       int((country_of_region == c).sum()),
            "n_test_seq":      int(m.sum()),
            "test_pos_rate":   round(float(y_test[m].mean()), 4),
            "auprc_mean":      round(float(vs.mean()), 4) if len(vs) else None,
            "auprc_std":       round(float(vs.std()),  4) if len(vs) else None,
        })
        print(f"  {c:<12} {table[-1]['n_regions']:>8} {table[-1]['n_test_seq']:>9} "
              f"{table[-1]['test_pos_rate']:>8.3f} "
              f"{table[-1]['auprc_mean']:>10.4f} ± {table[-1]['auprc_std']:.4f}")

    pooled_vs = np.array([r["pooled_test_auprc"] for r in per_seed])
    overall = {
        "country":       "Overall",
        "n_regions":     int(len(regions)),
        "n_test_seq":    int(len(y_test)),
        "test_pos_rate": round(float(y_test.mean()), 4),
        "auprc_mean":    round(float(pooled_vs.mean()), 4),
        "auprc_std":     round(float(pooled_vs.std()),  4),
    }
    print(f"  {'Overall':<12} {overall['n_regions']:>8} {overall['n_test_seq']:>9} "
          f"{overall['test_pos_rate']:>8.3f} "
          f"{overall['auprc_mean']:>10.4f} ± {overall['auprc_std']:.4f}")

    spread = max(t["auprc_mean"] for t in table) - min(t["auprc_mean"] for t in table)
    print(f"\n  Across-country AUPRC spread : {spread:.4f}")
    print(f"  Max seed std within country  : {max(t['auprc_std'] for t in table):.4f}")

    payload = {
        "experiment": "exp9_country_breakdown",
        "note": (
            "One pooled Events Only Attention-LSTM per seed, scored separately "
            "within each country's test sequences. Not per-country models. "
            "Read each country's AUPRC against its own test positive rate, not "
            "the pooled prevalence. Spread across countries should be compared "
            "with the seed-to-seed std before being described as a ranking."
        ),
        "config": {
            "feature_set": EVENT_FEATURES,
            "sequence_length": SEQUENCE_LENGTH,
            "split_ratio": SPLIT_RATIO,
            "seeds": SEEDS,
            "pos_weight": round(pos_weight, 4),
            "cutoff_month": str(cutoff_month),
        },
        "per_seed": per_seed,
        "table": table,
        "overall": overall,
        "across_country_spread": round(float(spread), 4),
        "max_within_country_seed_std": round(float(max(t["auprc_std"] for t in table)), 4),
    }

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n  Saved: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
