"""
Experiment 4 — Prior-History-Zero Subset Evaluation
=====================================================
Reviewer response: the current AUPRC=0.666 (No Past Disappearances model)
mixes regions with and without disappearance history. The early warning
claim requires isolating truly zero-history regions.

Method:
- Zero-history regions: admin1 units with ZERO disappearance events in the
  entire training period (Jul 2018 – Dec 2022)
- Retrain Attention-LSTM with No Past Disappearances features (15 feat)
  using seeds [42, 123, 456]
- Filter test set to sequences from zero-history regions only
- Evaluate the trained models on this subset — no retraining on subset
- Report: n_sequences, positive_rate, AUPRC (mean ± std across seeds)

Save to: results/exp4_zero_history_subset.json
"""

import sys
import os
import json
import numpy as np

# ── path plumbing ─────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))          # publication_code/
sys.path.insert(1, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # lstm_new_features/

from data_prep import (
    DATA_FILE,
    load_acled_data,
    create_monthly_aggregation_with_networks,
    temporal_split,
    normalize_sequences,
)
from models import AttentionLSTM, train_lstm, evaluate_lstm
from run_ablations import create_sequences_with_feature_subset

from sklearn.metrics import average_precision_score
import torch

# ── config ────────────────────────────────────────────────────────────────────
DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data",
    DATA_FILE,
)

SEQUENCE_LENGTH = 6
SPLIT_RATIO     = 0.7
SEEDS           = [42, 123, 456]

NO_DISAPPEARANCES_FEATURES = [
    "arrests", "violence", "fatalities",
    "num_actors", "num_edges", "density", "avg_degree", "max_degree",
    "num_components", "largest_component_size", "avg_clustering",
    "transitivity", "avg_edge_weight", "max_edge_weight", "degree_centralization",
]  # 15 features — all except 'disappearances'

RESULTS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "results",
    "exp4_zero_history_subset.json",
)


# ── helpers ───────────────────────────────────────────────────────────────────

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
    print("  EXPERIMENT 4 — Prior-History-Zero Subset Evaluation")
    print("  Early warning: can the model predict NEW outbreak regions?")
    print("="*70)

    # ── load data ─────────────────────────────────────────────────────────────
    print(f"\nData path: {DATA_PATH}")
    df         = load_acled_data(DATA_PATH)
    monthly_df = create_monthly_aggregation_with_networks(df)

    # ── build sequences with 15-feature subset (no disappearances) ────────────
    # create_sequences_with_feature_subset also returns region_ids and regions
    X, y, region_ids, regions, month_targets = create_sequences_with_feature_subset(
        monthly_df,
        sequence_length=SEQUENCE_LENGTH,
        feature_subset=NO_DISAPPEARANCES_FEATURES,
    )

    # ── identify cutoff month (mirrors temporal_split logic) ──────────────────
    unique_months        = np.unique(month_targets)
    unique_months_sorted = np.sort(unique_months)
    cutoff_idx           = int(len(unique_months_sorted) * SPLIT_RATIO)
    cutoff_month         = unique_months_sorted[cutoff_idx]

    print(f"\n  Cutoff month : {cutoff_month}")

    # ── find zero-history regions ─────────────────────────────────────────────
    # Regions where total disappearances in training period == 0
    train_monthly = monthly_df[monthly_df["month"] < cutoff_month]
    region_disapp = (
        train_monthly.groupby(["country", "admin1"])["disappearances"]
        .sum()
        .reset_index()
    )
    zero_history_pairs = set(
        tuple(row) for row in
        region_disapp[region_disapp["disappearances"] == 0][["country", "admin1"]].values
    )

    # Map (country, admin1) pairs → region_ids
    zero_history_ids = np.array([
        i for i, r in enumerate(regions) if r in zero_history_pairs
    ])

    print(f"\n  Zero-history regions (0 disappearances in training period):")
    print(f"    Total regions      : {len(regions)}")
    print(f"    Zero-history count : {len(zero_history_ids)}")
    for rid in sorted(zero_history_ids):
        print(f"      [{rid:3d}] {regions[rid][0]} — {regions[rid][1]}")

    # ── temporal split — preserve test region_ids ─────────────────────────────
    test_mask          = month_targets >= cutoff_month
    region_ids_test    = region_ids[test_mask]
    month_targets_test = month_targets[test_mask]

    X_train_raw, y_train, X_test_raw, y_test = temporal_split(
        X, y, region_ids, month_targets, split_ratio=SPLIT_RATIO
    )

    # ── zero-history test subset ───────────────────────────────────────────────
    zero_mask  = np.isin(region_ids_test, zero_history_ids)
    X_test_zh  = X_test_raw[zero_mask]
    y_test_zh  = y_test[zero_mask]

    print(f"\n  Zero-history TEST subset:")
    print(f"    Sequences  : {len(y_test_zh)} / {len(y_test)} total test sequences")
    print(f"    Pos. rate  : {y_test_zh.mean()*100:.1f}%")
    print(f"    Month range: {month_targets_test[zero_mask].min()} → {month_targets_test[zero_mask].max()}")

    if len(y_test_zh) == 0:
        print("\n  ⚠️  No zero-history test sequences found — all regions had at least one disappearance in training.")
        return

    if y_test_zh.sum() == 0:
        print("\n  ⚠️  Zero-history test subset has no positive labels — AUPRC undefined.")
        return

    # ── normalise ─────────────────────────────────────────────────────────────
    X_train_norm, X_test_norm, _ = normalize_sequences(X_train_raw, X_test_raw)
    X_test_zh_norm = X_test_norm[zero_mask]

    # ── train & evaluate across seeds ─────────────────────────────────────────
    seed_results = []

    for seed in SEEDS:
        print(f"\n{'='*70}")
        print(f"  SEED {seed}")
        print(f"{'='*70}")

        np.random.seed(seed)
        torch.manual_seed(seed)

        input_dim  = X_train_norm.shape[2]
        model      = AttentionLSTM(input_dim=input_dim, hidden_dim=64, num_layers=2, dropout=0.2)
        pos_weight = (1 - y_train.mean()) / y_train.mean()

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
            verbose=True,
        )

        # Full test set metrics (for reference)
        full_metrics = evaluate_lstm(
            model=train_result["model"],
            X_train=X_train_norm,
            y_train=y_train,
            X_test=X_test_norm,
            y_test=y_test,
            device="cpu",
        )

        # Zero-history subset evaluation
        y_pred_zh  = get_predictions(train_result["model"], X_test_zh_norm)
        auprc_zh   = average_precision_score(y_test_zh, y_pred_zh)

        print(f"\n  Full test AUPRC      : {full_metrics['test_auprc']:.4f}  (reference — matches paper ~0.666)")
        print(f"  Zero-history AUPRC   : {auprc_zh:.4f}  ← key number")
        print(f"  Random baseline      : {y_test_zh.mean():.4f}  (prevalence in zero-history subset)")

        seed_results.append({
            "seed":               seed,
            "full_test_auprc":    round(full_metrics["test_auprc"], 4),
            "zero_history_auprc": round(float(auprc_zh), 4),
            "zero_history_n":     int(len(y_test_zh)),
            "zero_history_pos_rate": round(float(y_test_zh.mean()), 4),
        })

    # ── aggregate results ─────────────────────────────────────────────────────
    full_auprcs = [r["full_test_auprc"]    for r in seed_results]
    zh_auprcs   = [r["zero_history_auprc"] for r in seed_results]

    print("\n" + "="*70)
    print("  EXPERIMENT 4 SUMMARY")
    print("="*70)
    print(f"\n  No Past Disappearances model — full test set:")
    print(f"    AUPRC : {np.mean(full_auprcs):.4f} ± {np.std(full_auprcs):.4f}  (paper reports 0.666 ± 0.006)")
    print(f"\n  No Past Disappearances model — ZERO-HISTORY regions only:")
    print(f"    AUPRC : {np.mean(zh_auprcs):.4f} ± {np.std(zh_auprcs):.4f}")
    print(f"    n     : {seed_results[0]['zero_history_n']} sequences")
    print(f"    Pos.  : {seed_results[0]['zero_history_pos_rate']*100:.1f}%")
    print(f"    Random baseline for this subset: {seed_results[0]['zero_history_pos_rate']:.4f}")

    if np.mean(zh_auprcs) > seed_results[0]["zero_history_pos_rate"]:
        print(f"\n  ✅ Model beats random on zero-history regions — supports early warning claim")
    else:
        print(f"\n  ⚠️  Model does NOT beat random on zero-history regions")

    # ── save JSON ─────────────────────────────────────────────────────────────
    output = {
        "experiment": "Exp4 — Prior-History-Zero Subset Evaluation",
        "description": (
            "Attention-LSTM with No Past Disappearances features (15 feat). "
            "Zero-history regions = admin1 units with 0 disappearances in training period. "
            "Evaluated on zero-history test sequences only. Seeds [42, 123, 456]."
        ),
        "zero_history_regions": [
            {"country": regions[rid][0], "admin1": regions[rid][1]}
            for rid in sorted(zero_history_ids)
        ],
        "n_zero_history_regions": int(len(zero_history_ids)),
        "n_total_regions":        int(len(regions)),
        "summary": {
            "full_test_auprc_mean": round(float(np.mean(full_auprcs)), 4),
            "full_test_auprc_std":  round(float(np.std(full_auprcs)),  4),
            "zh_auprc_mean":        round(float(np.mean(zh_auprcs)),   4),
            "zh_auprc_std":         round(float(np.std(zh_auprcs)),    4),
            "zh_n_sequences":       seed_results[0]["zero_history_n"],
            "zh_positive_rate":     seed_results[0]["zero_history_pos_rate"],
            "zh_random_baseline":   seed_results[0]["zero_history_pos_rate"],
        },
        "per_seed": seed_results,
    }

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n✅  Results saved to: {RESULTS_PATH}")
    print("\n" + "="*70)
    print("  DONE — Experiment 4 complete")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
