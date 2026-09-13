"""
Experiment 5 — SHAP Feature Attribution
=========================================
Purpose: Understand what the model is actually using for prediction.
Reframed from theory-testing to predictive power analysis:
  - Which features drive predictions?
  - Which months in the 6-month lookback window matter most?
  - Are the learned patterns temporally sensible?

Method:
- Retrain Events Only Attention-LSTM (seeds 42, 123, 456)
- shap.GradientExplainer on full test set
- Average SHAP values across 3 seeds

Figures produced:
  1. Bar chart  — mean |SHAP| per feature (collapsed over timesteps)
  2. Beeswarm  — direction + magnitude per feature×timestep
  3. Heatmap   — mean |SHAP| per [timestep × feature] (6 × 4)

Save to: results/exp5_shap/
"""

import sys
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

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
from models import AttentionLSTM, train_lstm

import torch
import shap

# ── config ────────────────────────────────────────────────────────────────────
DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data",
    DATA_FILE,
)

SEQUENCE_LENGTH  = 6
SPLIT_RATIO      = 0.7
SEEDS            = [42, 123, 456]
EVENT_FEATURES   = ["arrests", "violence", "disappearances", "fatalities"]
ALL_FEATURES     = [
    "arrests", "violence", "disappearances", "fatalities",
    "num_actors", "num_edges", "density", "avg_degree",
    "max_degree", "num_components", "largest_component_size",
    "avg_clustering", "transitivity",
    "avg_edge_weight", "max_edge_weight", "degree_centralization",
]
BACKGROUND_N     = 100   # background samples for GradientExplainer

OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "results", "exp5_shap",
)

# ── plot style (matches paper figures) ────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi":    300,
    "savefig.dpi":   300,
    "font.size":     10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
})
try:
    plt.style.use("seaborn-v0_8-paper")
except:
    plt.style.use("seaborn-paper")

COLORS = {
    "arrests":        "#d62728",
    "violence":       "#ff7f0e",
    "disappearances": "#1f77b4",
    "fatalities":     "#9467bd",
}

# ── helpers ───────────────────────────────────────────────────────────────────

def select_event_features(X: np.ndarray) -> np.ndarray:
    """Extract the 4 event feature columns from 16-feature sequences."""
    idx = [ALL_FEATURES.index(f) for f in EVENT_FEATURES]
    return X[:, :, idx]   # [N, 6, 4]


def train_seed(X_train, y_train, X_test, y_test, seed):
    """Train one Attention-LSTM seed, return trained model."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    model      = AttentionLSTM(input_dim=4, hidden_dim=64, num_layers=2, dropout=0.2)
    pos_weight = (1 - y_train.mean()) / y_train.mean()

    result = train_lstm(
        model=model,
        X_train=X_train, y_train=y_train,
        X_test=X_test,   y_test=y_test,
        num_epochs=50, batch_size=64,
        pos_weight=pos_weight,
        device="cpu", verbose=True,
    )
    return result["model"]


def compute_shap_values(model, X_train, X_test):
    """
    Compute SHAP values using GradientExplainer.

    Returns shap_values: np.ndarray [N_test, seq_len, n_features]
    """
    model.eval()

    # Background: random subsample of training set
    np.random.seed(0)
    bg_idx  = np.random.choice(len(X_train), size=BACKGROUND_N, replace=False)
    bg_data = torch.FloatTensor(X_train[bg_idx])

    X_test_t  = torch.FloatTensor(X_test)
    explainer = shap.GradientExplainer(model, bg_data)

    print(f"    Computing SHAP on {len(X_test)} test sequences...")
    shap_vals = explainer.shap_values(X_test_t)   # list or array

    # GradientExplainer can return a list (one per output) — take first element
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[0]

    sv = np.array(shap_vals)
    if sv.ndim == 4 and sv.shape[-1] == 1:
        sv = sv.squeeze(-1)       # [N, 6, 4, 1] → [N, 6, 4]
    return sv


# ── plotting ──────────────────────────────────────────────────────────────────

def plot_bar(shap_mean_abs, output_dir):
    """
    Figure 1: Mean |SHAP| per feature (collapsed over all timesteps).
    """
    # Mean over timesteps → [4]
    feat_importance = shap_mean_abs.mean(axis=0)   # [4]
    order           = np.argsort(feat_importance)[::-1]
    features_sorted = [EVENT_FEATURES[i] for i in order]
    values_sorted   = feat_importance[order]
    colors_sorted   = [COLORS[f] for f in features_sorted]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.barh(features_sorted[::-1], values_sorted[::-1],
                   color=colors_sorted[::-1], alpha=0.85, edgecolor="white")

    ax.set_xlabel("Mean |SHAP value|", fontsize=11)
    ax.set_title("Feature Importance (Events Only Attention-LSTM)", fontsize=12, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    ax.set_xlim(left=0)

    # Value labels
    for bar, val in zip(bars, values_sorted[::-1]):
        ax.text(val + 0.0005, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=9)

    plt.tight_layout()
    path = os.path.join(output_dir, "fig1_shap_bar.png")
    plt.savefig(path, bbox_inches="tight")
    plt.savefig(path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_heatmap(shap_mean_abs, output_dir):
    """
    Figure 2: Mean |SHAP| heatmap — 6 timesteps × 4 features.
    Rows = timestep (t-5 = oldest … t = most recent)
    """
    # shap_mean_abs: [6, 4]  (timesteps × features)
    labels_t = [f"t−{5-i}" if i < 5 else "t" for i in range(6)]
    labels_f = EVENT_FEATURES

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(
        shap_mean_abs,
        xticklabels=labels_f,
        yticklabels=labels_t,
        annot=True, fmt=".4f",
        cmap="YlOrRd",
        linewidths=0.5,
        ax=ax,
        cbar_kws={"label": "Mean |SHAP|"},
    )
    ax.set_title("Temporal Feature Importance (6-month lookback)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Feature", fontsize=11)
    ax.set_ylabel("Lookback month", fontsize=11)
    plt.tight_layout()
    path = os.path.join(output_dir, "fig2_shap_heatmap.png")
    plt.savefig(path, bbox_inches="tight")
    plt.savefig(path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_direction(shap_vals_raw, X_test, output_dir):
    """
    Figure 3: Direction plot — mean signed SHAP per feature per timestep.
    Shows whether high feature values push prediction UP or DOWN.
    """
    # shap_vals_raw: [N, 6, 4]
    # Mean signed SHAP: [6, 4]
    mean_signed = shap_vals_raw.mean(axis=0)    # [6, 4]

    labels_t = [f"t−{5-i}" if i < 5 else "t" for i in range(6)]

    fig, axes = plt.subplots(1, 4, figsize=(12, 4), sharey=True)
    fig.suptitle("Feature Direction Effect on Predictions\n(positive = increases disappearance risk)",
                 fontsize=11, fontweight="bold")

    for j, (feat, ax) in enumerate(zip(EVENT_FEATURES, axes)):
        vals   = mean_signed[:, j]
        cols   = [COLORS[feat] if v >= 0 else "#aaaaaa" for v in vals]
        ax.barh(labels_t, vals, color=cols, alpha=0.85, edgecolor="white")
        ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_title(feat, fontsize=10, fontweight="bold")
        ax.set_xlabel("Mean SHAP", fontsize=9)
        ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "fig3_shap_direction.png")
    plt.savefig(path, bbox_inches="tight")
    plt.savefig(path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "="*70)
    print("  EXPERIMENT 5 — SHAP Feature Attribution")
    print("  Focus: predictive power — what is the model actually using?")
    print("="*70)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── load data and build full 16-feature sequences ─────────────────────────
    print(f"\nData path: {DATA_PATH}")
    df         = load_acled_data(DATA_PATH)
    monthly_df = create_monthly_aggregation_with_networks(df)

    X_full, y, region_ids, regions, month_targets = create_temporal_sequences(
        monthly_df, sequence_length=SEQUENCE_LENGTH
    )

    X_train_raw, y_train, X_test_raw, y_test = temporal_split(
        X_full, y, region_ids, month_targets, split_ratio=SPLIT_RATIO
    )

    # Normalise using full 16-feat scaler, then slice event features
    X_train_norm, X_test_norm, _ = normalize_sequences(X_train_raw, X_test_raw)

    X_train_ev = select_event_features(X_train_norm)   # [N_train, 6, 4]
    X_test_ev  = select_event_features(X_test_norm)    # [N_test,  6, 4]

    print(f"\n  Train shape: {X_train_ev.shape}")
    print(f"  Test  shape: {X_test_ev.shape}")
    print(f"  Train pos rate: {y_train.mean()*100:.1f}%")
    print(f"  Test  pos rate: {y_test.mean()*100:.1f}%")

    # ── train 3 models and accumulate SHAP values ─────────────────────────────
    all_shap_vals = []

    for seed in SEEDS:
        print(f"\n{'='*70}")
        print(f"  SEED {seed} — train + SHAP")
        print(f"{'='*70}")

        model = train_seed(X_train_ev, y_train, X_test_ev, y_test, seed)

        print(f"  Computing SHAP values...")
        sv = compute_shap_values(model, X_train_ev, X_test_ev)   # [N, 6, 4]
        print(f"  SHAP shape: {sv.shape}")
        all_shap_vals.append(sv)

    # ── average across seeds ──────────────────────────────────────────────────
    shap_vals_avg  = np.mean(all_shap_vals, axis=0)   # [N, 6, 4]
    shap_mean_abs  = np.abs(shap_vals_avg).mean(axis=0)  # [6, 4] — timestep × feature

    print(f"\n{'='*70}")
    print(f"  SHAP SUMMARY (mean |SHAP| per feature, averaged over timesteps)")
    print(f"{'='*70}")
    feat_importance = shap_mean_abs.mean(axis=0)   # [4] — collapsed over timesteps
    for feat, imp in sorted(zip(EVENT_FEATURES, feat_importance.tolist()), key=lambda x: -x[1]):
        bar = "█" * int(imp / float(feat_importance.max()) * 30)
        print(f"  {feat:<20} {imp:.5f}  {bar}")

    print(f"\n  Temporal pattern (which month matters most, averaged over features):")
    timestep_importance = shap_mean_abs.mean(axis=1)   # [6]
    labels_t = [f"t-{5-i}" if i < 5 else "t (most recent)" for i in range(6)]
    for label, imp in zip(labels_t, timestep_importance.tolist()):
        bar = "█" * int(imp / float(timestep_importance.max()) * 30)
        print(f"  {label:<20} {imp:.5f}  {bar}")

    # ── produce figures ───────────────────────────────────────────────────────
    print(f"\n  Generating figures → {OUTPUT_DIR}")
    plot_bar(shap_mean_abs, OUTPUT_DIR)
    plot_heatmap(shap_mean_abs, OUTPUT_DIR)
    plot_direction(shap_vals_avg, X_test_ev, OUTPUT_DIR)

    # ── save numpy array ──────────────────────────────────────────────────────
    npy_path = os.path.join(OUTPUT_DIR, "shap_values.npy")
    np.save(npy_path, shap_vals_avg)
    print(f"  Saved: {npy_path}")

    # ── save JSON summary ─────────────────────────────────────────────────────
    json_path = os.path.join(OUTPUT_DIR, "shap_summary.json")
    output = {
        "experiment": "Exp5 — SHAP Feature Attribution",
        "model": "Events Only Attention-LSTM",
        "seeds": SEEDS,
        "n_test_sequences": int(len(y_test)),
        "background_samples": BACKGROUND_N,
        "feature_importance": {
            feat: round(float(imp), 6)
            for feat, imp in zip(EVENT_FEATURES, feat_importance)
        },
        "timestep_importance": {
            (f"t-{5-i}" if i < 5 else "t"): round(float(v), 6)
            for i, v in enumerate(timestep_importance)
        },
        "shap_mean_abs_per_timestep_feature": {
            (f"t-{5-i}" if i < 5 else "t"): {
                feat: round(float(shap_mean_abs[i, j]), 6)
                for j, feat in enumerate(EVENT_FEATURES)
            }
            for i in range(SEQUENCE_LENGTH)
        },
    }
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Saved: {json_path}")

    print(f"\n✅  All outputs in: {OUTPUT_DIR}")
    print("\n" + "="*70)
    print("  DONE — Experiment 5 complete")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
