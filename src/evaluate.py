"""
Evaluation utilities for disappearance prediction models.

Provides metrics, plotting, and analysis functions for model evaluation.
"""

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, Optional


def evaluate_model(
    model: torch.nn.Module,
    X_test: np.ndarray,
    y_test: np.ndarray,
    threshold: float = 0.5,
    verbose: bool = True
) -> Dict[str, float]:
    """
    Comprehensive evaluation of a trained model.

    Args:
        model: Trained PyTorch model
        X_test: Test sequences [num_samples, seq_len, features]
        y_test: Test labels [num_samples]
        threshold: Classification threshold for binary predictions
        verbose: Print detailed results

    Returns:
        Dictionary of metrics
    """
    model.eval()

    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_test)
        y_pred_proba = model(X_tensor).numpy().flatten()

    y_pred_binary = (y_pred_proba >= threshold).astype(int)

    # Compute metrics
    metrics = {
        'auprc': average_precision_score(y_test, y_pred_proba),
        'auroc': roc_auc_score(y_test, y_pred_proba),
        'accuracy': (y_pred_binary == y_test).mean(),
        'baseline_auprc': y_test.mean(),  # Random baseline
        'baseline_auroc': 0.5,  # Random baseline for ROC
    }

    # Confusion matrix metrics
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_binary).ravel()

    metrics['true_positives'] = int(tp)
    metrics['false_positives'] = int(fp)
    metrics['true_negatives'] = int(tn)
    metrics['false_negatives'] = int(fn)

    # Precision, Recall, F1
    if tp + fp > 0:
        metrics['precision'] = tp / (tp + fp)
    else:
        metrics['precision'] = 0.0

    if tp + fn > 0:
        metrics['recall'] = tp / (tp + fn)
    else:
        metrics['recall'] = 0.0

    if metrics['precision'] + metrics['recall'] > 0:
        metrics['f1'] = 2 * (metrics['precision'] * metrics['recall']) / (metrics['precision'] + metrics['recall'])
    else:
        metrics['f1'] = 0.0

    # Improvement over baseline
    metrics['auprc_improvement'] = metrics['auprc'] / metrics['baseline_auprc']

    if verbose:
        print("\n" + "="*60)
        print("MODEL EVALUATION RESULTS")
        print("="*60)
        print(f"\nPrimary Metrics:")
        print(f"  AUPRC:     {metrics['auprc']:.3f} (baseline: {metrics['baseline_auprc']:.3f})")
        print(f"  AUROC:     {metrics['auroc']:.3f} (baseline: {metrics['baseline_auroc']:.3f})")
        print(f"  Improvement: {metrics['auprc_improvement']:.2f}× over random baseline")

        print(f"\nClassification Metrics (threshold={threshold}):")
        print(f"  Accuracy:  {metrics['accuracy']:.3f}")
        print(f"  Precision: {metrics['precision']:.3f}")
        print(f"  Recall:    {metrics['recall']:.3f}")
        print(f"  F1-Score:  {metrics['f1']:.3f}")

        print(f"\nConfusion Matrix:")
        print(f"  True Positives:  {metrics['true_positives']:4d}")
        print(f"  False Positives: {metrics['false_positives']:4d}")
        print(f"  True Negatives:  {metrics['true_negatives']:4d}")
        print(f"  False Negatives: {metrics['false_negatives']:4d}")

        print(f"\nTest Set Statistics:")
        print(f"  Total samples: {len(y_test)}")
        print(f"  Positive rate: {y_test.mean()*100:.1f}%")
        print("="*60 + "\n")

    return metrics


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    title: str = "Precision-Recall Curve",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot precision-recall curve.

    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        title: Plot title
        save_path: Path to save figure (optional)

    Returns:
        Matplotlib figure
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    auprc = average_precision_score(y_true, y_pred_proba)
    baseline = y_true.mean()

    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot PR curve
    ax.plot(recall, precision, linewidth=2, label=f'Model (AUPRC={auprc:.3f})')

    # Plot baseline
    ax.axhline(y=baseline, color='gray', linestyle='--', linewidth=1.5,
               label=f'Random Baseline ({baseline:.3f})')

    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved PR curve to {save_path}")

    return fig


def plot_roc_curve(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    title: str = "ROC Curve",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot ROC curve.

    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        title: Plot title
        save_path: Path to save figure (optional)

    Returns:
        Matplotlib figure
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    auroc = roc_auc_score(y_true, y_pred_proba)

    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot ROC curve
    ax.plot(fpr, tpr, linewidth=2, label=f'Model (AUROC={auroc:.3f})')

    # Plot diagonal (random classifier)
    ax.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=1.5,
            label='Random Baseline (0.500)')

    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved ROC curve to {save_path}")

    return fig


def plot_calibration_curve(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    n_bins: int = 10,
    title: str = "Calibration Curve",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot calibration curve to assess probability calibration.

    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        n_bins: Number of bins for calibration
        title: Plot title
        save_path: Path to save figure (optional)

    Returns:
        Matplotlib figure
    """
    # Bin predictions
    bins = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # Calculate empirical probabilities
    empirical_probs = []
    counts = []

    for i in range(n_bins):
        mask = (y_pred_proba >= bins[i]) & (y_pred_proba < bins[i+1])
        if mask.sum() > 0:
            empirical_probs.append(y_true[mask].mean())
            counts.append(mask.sum())
        else:
            empirical_probs.append(np.nan)
            counts.append(0)

    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot calibration curve
    ax.plot(bin_centers, empirical_probs, 'o-', linewidth=2, markersize=8,
            label='Model Calibration')

    # Plot perfect calibration
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Perfect Calibration')

    ax.set_xlabel('Predicted Probability', fontsize=12)
    ax.set_ylabel('Empirical Probability', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved calibration curve to {save_path}")

    return fig


def compare_models(
    results: Dict[str, Dict[str, float]],
    metric: str = 'auprc',
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Compare multiple models on a single metric.

    Args:
        results: Dictionary mapping model names to metric dictionaries
        metric: Metric to compare (e.g., 'auprc', 'auroc')
        save_path: Path to save figure (optional)

    Returns:
        Matplotlib figure
    """
    model_names = list(results.keys())
    scores = [results[name][metric] for name in model_names]

    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.barh(model_names, scores, color='steelblue', alpha=0.8)

    # Add baseline line if applicable
    if metric == 'auprc' and 'baseline_auprc' in results[model_names[0]]:
        baseline = results[model_names[0]]['baseline_auprc']
        ax.axvline(x=baseline, color='red', linestyle='--', linewidth=2,
                  label=f'Random Baseline ({baseline:.3f})')
        ax.legend()

    ax.set_xlabel(metric.upper(), fontsize=12)
    ax.set_title(f'Model Comparison: {metric.upper()}', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')

    # Add value labels on bars
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2,
                f'{scores[i]:.3f}',
                ha='left', va='center', fontsize=10, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved comparison plot to {save_path}")

    return fig


if __name__ == "__main__":
    # Example usage
    print("Evaluation utilities for disappearance prediction.")
    print("Import this module to use evaluation functions.")
    print("\nExample:")
    print("  from evaluate import evaluate_model, plot_precision_recall_curve")
    print("  metrics = evaluate_model(model, X_test, y_test)")
    print("  plot_precision_recall_curve(y_test, predictions)")
