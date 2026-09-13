# Validation-split early-stopping diagnostic

**Date:** 2026-09-13
**Environment:** conda env `disact-repro` (torch 2.0.1, scikit-learn 1.3.0, networkx 3.1, pandas 2.0.2)
**Script:** `val_split_diagnostic.py` (this directory)
**Raw output:** `val_split_diagnostic.json`
**Figure:** `val_split_diagnostic_comparison.png`

## What was tested

`src/models.py::train_lstm` monitors test-set loss (not test-set AUPRC, despite
what §3.6/§3.8 said before this round's correction) for early stopping,
checkpoint selection, and the LR scheduler. There is no independent
validation split anywhere in the pipeline — `train.py` and `run_ablations.py`
both pass the test set directly into the monitoring argument.

This diagnostic reruns the Full Model (16 features, Attention-LSTM,
hidden=64/layers=2/dropout=0.2, 50 epochs/batch=64/lr=0.001/patience=10,
seeds 0/1/2 — identical to `run_ablations.py`) under two conditions:

1. **`test_loss (as published)`** — replicates the existing code exactly,
   monitoring test loss. Sanity-checks against the published number.
2. **`val_loss (corrected)`** — carves the last 15% of the training months
   (5,490 → 1,170 sequences) into a validation split; the scheduler, patience
   counter, and checkpoint selection key off validation loss instead. The
   test set (3,036 sequences, 2023-2024, identical in both conditions) is
   touched exactly once, after training, for the reported metric.

## Result

| Setting | Test AUPRC (mean ± std, 3 seeds) | Test AUROC (mean ± std) |
|---|---|---|
| test-loss-monitored (as published) | 0.8102 ± 0.0007 | 0.8217 ± 0.0012 |
| validation-loss-monitored (corrected) | 0.8053 ± 0.0015 | 0.8169 ± 0.0026 |
| Tuned LR baseline (`exp1_tuned_lr.json`, CV on train only) | 0.8109 | 0.8189 |

Published Full Model reference (`results/ablations_5c.json`): 0.8088 ± 0.0006
test AUPRC. The test-loss-monitored replicate here (0.8102) is close enough
to serve as a sanity check that the diagnostic code is faithful to the
published pipeline; the residual gap is consistent with environment/package
version noise.

## Status

Sensitivity check only. **Not propagated into any results table, ablation
result, country-level result, SHAP result, or reported AUPRC/AUROC figure.**
The published numbers stand as originally reported. This diagnostic exists
to confirm that the qualitative conclusion (no LSTM advantage over the tuned
LR baseline) is not an artifact of the early-stopping criterion — it is not
a replacement experiment and the full ablation suite was not rerun under the
corrected protocol.

## Internal note (for the record, not the manuscript or reviewer response)

> Validation-split diagnostic: Replacing test-loss-driven early stopping with
> a temporally held-out validation split reduced Full Model Attention-LSTM
> test AUPRC from 0.8102 ± 0.0007 to 0.8053 ± 0.0015. The qualitative
> architecture comparison was unchanged; tuned LR remained comparable at
> 0.8109.

Neither reviewer raised this point in round 2. It is not volunteered in
`RESPONSE_TO_REVIEWERS_R3.md` this cycle. Manuscript text was corrected in
§3.6 and §3.8 to describe the actual stopping signal (evaluation/test loss,
not AUPRC) without introducing a claim about a validation split that the
reported numbers don't use.

---

# Class-weight discrepancy diagnostic

**Date:** 2026-09-13
**Environment:** conda env `disact-repro`
**Scripts:** `classweight_lr_rerun.py`, `classweight_xgb_rerun.py` (copies of
`src/revision_experiments/exp1_tuned_lr.py` / `exp2_gbm_baseline.py`, output
path redirected only — no other code change)
**Raw output:** `classweight_lr_rerun.json`, `classweight_xgb_rerun.json`
**Figure:** `classweight_diagnostic_comparison.png`

## What was tested

§3.6, §3.8, and §3.9 stated $w_+ = (1-p)/p = 0.627/0.373 = 1.683$. That is
the three-country training prevalence ($p=0.373$) from the original
submission. On the five-country sample the actual training prevalence is
31.1%, giving $w_+ = 2.213$ — the value the LSTM code computes fresh every
run (`run_ablations.log`: `Positive class weight: 2.213 (balances 31.1%
positive rate)`) and the value Figure 1 (`exp8_architecture_comparison.json`,
`"pos_weight": 2.2127`) already reports.

`exp1_tuned_lr.py` and `exp2_gbm_baseline.py` both recompute the weight
dynamically from `y_train.mean()` at runtime — the module-level constants
(`CLASS_WEIGHT = {0: 1.0, 1: 1.683}`, `SCALE_POS_WEIGHT = 1.683`) are
overwritten before fitting. The published `results/exp1_tuned_lr.json` and
`results/exp2_gbm.json` (dated 8 September, description field literally
reads `1.683`) predate that fix. This diagnostic reran both scripts unmodified
except for the output path, to see whether re-fitting under the corrected
2.213 weight changes the reported baseline numbers.

## Result

| Model | Feature set | test AUPRC (published, w+=1.683) | test AUPRC (rerun, w+=2.213) |
|---|---|---|---|
| Tuned LR | Full Model | 0.8109 | 0.8109 |
| Tuned LR | Events Only | 0.8115 | 0.8115 |
| XGBoost | Full Model | 0.7920 | 0.7920 |
| XGBoost | Events Only | 0.8054 | 0.8054 |

test AUROC, test Brier score, and the GridSearchCV-selected hyperparameters
(`best_params`) are identical between the two weight settings for all four
model/feature-set combinations, to the four decimal places both files are
saved at.

## Status

Sensitivity check only. **Not propagated into `results/exp1_tuned_lr.json`,
`results/exp2_gbm.json`, or any reported baseline number.** The published
baseline numbers stand as originally reported — this diagnostic confirms the
stale 1.683 weight did not affect them, so no rerun of the baselines was
necessary. Only the manuscript prose (§3.6, §3.8, §3.9) needed correcting,
from 1.683 to 2.213, to describe the weight the LSTM runs and Figure 1
actually use.

Neither reviewer raised this point. Not volunteered in
`RESPONSE_TO_REVIEWERS_R3.md`.
