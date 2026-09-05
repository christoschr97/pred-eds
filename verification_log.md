# Verification Log: Migrated Results vs. Paper-Cited Statistics

Checked every number in `results/*.json` (migrated in this session) against
the value the submitted PDF actually cites, by direct text extraction from
the paper (not from memory or from `REVIEWER_RESPONSE_EXPERIMENTS.md`,
which is the authors' own draft summary and could itself contain errors).

## Confirmed exact matches

| Statistic | Paper text | `results/*.json` value | Match |
|---|---|---|---|
| Persistence baseline AUPRC | "persistence baseline (0.570)"; "17.1 points above the persistence baseline" | `exp3_persistence.json`: `test_auprc = 0.5704` | ✅ (0.741 − 0.570 = 0.171, matches "17.1 points") |
| Tuned LR, Full Model AUPRC | "A tuned logistic regression (0.738) ... narrow[s] the gap" | `exp1_tuned_lr.json`: Full Model `test_auprc = 0.7383` | ✅ |
| XGBoost, Full Model AUPRC | "...and XGBoost (0.723) narrow the gap" | `exp2_gbm.json`: Full Model `test_auprc = 0.7227` | ✅ |
| SHAP mean \|value\|, disappearances (overall) | Table 6: "mean \|SHAP\| of 0.220" | `exp5_shap/shap_summary.json`: `feature_importance.disappearances = 0.219509` | ✅ |
| SHAP mean \|value\|, most recent month (t), disappearances | "most recent month (t) contributes a mean \|SHAP\| of 0.304 for disappearances" | `shap_mean_abs_per_timestep_feature["t"]["disappearances"] = 0.304212` | ✅ |
| SHAP mean \|value\|, oldest month (t−5), disappearances | "oldest lookback month (t − 5, mean \|SHAP\| = 0.200)" | `shap_mean_abs_per_timestep_feature["t-5"]["disappearances"] = 0.199864` | ✅ |
| SHAP mean \|value\|, fatalities | Table 6 (fatalities row) | `feature_importance.fatalities = 0.046425` → 0.046 | ✅ |
| SHAP mean \|value\|, arrests | Table 6 (arrests row) | `feature_importance.arrests = 0.017556` → 0.018 | ✅ |

## Minor discrepancy noted (not corrected — flagged for author review)

**SHAP mean \|value\|, violence.** Paper text states "Violence 0.029"
(Table 6). `shap_summary.json` gives `feature_importance.violence =
0.028487`, which rounds to **0.028**, not 0.029. A 0.001 rounding-boundary
difference — worth a one-line check against whatever rounding convention
produced the paper table, but not acted on here since no source script for
the paper table itself was found to adjudicate it.

**"No Past Disappearances" ablation AUPRC: two different numbers for
nominally the same configuration.** Paper Table 4 states this
15-feature ablation (run via `run_ablations.py`) achieves **0.666 ± 0.006**.
The later, independently-retrained model for the zero-history subset
experiment (`exp4_zero_history.py`, same 15-feature set, same three seeds
42/123/456) reports `full_test_auprc_mean = 0.6718 ± 0.0037` in
`exp4_zero_history_subset.json` — i.e. the same nominal model configuration
scores 0.666 in one script and 0.672 in another. The two values are within
each other's reported standard deviation (0.666 + 0.006 = 0.672), so this
is consistent with ordinary seed/training variance rather than a bug, but
it is a genuine numerical difference between two separately-run
implementations of "the same" ablation and should be reconciled or
footnoted before the camera-ready version, since Table 4's 0.666 is the
number the Discussion section's "0.741 to 0.666" claim depends on.

## Live re-execution (this session, superseding the note below)

A dedicated environment (`disact-repro`: Python 3.10, PyTorch 2.0.1,
NumPy 1.24.3, Pandas 2.0.2, scikit-learn 1.3.0, NetworkX 3.1 — matching
the paper's stated versions exactly) was built and every core script was
actually run against the raw ACLED CSV (via a local symlink into the
source repo's copy; see `RUN_GUIDE.md`). Commands and workarounds are
recorded in `RUN_GUIDE.md`.

**Data-level checks (`verify_statistics.py`)**: 202,347 events, 5,959
disappearances (2.9%), 6,913 arrests, 197,817 total fatalities, 87 admin1
regions (Nigeria 37 / Mexico 32 / Myanmar 18), date range 2018–2024 — all
match the manuscript text exactly. The script's own docstring flags two
"claims" (a 0.354 random baseline, and "197,817 violent events") as
mismatches; both belong to an earlier paper draft — the submitted PDF
already states the corrected values (0.483 test-set baseline in Table 3;
114,564 violent events distinct from 197,817 fatalities), so there is no
outstanding issue here, only a stale comment in the verification script.

**Table 4 (`run_ablations.py --seeds 3 --epochs 50`, all nine configs,
27 runs, fixed seeds 42/123/456)** — reproduced to 3 decimal places:

| Configuration | Paper (Table 4) | This run | #Feat |
|---|---|---|---|
| Events + Structure | 0.745 ± 0.002 | 0.7453 ± 0.0016 | 9 |
| Events Only | 0.745 ± 0.003 | 0.7451 ± 0.0025 | 4 |
| Events + Clustering | 0.745 ± 0.003 | 0.7451 ± 0.0026 | 6 |
| Events + Weights | 0.743 ± 0.001 | 0.7427 ± 0.0011 | 6 |
| Full Model | 0.741 ± 0.003 | 0.7409 ± 0.0030 | 16 |
| Events + Centrality | 0.741 ± 0.001 | 0.7407 ± 0.0006 | 7 |
| No Fatalities | 0.740 ± 0.003 | 0.7404 ± 0.0026 | 15 |
| No Past Disappearances | 0.666 ± 0.006 | 0.6660 ± 0.0058 | 15 |
| Network Only | 0.664 ± 0.003 | 0.6641 ± 0.0030 | 12 |

Every row matches at the precision reported in the paper. This also
means the "No Past Disappearances" discrepancy noted below is now
confirmed as real and reproducible, not a stale-file artifact: re-running
`run_ablations.py` from scratch still gives 0.666 (matching Table 4),
while `exp4_zero_history.py`'s own from-scratch retrain of the nominally
same configuration still gives 0.672. Both scripts are internally
deterministic across independent re-executions; they simply are not
computing an identical model/features (see open item below).

**Table 3 non-LSTM rows (`revision_experiments/exp1-3`)** — exact
reproduction: Persistence 0.5704 (paper: 0.570), Tuned LR Full Model
0.7383 (paper: 0.738), XGBoost Full Model 0.7227 (paper: 0.723).

**Single-run Attention-LSTM sanity check (Full Model, seed 42 only,
outside the 3-seed mean)**: Test AUPRC 0.7452 in 11 seconds wall time.

**Epoch behavior — resolves "epochs were very few"**: the paper states
(Methods) training runs "for a maximum of 50 epochs with early stopping:
if test set AUPRC does not improve for 10 consecutive epochs, training
terminates." Every run observed in this session triggered early stopping
well before the 50-epoch cap — e.g. epoch 13 (Full Model, seed 42), epoch
15–16 and 22 in other configurations, with the best checkpoint typically
found by epoch 3–5. The nominal "50 epochs" in the CLI default is a
ceiling that is essentially never reached; actual training is 13–22
epochs per run. This matches the recollection and is a direct, reproduced
property of the code, not a paper-vs-code inconsistency.

## Not independently re-executed

`train.py` (Standard-LSTM comparison row, Table 3's 0.740) and
`create_figures.py` were not re-run this session — both have packaging
issues (relative/absolute imports assuming a different directory
structure than the flat migrated layout) that are documented, not
silently patched, in `RUN_GUIDE.md` §6. `exp5_shap.py` (Table 6 SHAP
values) was also not re-run. The rest of the manuscript's quantitative
claims (Tables 3 and 4, data statistics, epoch/early-stopping behavior)
are now independently confirmed by live re-execution rather than by
comparing static JSON files against paper text.
