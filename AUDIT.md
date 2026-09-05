# Code Audit: Migration to `disact-pred-eds`

Audit date: 2026-09-05. Source repo: `projects/DISACT-GNN/` (git branch
`revise_and_resubmit_experiments`, HEAD `b144129` "Run Tuned Logistic
Regression", 2026-05-21). Target paper: "Predicting Enforced Disappearances:
A Machine Learning Analysis of Temporal and Network Features"
(Christodoulou, Djouvas, Ioannidis), submission
`pred_eds_ml_forecasting_REVISIONS_SUB_READY_2_June_V1 (6).pdf`.

Method: every file below was matched to the paper by (a) reading the paper's
Methods/Results/Reproducibility sections in full, (b) reading the source
repo's git log/status/diff (not just file mtimes, which reflect a bulk
checkout and are not reliable on their own), and (c) tracing which script
produces each numbered table/figure. No code was altered during audit; the
one intentional content correction (requirements.txt pinning) is called out
separately below and applied in a later migration step, not silently.

## 1. Core pipeline — matches paper Tables 2 and 4 exactly

All of the following live in `lstm_new_features/publication_code/` and were
committed in `5bdab1b` ("finalizing everything", 2026-02-15) unless noted.
I verified the ablation configuration dict in `run_ablations.py` line-by-line
against paper Table 2: all nine configurations match on feature count and
composition (Full=16, Events Only=4, Network Only=12, No Disappearances=15,
No Fatalities=15, Events+Structure=9, Events+Centrality=7,
Events+Clustering=6, Events+Weights=6).

| File | Git blob hash | Status |
|---|---|---|
| `data_prep.py` | `30764e4c...` | committed, matches paper §3.1-3.4 |
| `train.py` | `2762cea2...` | committed |
| `run_ablations.py` | `b146d374...` | committed, configs verified against Table 2 |
| `evaluate.py` | `5ec68a76...` | committed |
| `verify_statistics.py` | `34561656...` | committed (identical to `lstm_new_features/verify_paper_statistics.py`, a duplicate — only one copy migrated) |
| `create_figures.py` | `73c2340b...` | committed (identical to `lstm_new_features/create_paper_figures.py`, a duplicate — only one copy migrated) |
| `LICENSE` | `e5e609ee...` | committed |
| `models.py` | committed `ed615c9d...` → working tree `4790ff41...` | **uncommitted 1-line change**: drops deprecated `verbose=False` arg from `torch.optim.lr_scheduler.ReduceLROnPlateau` (PyTorch 2.x compatibility). Migrating the working-tree version, since it is what the paper's stated PyTorch 2.0.1 dependency requires. |

`data_prep.py`, `train.py`, `models.py` etc. are also duplicated verbatim at
`lstm_new_features/*.py` (package root, used by `cli.py`/`runner.py` for
interactive runs). Confirmed byte-identical via `diff`; only the
`publication_code/` copies were migrated, as that folder is the
already-designated redistribution snapshot.

## 2. Reviewer-response revision experiments — uncommitted in source repo

These five scripts, all in `lstm_new_features/publication_code/`, are the
material behind `REVIEWER_RESPONSE_EXPERIMENTS.md` (root of source repo,
also untracked) and directly produce paper Table 3 (architecture
comparison), the persistence-baseline discussion, and Figs. 4-5 (SHAP).
`exp1`/`exp2` are committed in `b144129`; `exp3`-`exp5` and the modified
`models.py` above are untracked working-tree files — none of this had been
pushed to any branch prior to this migration.

| File | Git status | Paper element addressed |
|---|---|---|
| `exp1_tuned_lr.py` | committed in `b144129` | Table 3 "Tuned LR" rows (0.738 full, 0.743 events-only); reviewer comment on untuned LR baseline |
| `exp2_gbm_baseline.py` | committed in `b144129` | Table 3 XGBoost rows (0.723 full, 0.733 events-only); reviewer comment on gradient-boosted baseline |
| `exp3_persistence.py` | **untracked** (`1a9a54ad...`) | Persistence baseline (AUPRC 0.570) cited throughout Results/Discussion; reviewer comment on ViEWS-standard sanity check |
| `exp4_zero_history.py` | **untracked** (`388ceca1...`) | Zero-history subset analysis (2/87 regions), Discussion §"Predictive Patterns" |
| `exp5_shap.py` | **untracked** (`fb4a8e4b...`) | Figs. 4-5 (SHAP recency gradient, directional effects), Table on SHAP feature attribution |

Corresponding results migrated: `results/exp1_tuned_lr.json`,
`exp2_gbm.json`, `exp3_persistence.json` (untracked),
`exp4_zero_history_subset.json` (untracked), `exp5_shap/` directory
(untracked: `shap_values.npy`, `shap_summary.json`, three figures in
PNG+PDF).

`REVIEWER_RESPONSE_EXPERIMENTS.md` migrated to `docs/` unchanged — it maps
every reviewer comment (both referees) to the experiment or writing fix that
addresses it and is the working reference for the response letter.

## 3. Figure provenance — every numbered figure in the paper traced to source

| Paper figure | Caption (short) | Producing script | Migrated? |
|---|---|---|---|
| Fig. 1 | End-to-end pipeline schematic | not code-generated (manual diagram) | n/a |
| Fig. 2 | Monthly disappearances by country, 2018-2024 | `create_paper_figures.py` → `figure1_time_series.{png,pdf}` | yes |
| Fig. 3 | Feature ablation results, sorted by AUPRC | `create_paper_figures.py` → `figure3_ablation_results.{png,pdf}` | yes |
| Fig. 4 | SHAP temporal heatmap (recency gradient) | `exp5_shap.py` → `results/exp5_shap/fig2_shap_heatmap.{png,pdf}` | yes |
| Fig. 5 | Signed SHAP direction per feature | `exp5_shap.py` → `results/exp5_shap/fig3_shap_direction.{png,pdf}` | yes |

**Not migrated, flagged for your decision:**
- `figure2_architecture`, `figure4_precision_recall`,
  `figure5_feature_importance`, `figure6_temporal_stability`,
  `figure7_attention_weights` (all from `create_paper_figures.py`'s 7-figure
  output) — none are cited by figure number in the submitted PDF (`grep` for
  "Figure N" found only Figs. 1-5).
- `lstm_new_features/generate_predictions_for_figures.py` — its own
  docstring says it targets "Figures 4, 6, 7", an older 7-figure numbering
  scheme superseded by the SHAP-based Figs. 4-5 now in the paper.
- Root-level `figures/` directory (`calibration_curves`,
  `distribution_shift`, `error_analysis`, `geographic_performance`,
  `lstm_temporal_importance`, `performance_comparison`) — dated 2026-01-17,
  predates the "adding also the paper" commit (`b8ea051`, 2026-02-10) and
  predates the revision entirely. None of these titles match a captioned
  figure in the current PDF text.

## 4. Requirements.txt — reproducibility gap, corrected

The paper's §3.10 Reproducibility statement specifies exact package
versions: PyTorch 2.0.1, NumPy 1.24.3, Pandas 2.0.2, NetworkX 3.1,
scikit-learn 1.3.0. The source `publication_code/requirements.txt` instead
has loose lower bounds (`torch>=1.13.0`, `numpy>=1.23.0`, `pandas>=1.5.0`,
`scikit-learn>=1.2.0`, `networkx>=2.8.0`). This is the one deliberate content
change made during migration (not a refactor of any analysis code) — see
step "Correct requirements.txt pinning".

## 5. Confirmed out of scope (deprecated / unrelated), with evidence

| Item | Evidence |
|---|---|
| `analyze_sparsity.py`, `visualize_sequences.py` (repo root) | Import `baseline_experiments.data_prep`, a different pipeline using a `log_pop` feature; current `data_prep.py` docstring explicitly excludes population ("NO population feature (noise reduction)") |
| `parse_data.py`, `parse_data_v2.py`, `parse_data_monthly.py` (repo root) | Output-saving lines commented out (`# df_grid.to_csv(...) # Save if needed`); not wired to any current script |
| `models/ST_HGAT.py`, `ST_HGNN.py`, `ST_HGTNN.py`, `ST_HTANN.py`, `ST_Baseline_GRU.py` | Graph-neural-network / temporal-graph architectures, superseded line, out of scope per project constraints |
| `disact_gnn/`, `gnn_experiments/`, `baseline_experiments/`, `simple_models/` | Deprecated experiment folders (per project context — anything outside `lstm_new_features` other than what's listed above is superseded work) |
| `00_EDA_ACLED.ipynb`, `01_EDA_ACLED.ipynb` | Exploratory notebooks, not cited in Methods |
| `sn-article-template/` | Springer Nature LaTeX manuscript template, not analysis code |
| `data/*.csv`, `data/*.xlsx` | Raw ACLED exports; not redistributable (ACLED terms of use / prior README already noted this restriction). Migration documents the exact source filename (`ACLED Data_2025-12-31_Nigeria_Mexico_Myanmar.csv`) instead of copying it. |

## 6. Stale documentation — flagged, not silently rewritten

`publication_code/README.md`, `STRUCTURE.md`, and `PUBLICATION_CHECKLIST.md`
reference a different paper title ("Enforced Disappearances are
Institutionally-Driven: Evidence from Machine Learning as Theory Testing")
than the submitted PDF, and describe ablation feature counts that don't
match the actual `run_ablations.py` (which does match the paper). These
were **not migrated as-is** and are **not yet rewritten** — that decision
is left open for you and co-authors; see the open-questions note at the end
of this migration.
