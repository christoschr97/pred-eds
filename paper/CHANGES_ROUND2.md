# CHANGES_ROUND2.md — Round 2 Revision Diff Summary

Diff base: `git show HEAD:paper/sn-article.tex` (the migrated, unmodified state of the
June submission-ready manuscript). Diff target: current working copy of
`paper/sn-article.tex` / `paper/sn-bibliography.bib`, compiled to `sn-article.pdf`
(35 pages, up from the June PDF; recompiled clean with `pdflatex` → `bibtex` →
`pdflatex` → `pdflatex`, zero undefined references, zero bibtex warnings).

Full machine diff: `git diff HEAD -- paper/sn-article.tex paper/sn-bibliography.bib`
(83 insertions, 33 deletions across sn-article.tex; 23 insertions across
sn-bibliography.bib). What follows groups those line-level changes by the
reviewer point or scope decision that motivated them, in manuscript order.

---

## 1. Reviewer 2 — outcome terminology ("enforced disappearance" → "abduction/forced-disappearance")

**Comment addressed:** the paper's coded outcome (ACLED's `Abduction/forced disappearance`
sub-event type) does not require state-actor perpetration, so calling it "enforced
disappearance" throughout overstates what the coding actually captures — that legal
term is reserved (Section~\ref{sec:related}) for acts by state agents or with state
acquiescence.

**Changes**, all in `sn-article.tex`:
- Abstract: "predicting binary reported disappearance onset" → "predicting binary
  reported abduction/forced-disappearance incidence".
- Introduction: "Predicting enforced disappearances presents..." → "Predicting
  abduction/forced-disappearance events presents...".
- Research Question 1: "Can enforced disappearances be predicted... despite their
  rarity (2.9% of events)?" → "Can abduction/forced-disappearance events be
  predicted... despite their rarity **at the event level** (2.9% of events)?"
  (also flags the rarity figure as event-level, feeding into item 3 below).
- Contribution 1: "no additional predictive value" → "no incremental predictive
  value beyond the event-based features in this dataset and design" (also
  narrows the causal-sounding "no additional value" claim to what the ablation
  design actually supports).
- Contribution 3, Related Work ("previously unexamined outcome: enforced
  disappearances" → "...abduction/forced-disappearance events"), "Critical gap"
  paragraph, dataset description ("5,959 enforced disappearances" → "5,959
  abduction/forced-disappearance events"), outcome variable definition, Discussion
  opening sentence, and the Conclusions' opening sentence — same substitution
  applied consistently at each occurrence of the paper's own outcome (11 sites
  total). Citations to others' use of "civil war onset" (Ward et al.) and "onset"
  as a term of art in the cited literature are left untouched, since those refer
  to a different outcome in a different study, not to this paper's own coding.
- Methods §"Data Collection and Preparation": added two sentences after the
  `sub_event_type` coding description clarifying (a) what a corroborated ACLED
  report requires and that ACLED records actor type per event, and (b) that this
  coding is broader than the ICPPED legal definition and that no perpetrator-type
  filter is applied, so the modeled outcome is the operational ACLED category, not
  a verified legal determination.

## 2. VIEWS/admin1 citation contradiction

**Comment addressed:** the admin1-aggregation justification cited ViEWS
(`hegre2019views`), but ViEWS forecasts at PRIO-GRID and country-month level, not
admin1 — the citation did not support the claim it was attached to.

**Change:** "following the spatial granularity employed by leading conflict
forecasting systems \cite{hegre2019views}" → "following the spatial granularity
employed by subnational conflict forecasting work operating at the admin1/admin0
level \cite{oswald2026googletrends}". New bibliography entry added:
`oswald2026googletrends` (Oswald, *Journal of Conflict Resolution*, 2026,
DOI 10.1177/00220027251362832), verified against the published record before
citing.

## 3. Onset/incidence conflation and the rare-event framing

**Comment addressed:** the paper's Introduction motivated AUPRC and rare-event
methods using the 2.9% *event-level* disappearance rate, but the actual
classification task operates on a 37–48% positive *region-month* prevalence — a
factor-of-15 difference that made the rare-event framing internally
inconsistent with the paper's own random-baseline AUPRC of 0.483 (not the ≈0.03
that 2.9% prevalence would imply).

**Changes:**
- "Rare Events in Conflict: Methodological Foundations" — reworded to state the
  2.9% figure describes event-level rarity, and added a clause noting the
  region-month classification task itself has a substantially more moderate
  imbalance (37–48% positive), per Section~\ref{sec:methods}.
- AUPRC-justification paragraph — rewritten so AUPRC is motivated by the
  operational goal (identifying positive region-months) rather than by the
  2.9% event-level rarity, and states explicitly that the region-month random
  baseline is 0.483, not the 0.029 that the event-level figure would imply.
- Civil-war-onset comparison sentence — reworded from "resulting in severe class
  imbalance" framing to state plainly that King & Zeng's bias-correction methods
  are foundational for low-prevalence outcomes such as civil war onset (1–2%
  prevalence), without implying the present study's operationalized task shares
  that prevalence.

## 4. Motivation, LSTM justification, and related-work gap (co-author scope decision)

**Changes:**
- Introduction reordered to lead with the theoretical motivation paragraph
  (network-centric vs. state-capacity accounts) and the paper's framing sentence
  before the disappearance-specific background paragraphs, so the paper opens
  with why the question matters rather than with definitional background.
- Related Work §"Conflict Prediction and Early Warning Systems": added a
  sentence noting the field's move toward regression-based fatality/magnitude
  forecasting, and explaining the paper's binary framing as a deliberate choice
  matched to an early-warning trigger use case, cross-referencing the new
  Poisson regression supplement (item 5).
- Related Work §"Temporal Modeling for Conflict Forecasting": added a sentence
  justifying the choice of a standard (non-graph, non-transformer) LSTM given
  the small input space (six-month sequences, 4–16 features) and precedent in
  the closest prior work (Malone 2022, Radford 2022).

## 5. Supplementary Poisson regression baseline (Reviewer 3)

**Comment addressed:** does the binary classification framing conceal
count-level signal that a regression model would capture differently?

**Changes:**
- New Methods subsection "Supplementary Regression Analysis"
  (`\label{sec:poisson_methods}`): L2-penalized Poisson GLM (scikit-learn
  `PoissonRegressor`), same input construction and temporal split as the main
  classifier, `GridSearchCV`/`TimeSeriesSplit` tuning on negative mean Poisson
  deviance, evaluated via MAE, RMSE, mean Poisson deviance, McFadden pseudo-$R^2$,
  and Lin's CCC (citing Correndo et al. 2022, new bib entry `correndo2022metrica`).
- New Results subsection "Supplementary Poisson Regression on Monthly Counts"
  (`\label{sec:poisson_results}`, Table~\ref{tab:poisson}, new Table 9):
  Events Only outperforms Full Model on every metric (MAE 1.242 vs. 1.259, RMSE
  2.422 vs. 2.450, deviance 2.454 vs. 2.741, McFadden $R^2$ 0.186 vs. 0.123, CCC
  0.240 vs. 0.143) — same direction as the classification ablations.
- Script: `src/revision_experiments/exp6_poisson_regression.py`; results:
  `results/exp6_poisson_regression.json`.

## 6. Paired bootstrap/permutation test on the LSTM-vs-LR AUPRC gap (Reviewer 3)

**Comment addressed:** is the Attention-LSTM's 0.741 vs. tuned LR's 0.738 test
AUPRC (Table~\ref{tab:model_comparison}) a real generalization advantage, or is
it within noise — i.e., is the LSTM architecture actually needed?

**Changes:**
- New paragraph in Results §"Model Architecture Comparison" (right after the
  existing train/test-gap discussion): paired bootstrap (B=10,000 resamples of
  the shared n=2,035 test set) gives a 95% CI on the AUPRC difference of
  $[-0.008, +0.014]$ (crosses zero); a paired permutation test (P=10,000
  label-swaps) gives a two-sided $p=0.708$. States plainly that the raw AUPRC
  advantage is not statistically distinguishable from zero, and that this does
  not affect the (separately valid) train-test generalization-gap finding.
- One linking sentence added to Discussion §"Implications for Early Warning and
  Methodology", extending the "simple features suffice" argument to model
  architecture: the LSTM-vs-LR choice is not decided by aggregate AUPRC on this
  task.
- Script: `src/revision_experiments/exp7_bootstrap_lr_vs_lstm.py`; results:
  `results/exp7_bootstrap_lr_vs_lstm.json`; figure:
  `paper/figures/exp7_bootstrap_diff.png` (bootstrap-distribution histogram,
  supplementary — not embedded in the manuscript body).

## 7. Scope-adjacent wording tightened alongside the above (not separate reviewer points, but touched by the same edits)

Several sentences claiming network features have "no predictive value" or
"no additional predictive value" outright were narrowed, wherever the surrounding
sentence was already being edited for another reason, to "no incremental
predictive value beyond the event-based features in this dataset and design" —
matching the paper's own ablation design (which tests incremental contribution
conditional on the event features already being present, not an unconditional
absence of signal). Sites: pipeline figure caption, Contribution 1, ablation
results §"Feature Ablation Results" (two sentences), Discussion §"Why Network
Features Fail", §"Predictive Patterns and Their Interpretation", and the
Conclusions. This was not a separate reviewer point — it was tightened
incidentally at each location already touched for items 1–2 above, not applied
as a standalone global find-replace.

Two additional sentences were softened from causal/generalization overreach
independent of the above: the country-generalizability sentence in Methods
("we evaluate the robustness... to determine if the limited utility of network
features is a consistent property of disappearance dynamics" retained, but the
following sentence now distinguishes robustness *within* the three selected
contexts from generalizability *beyond* them), and the country-level-performance
sentence in Results (same distinction: consistent scores across countries do not
by themselves prove the model captures the same substantive precursors in each).
The Limitations paragraph on reporting bias was expanded by one clause
acknowledging that reporting gaps may themselves correlate with repression
intensity.

---

## Bibliography additions

| Key | Source | Used for |
|---|---|---|
| `oswald2026googletrends` | Oswald, *J. Conflict Resolution* 70(2-3):499-524, 2026 | admin1-granularity citation (item 2) |
| `correndo2022metrica` | Correndo et al., *JOSS* 7(79):4655, 2022 | Poisson regression metrics (item 5) |

## Files changed / added

- `paper/sn-article.tex` — 83 insertions, 33 deletions (git diff vs. HEAD)
- `paper/sn-bibliography.bib` — 23 insertions (2 new entries)
- `paper/sn-article.pdf` — recompiled, 35 pages
- `src/revision_experiments/exp6_poisson_regression.py` — new
- `src/revision_experiments/exp7_bootstrap_lr_vs_lstm.py` — new
- `results/exp6_poisson_regression.json` — new
- `results/exp7_bootstrap_lr_vs_lstm.json` — new
- `paper/figures/exp7_bootstrap_diff.png` — new (supplementary, not in manuscript body)

No changes were made to `src/data_prep.py`, `src/models.py`, `src/train.py`,
`src/run_ablations.py`, or any existing `results/exp[1-5]*` outputs — the
original pipeline code and its previously reported results are unchanged.

---

## 8. Post-submission audit findings (informational — no manuscript or codebase changes)

These items surfaced from follow-up questions after the round-2 text edits above were
already complete. None of them changed `sn-article.tex`, `sn-bibliography.bib`, or any
file under `src/` or `results/`. They are logged here because they bear on claims made
in `RESPONSE_TO_REVIEWERS.md` and `CO_AUTHOR_BRIEFING.md` and should be on the table for
the co-author meeting (see the pending list below).

### 8.1 Data pipeline: last-observed-month labeling defect

`data_prep.py`'s monthly aggregation only creates rows for (region, month) pairs with at
least one recorded event, so a region's timeline can have calendar gaps. The sequence
target is built with `shift(-1)`, which advances to the next *observed* row rather than
the next calendar month; a region's true final observed month therefore has no valid
target and should be dropped via a NaN check. The comparison used to build the binary
label evaluates `NaN > 0`, which is `False`, not `NaN` — so instead of being dropped, the
row is coded as a confirmed negative (target = 0).

Traced through `create_temporal_sequences`, not just the raw row table: this reaches 87
of 6,293 total sequences (1.4%), one per each of the 87 admin1 regions. All 87 land in
the test set (87 / 2,035, 4.3% of test rows; 0 / 4,258 in train), because the temporal
cutoff (2023-01) precedes the true final observed month for every region given the raw
data runs through December 2024. The mean predicted probability the tuned LR assigns to
these forced-zero rows is 0.592 — a region's last recorded month is typically still an
active conflict zone, not a quiet one, so the mislabeling tends to convert a correct
positive prediction into a scored false positive.

**Three ad hoc checks, run in `/tmp/adhoc_sensitivity/` only, not added to the repository
per explicit instruction:**

- **Reproduction + exclusion.** Full test set reproduces the published Table 3 numbers
  exactly (LR 0.7383, LSTM 0.7413 vs. paper's 0.738 / 0.741 ± 0.003). Excluding the 87
  phantom rows: LR rises to 0.7820 (+0.0437), LSTM to 0.7810 (+0.0396). The LSTM − LR
  gap flips sign, from +0.0030 (full test set, matching the paper) to −0.0011 (phantom
  rows excluded) — consistent with the existing bootstrap finding (Section 6) that this
  gap is not distinguishable from noise.
- **Full Model vs. Events Only ablation, with and without the 87 rows.** Full test set:
  Full Model 0.7409 ± 0.0030, Events Only 0.7451 ± 0.0025 (gap −0.0042). Excluding
  phantom rows: Full Model 0.7805 ± 0.0023, Events Only 0.7860 ± 0.0024 (gap −0.0055).
  Direction and approximate magnitude of the "network features add no incremental
  value" finding are unaffected either way.
- **GRU / Transformer comparison**, reusing the paper's own architecture-agnostic
  trainer, splits, and seeds. Full Model: LSTM 0.741, GRU (attention) 0.7420 ± 0.0022
  (44,930 params), tiny Transformer 0.7400 ± 0.0006 (18,369 params). Events Only: LSTM
  0.745, GRU 0.7479 ± 0.0003 (42,626 params), Transformer 0.7457 ± 0.0017 (17,985
  params). All four architectures land within a 0.740–0.748 band, and Events Only beats
  Full Model for every architecture tested, not just the paper's LSTM — the network-null
  finding is not an LSTM-specific artifact.

No reported number changes as a result of this — the defect exists in the current
pipeline and is unfixed. It is a candidate for a one-sentence Limitations footnote
(Table 3's absolute AUPRC is a conservative estimate given this labeling artifact at
each region's final observed month) but has not been added to the manuscript.

### 8.2 Poisson regression vs. a naive persistence baseline

Table 9 reports the supplementary Poisson regression against an intercept-only null
model (training mean rate): Full Model MAE 1.259 / RMSE 2.450 / McFadden $R^2$ 0.123 /
CCC 0.143; Events Only MAE 1.242 / RMSE 2.422 / McFadden $R^2$ 0.186 / CCC 0.240. A
post-hoc check of that same null baseline (predicting the training mean rate, 0.674, for
every test row) gives MAE 1.341 / RMSE 2.640 — confirming the tuned model clears it.

A second post-hoc baseline, not reported anywhere in the manuscript: predicting each
region's previous month's count as this month's count (naive persistence) gives MAE
1.220 / RMSE 2.323 across the same 2,035 test rows — slightly *better* than the tuned
Poisson GLM (Events Only: MAE 1.242 / RMSE 2.422) on both raw error metrics. The
McFadden $R^2$ / CCC framework used in Table 9 does not test against this baseline, only
against the null. This is an asymmetry with the binary classification task, where the
main model clears a persistence baseline by a wide margin (Table 3: 0.741 vs. 0.570) —
the count-regression supplement has not been checked against, or shown to beat, the
equivalent naive baseline.

### 8.3 HuRiViRe/ACLED provenance correction in round-2 documents

`ROUND2_REVIEWER_ASSESSMENT.md`, `RESPONSE_TO_REVIEWERS.md`, and
`CO_AUTHOR_BRIEFING.md` each contained one instance of "HuRiViRe" framing (a
four-watchdog dataset-consolidation description) in the justification for declining
Reviewer 3's country/date-extension request. The manuscript itself makes no mention of
HuRiViRe anywhere and describes its data source exclusively and consistently as ACLED
(`load_acled_data`, the raw file `ACLED Data_2025-12-31_Nigeria_Mexico_Myanmar.csv`, and
the Data Availability section pointing to acleddata.com). All three documents have been
corrected to say "ACLED extract" in place of "HuRiViRe dataset," preserving the original
argument structure (scope was fixed at collection time; extension is future work). A
repository-wide search confirms no remaining HuRiViRe or "four watchdog" references in
code, docs, or the manuscript. Whether HuRiViRe describes a separate or later research
effort, or should have been part of this submission's data description, is unresolved —
see the pending list below.

---

## Pending — for discussion at the co-author meeting

Four items, none of which block re-submission on their own, but each changes either the
response letter's wording or the manuscript's scope depending on how it's resolved:

1. **Country/date-range extension (Reviewer 3, point G).** Current response letter
   frames the fixed three-country, 2018–2024 scope as a future-work item given the cost
   of re-collecting and re-validating a larger ACLED extract. Decide whether to commit to
   a timeline for this as follow-up work, or leave the response as an open-ended
   deferral.
2. **Poisson regression supplement placement (Reviewer 3, point H).** Currently in the
   main text as Table 9. Decide whether it stays there, moves to supplementary material,
   or whether to add the persistence-baseline comparison from Section 8.2 before either
   placement — the supplement's evaluation currently checks against a null baseline only.
3. **Title unchanged despite the outcome-terminology rename (Reviewer 2, point 1).** Body
   text now reads "abduction/forced-disappearance events" throughout; the title still
   reads "Predicting Enforced Disappearances." Decide whether the abstract's opening
   sentence sufficiently resolves this or whether the title itself should change.
4. **HuRiViRe provenance (Section 8.3, new this round).** Confirm whether HuRiViRe is
   unrelated to this submission (in which case no manuscript change is needed beyond the
   document corrections already made) or whether the Data section should describe a
   broader data-consolidation effort that isn't currently reflected in the text or code.

Not blocking, but worth a mention if item 1 or 2 comes up: the last-observed-month
labeling defect (Section 8.1) is unfixed in the pipeline and unmentioned in the
manuscript; the three ad hoc checks against it did not move any conclusion, so it is a
footnote-level item, not a rerun.
