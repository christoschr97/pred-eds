# Round 3 Decisions — EPJ Data Science Major Revision

Companion to `docs/ROUND2_REVIEWER_ASSESSMENT.md`. Records what was decided, on what
evidence, and which reviewer points each decision closes. Every number cited here is read
from a file in `results/`; the file is named in each case.

---

## Settled

### S1. Report the five-country sample only

**Decision:** all round-3 results come from the five-country ACLED extract — Nigeria,
Mexico, Myanmar, Afghanistan, Syria. Three-country results are not reported.

Closes the country half of **Reviewer 3, point G**. Five countries span West Africa,
Latin America, Southeast Asia, South-Central Asia and the Middle East.

Two exclusions apply at load time and both need one sentence in §3.1:

- Four maritime pseudo-entities (Atlantic Ocean, Indian Ocean, Mediterranean Sea,
  Pacific Ocean; 149 events). These are not administrative units, so the region-month
  unit of analysis is undefined for them.
- Taiwan. 6,190 events across 21 admin1 units, of which 6,093 are protests and **zero**
  are abduction/forced disappearance. It falls outside the sampling frame the original
  design applied — countries with documented disappearance activity in the watchdog
  sources — and entered the extract as an artifact of how the ACLED pull was specified.
  Including it would have *raised* reported AUROC by 2.9 points (0.823 → 0.852) and
  [2026-09-12 note: `src/data_prep.py` lines 52–53 document this same comparison as
  3.1 points, 0.8174 → 0.8483, with AUPRC 0.8088 → 0.8081. The two figures come from
  different working runs and neither has a file in `results/`. §3.1 of the manuscript
  and the response letter quote the `data_prep.py` figures, since that is the in-repo
  source tied to the exclusion constant. Open item for the verification pass: either
  run the with-Taiwan configuration once into `results/` or drop the point estimates
  from the paper and state the direction only.]
  widened the margin over persistence by 1.5 points while leaving AUPRC flat, purely by
  supplying easy negatives. Excluding it lowers our reported numbers, which is the
  defense against a cherry-picking objection.

### S2. No editor-facing statement on the target-index change

**Decision:** the target-construction change is not raised with the editor or in the
response letter. The manuscript under review is unpublished, the resubmitted version
states and implements the same one-month horizon, and every number moves anyway on the
new sample.

Two constraints follow, and they are binding on later steps:

1. No sentence in the response letter or manuscript may characterise what the round-2
   numbers represented, and none may state or imply that the pipeline was rerun unchanged
   on additional countries. Silence is available; a false methodology statement is not.
2. If a reviewer or the editor asks directly about the target definition, answer plainly.

An internal note for Constantinos and Nikandros is in
`docs/INTERNAL_NOTE_target_index.md`.

---

## Settled 8 Sep 2026

### S3. Reviewer 3, point H — count/regression target

The reviewer wants counts predicted, not just a binary threshold, with a proper
regression metric suite, and pre-empts the paper's rare-event defense ("RMSE is no
excuse").

`results/exp6_poisson_regression.json` already exists and already carries the metric
suite he asked for. Read from that file:

Five-country sample, from `results/exp6_poisson_regression.json`:

| | Full (16 feat) | Events only (4 feat) |
|---|---|---|
| Test MAE | 1.7752 | **1.6860** |
| Test RMSE | 7.4121 | **6.4769** |
| Test Poisson deviance | **2.8915** | 2.9018 |
| Test McFadden $R^2$ | **0.3615** | 0.3599 |
| Test Lin's CCC | 0.4636 | **0.4695** |

Events-only wins on three of five metrics and loses two by margins in the fourth decimal,
so the reading is that the 12 network features add nothing to a count target either — the
same conclusion the classification ablation reaches, reached on a different target with a
different model class. It is not the stronger claim that events-only dominates.

The count model is moderately predictive on this sample: McFadden $R^2$ 0.36, Lin's CCC
0.47. That removes an argument we might have made and should not — we cannot defend the
binary target by claiming counts are unpredictable here. The defense is operational
instead: RMSE at 6.5–7.4 against MAE at 1.7 indicates a heavy right tail, so the count
model's error is dominated by a few large months, and a threshold-crossing trigger is what
the monitoring use case consumes. `sanity_check_binary_match_rate` is 1.0, so the count
target binarizes exactly to the classification target.

The three-country figures previously recorded here are superseded and not comparable: that
run's count lookup was aligned to the earlier target index, which its own sanity check
flagged on the corrected data (72.87% match) before the lookup was keyed directly on the
target month.

**Decision: include it as an appendix**, with the full metric suite for both feature sets
and two sentences in §5 pointing to it. It answers the reviewer with the metrics he
specified, costs a rerun of a script that takes under a second, and converts an
"out of scope" response into independent corroboration of the ablation result.

### S4. Reviewer 3, point D — which model is primary

> **Corrected 8 Sep 2026, and superseded in part by S5 below.** As first written, this
> entry's table cited `lr_vs_lstm_corrected.csv`, which does not exist in the repository
> and which no script produces. The numbers were assembled in a working session and never
> written to a file, and several did not match the saved results. The table below is
> re-derived from files that exist. The events-only row in particular was not a saved run
> at the time; `exp10_bootstrap_events_only.py` is the first reproducible version of it,
> and its result changes this decision — see S5.

Tuned logistic regression is statistically indistinguishable from the Attention-LSTM.
Five-country sample, corrected target. Differences and tests are computed on the
3-seed ensemble-averaged probabilities, which is what exp7 and exp10 resample:

| Feature set | Tuned LR | LSTM seed mean | LSTM ensemble | ensemble − LR | 95% CI | perm $p$ |
|---|---|---|---|---|---|---|
| Full (16) | 0.8109 | 0.8081 ± 0.0013 | 0.8099 | −0.0010 | [−0.0065, +0.0045] | 0.8047 |
| Events (4) | 0.8115 | 0.8140 ± 0.0003 | 0.8143 | +0.0028 | [−0.0002, +0.0059] | 0.4444 |

Sources: `results/exp1_tuned_lr.json` (LR), `results/exp8_architecture_comparison.json`
(full-model seed mean), `results/exp7_bootstrap_lr_vs_lstm.json` (full-model test),
`results/exp10_bootstrap_events_only.json` (events-only test).

The train→test generalization-gap defense currently in the manuscript is retired
regardless of this decision. On the five-country sample every model shows test AUPRC
above train — Attention-LSTM by 0.0439, tuned LR by 0.0747, untuned LR by 0.0714,
XGBoost by 0.0505 — and the two logistic regressions show the largest margins, the
opposite of the manuscript's claim. Training prevalence is 31.1% against 45.9% in
test, and AUPRC's baseline is the positive rate, so the quantity was tracking the
prevalence shift rather than generalization. It was not a valid comparison in either
round.

**Decision: keep the Attention-LSTM primary, report tuned LR as co-equal**, with the
bootstrap intervals and permutation p-values stated plainly in §5. The
paper's central claim is negative — 12 network structure features add nothing beyond 4
event counts. Established from a linear model alone, that claim is answerable in one
line: a linear model on flattened lags cannot represent interactions among centrality,
clustering and edge weights. Established from an attention model that *can* represent
them, it holds. The LSTM earns its place as the model that could have found the signal
and did not, and the LR result independently corroborates it: if the signal were a deep
nonlinear temporal interaction, a linear model could not match a sequence model. It
matches.

Promoting LR would also orphan the §2.4 LSTM literature review and require replacing the
existing SHAP figures with a coefficient analysis.

---

### S5. Reviewer 3, point D — revised: two nulls, LSTM retained as the analysis vehicle

S4 kept the Attention-LSTM primary partly on accuracy. Two results retire that basis.

First, on the full feature set the ranking inverts. Tuned LR 0.8109, untuned LR 0.8094,
Standard LSTM 0.8089, Attention-LSTM 0.8081, XGBoost 0.7920 — the top four within
0.0028, and the *untuned* linear model ahead of both sequence models
(`results/exp8_architecture_comparison.json`).

Second, the events-only margin that S4 relied on does not exclude zero.
`exp10_bootstrap_events_only.py` applies exp7's design — same seeds, same LSTM
configuration, same GridSearchCV grid, B=10,000, P=10,000, same paired resampling and
label-swap null — to the four-feature set. Ensemble difference +0.0028, bootstrap 95% CI
[−0.0002, +0.0059], P(diff ≤ 0) = 0.0366, permutation two-sided p = 0.4444. Seed spread
is 0.0003, so the margin is not seed noise; it is test-set sampling noise.

**Decision: report parsimony as the finding. Two null results, and the Attention-LSTM
stays in the paper as the instrument for the temporal attention analysis rather than as
the best-performing model.**

The two nulls are: 12 network structure features add nothing beyond 4 event counts
(replicated across LSTM +0.0052, tuned LR +0.0006, XGBoost +0.0134); and no sequence
architecture improves on a tuned linear model on those same 4 features (exp10). Each
strengthens the other. The network null established from a linear model alone invites the
objection that a linear model on flattened lags cannot represent interactions among
centrality, clustering and edge weights. Established from an attention model that can
represent them, it holds. The architecture null then says the converse: if the signal
were a deep nonlinear temporal interaction, a linear model could not match a sequence
model. It matches.

What this does not change: the §2.4 LSTM literature review stays, the SHAP figures stay,
and the title stays. The Attention-LSTM is what makes the timestep attribution possible,
and that analysis is what answers the reviewer on temporal structure. What changes is
every sentence that ranks the models or claims a generalization advantage.

The paper is not weakened by this. The events-only model reaches 0.8140 AUPRC against a
0.459 prevalence baseline and 0.5967 persistence. What changes is which claim is being
made about it.

---

## Flagged for Text tranche B — changes the framing strategy

`results/exp4_zero_history_subset.json`, five-country sample:

- Zero-history regions (no disappearances anywhere in the training period): **8 of 135**.
- Test sequences from those regions: **160**, positive rate 0.175.
- Attention-LSTM without past-disappearance features: **AUPRC 0.1387 ± 0.0100** against a
  random baseline of **0.175**. The same model on the full test set scores 0.7191 ± 0.0101.

The model is below the no-skill baseline on the subset where Reviewer B says the
early-warning claim lives, and it drops from 0.72 to 0.14 when the evaluation moves from
regions with history to regions without. The three-country run gave the same direction
(0.2166 against a 0.2979 baseline on 47 sequences from 2 regions), so this replicates
across samples rather than resting on one split.

160 sequences with 28 positives still will not carry a tight interval, so the claim to make
is that predictive skill in previously unaffected regions is not demonstrated — not that
skill is absent. Reviewer B's point is then answered with a measurement instead of a
vocabulary change, and the framing that follows is recurrence forecasting in affected
regions with onset scoped out explicitly and quantified.
