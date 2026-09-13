# Manuscript delta — submitted text vs. five-country regeneration

Round 3. For Christos, Constantinos, Nikandros.

Every line number refers to `paper/sn-article.tex` as submitted. Every "new"
value is traceable through `docs/numbers_of_record.csv` to a file in
`results/`. Nothing in this document is an estimate.

**How to read the status column**

| status | meaning |
|---|---|
| SWAP | replace the number, sentence structure survives |
| REWRITE | the claim itself changes, not just the digits |
| UNCHANGED | verified as still correct on the new sample |

---

## 1. The sample changed

The analysis sample is now five countries. `data_prep.load_acled_data`
excludes Taiwan and the four maritime pseudo-entities, so the post-exclusion
figures below — not the raw export totals — are what belongs in the paper.

| quantity | submitted | new | status |
|---|---|---|---|
| Countries | Nigeria, Mexico, Myanmar | + Afghanistan, Syria | REWRITE |
| Conflict events | 202,347 | 360,530 | SWAP |
| Disappearance events | 5,959 (2.9%) | 11,483 (3.19%) | SWAP |
| Arrest events | 6,913 (3.4%) | 8,371 (2.32%) | SWAP |
| Violent events | 114,564 (56.6%) | 252,808 (70.1%) | SWAP |
| Total fatalities | 197,817 | 438,917 | SWAP |
| admin1 regions | 87 | 135 | SWAP |
| Date range | 2018–2024 | 2018–2024 | UNCHANGED |
| Sequences | 6,293 | 9,696 | SWAP |
| Train sequences | — | 6,660 | SWAP |
| Test sequences | 2,035 | 3,036 | SWAP |
| Train positive rate | 37.3% | 31.1% | SWAP |
| Test positive rate | 48.3% | 45.9% | SWAP |
| Class weight w+ | 1.683 | 2.213 | SWAP |
| Random baseline | 0.483 | 0.459 | SWAP |

Regions by country: Nigeria 37, Afghanistan 34, Mexico 32, Myanmar 18,
Syria 14.

Disappearance events by country: Syria 4,841, Nigeria 2,899, Myanmar 1,744,
Mexico 1,472, Afghanistan 527.

Source: `results/descriptives_5c.json`, `results/logs/run_ablations.log`.

**Line-level:**

- L155 — 202,347 → 360,530; 2.9% → 3.19%; and the region-month imbalance
  range "37--48\% positive" → 31--46%. SWAP
- L162 — 2.9% → 3.19%. SWAP
- L208 — 2.9% → 3.19%. SWAP
- L276 — all three descriptives. SWAP
- L281 (Fig. 2 caption) — 48.3% vs. 37.3% → 45.9% vs. 31.1%. SWAP
- L352 — train July 2018–Dec 2022, test Jan 2023–Dec 2024. UNCHANGED
- L366 — 37.3% positive / 62.7% negative → 31.1% / 68.9%. SWAP
- L372 — `w+ = (1-p)/p = 0.627/0.373 = 1.683` → `0.689/0.311 = 2.213`.
  SWAP. Note the displayed rates are rounded; the exact training prevalence
  gives 2.213, which is the value the code uses. Do not write 2.215.
- L390 (Table 3 caption) — 6,293 sequences → 9,696. SWAP

---

## 2. Ablation table (L488–L498)

Sorted by test AUPRC, mean ± SD over 3 seeds. Source
`results/ablations_5c.json`.

| # | configuration | submitted | new | Δ vs. full (new) | feats |
|---|---|---|---|---|---|
| 1 | Events Only ★ | 0.745 ± 0.003 | **0.8140 ± 0.0003** | +0.0052 | 4 |
| 2 | Events + Structure | 0.745 ± 0.002 | 0.8130 ± 0.0009 | +0.0042 | 9 |
| 3 | Events + Clustering | 0.745 ± 0.003 | 0.8130 ± 0.0004 | +0.0042 | 6 |
| 4 | Events + Centrality | 0.741 ± 0.001 | 0.8126 ± 0.0008 | +0.0038 | 7 |
| 5 | Events + Weights | 0.743 ± 0.001 | 0.8118 ± 0.0010 | +0.0030 | 6 |
| 6 | Full Model | 0.741 ± 0.003 | 0.8088 ± 0.0006 | — | 16 |
| 7 | No Fatalities | 0.740 ± 0.003 | 0.8080 ± 0.0010 | −0.0008 | 15 |
| 8 | No Disappearances | 0.666 ± 0.006 | 0.7225 ± 0.0033 | −0.0863 | 15 |
| 9 | Network Only | 0.664 ± 0.003 | 0.6930 ± 0.0048 | −0.1158 | 12 |
| — | Random Baseline | 0.483 | 0.459 | −0.3498 | — |

The rank order changed. Events Only moves from rank 2 to rank 1, and
Events + Centrality from 6 to 4. Every row is a SWAP, but the sorted
sequence must be regenerated, not edited in place.

**Prose depending on this table:**

- L443 (caption) — "−10.1%" → −10.7%; 0.666 → 0.7225; random 0.483 → 0.459.
  Also REWRITE: the caption says all top-tier deltas are within ±0.004;
  Events Only is now +0.0052 above full. SWAP + REWRITE
- L503 — "top tier (ranks 1–7, AUPRC 0.740–0.745)" → 0.8080–0.8140. SWAP
- L505 — events-only outperforms full. UNCHANGED (margin grew 0.004 → 0.0052)
- L507 — "0.741 to 0.666 (7.5-point drop, 10.1%)" → "0.8088 to 0.7225
  (8.6-point drop, 10.7%)"; and "remains 18.3 points above random" → 26.4
  points. SWAP. The second half of that sentence — that arrests, violence
  and fatalities alone identify new outbreak regions — is contradicted by
  §8 and needs the qualifier described there.
- L509 — "0.741 to 0.740" → "0.8088 to 0.8080". SWAP
- L655 — "10.1% of performance" → 10.7%. SWAP
- L471 — "≈0.74, 25.8-point improvement over random and 17.1 points above
  persistence" → ≈0.81, 35.0 points over random, 21.2 above persistence
  (full model). For Events Only the figures are 35.5 and 21.7. SWAP
- L176 — "AUPRC 0.666, 9.6 points above [persistence]" → 0.7225, 12.6
  points above persistence. SWAP
- L643, L661, L588 — 0.745 → 0.8140. SWAP
- L594 — "+0.004 AUPRC with 75% fewer features" → +0.0052. SWAP
- L263 (Fig. 1 caption) — country list REWRITE; "AUPRC 0.745" → 0.8140;
  "27 experiments" (9 × 3 seeds) UNCHANGED.

### 2a. Network categories were never a ranking — REWRITE

The four network-category configurations span 0.0012 (0.8130 down to
0.8118). The largest seed-to-seed SD among them is 0.0010. The spread does
not exceed the noise. Any sentence that says one network category
contributes more than another has to go, wherever it appears. This is not a
number swap.

---

## 3. Architecture comparison, Table 4 (L460–L466)

Full 16-feature set. The submitted "Gap" column is train AUPRC − test AUPRC,
matching `models.py`. `exp1` and `exp2` print the opposite sign — do not copy
their `gap` field into this table; derive it from their train/test AUPRC.

Sorted by new test AUPRC. Gap is train − test.

| model | submitted test | new test | new train | new gap | new Brier |
|---|---|---|---|---|---|
| Tuned LR | 0.738 | **0.8109** | 0.7362 | −0.0747 | 0.1818 |
| LR (untuned) | 0.717 | **0.8094** | 0.7380 | −0.0714 | 0.1829 |
| Standard LSTM | 0.740 ± 0.003 | **0.8089 ± 0.0005** | 0.7584 | −0.0505 | 0.1805 |
| Attention-LSTM | 0.741 ± 0.003 | **0.8081 ± 0.0013** | 0.7642 | −0.0439 | 0.1860 |
| XGBoost | 0.723 | **0.7920** | 0.7415 | −0.0505 | 0.1948 |
| Persistence | 0.570 | 0.5967 | — | — | 0.3073 |
| Random | 0.483 | 0.459 | — | — | — |

Sources: exp8 for both LSTMs and untuned LR, exp1 for tuned LR, exp2 for
XGBoost, exp3 for persistence.

exp8's independent Attention-LSTM run (0.8081 ± 0.0013) reconciles with
`run_ablations` full-model (0.8088 ± 0.0006) inside one SD.

### 3b. The submitted ranking inverts — REWRITE

Submitted order: Attention-LSTM > Standard LSTM > tuned LR > XGBoost >
untuned LR. New order: tuned LR > untuned LR > Standard LSTM >
Attention-LSTM > XGBoost.

The top four now sit within 0.0028 of each other, and the **untuned**
logistic regression beats both sequence models. Any sentence ordering these
models has to go. L471's account — that tuning narrowed a gap which partly
reflected insufficient regularization — is now stronger than written, not
weaker: regularization was the whole story, since even the unregularized
default beats the LSTMs.

### 3c. The generalization argument is gone — REWRITE, and it needs a decision

Submitted (L136, L471): the LSTMs show negative train−test gaps (−0.036,
−0.039) while tuned LR and XGBoost show positive gaps (+0.049, +0.018), so
the temporal models generalize better under distributional shift. That
contrast is what justified keeping a sequence model.

On five countries **every** model has a negative gap, and the two logistic
regressions have the most negative ones (−0.0747, −0.0714) — the ordering is
exactly reversed.

The mechanism is visible in the pipeline's own diagnostic: train prevalence
31.1%, test prevalence 45.9%, a 14.8-point jump. AUPRC's baseline is the
positive rate, so a higher-prevalence test set raises AUPRC mechanically.
A negative train−test AUPRC gap across sets with different prevalence is
expected for any model and says nothing about generalization. The gap
ordering just tracks which model had the lowest train AUPRC.

This cuts both ways and should be stated plainly: it is a correct
methodological point that improves the paper, and it removes an argument
the paper currently relies on. Comparing AUPRC across sets of unequal
prevalence is not a valid generalization test in either round.

### 3d. L473 — the LSTM-versus-LR sentence reverses sign, REWRITE

Submitted: a 0.003 AUPRC gap with the Attention-LSTM ahead of tuned LR.

New (`exp7_bootstrap_lr_vs_lstm.json`): tuned LR 0.8109, LSTM ensemble
0.8099, LSTM 3-seed mean 0.8088 ± 0.0006. Difference (LSTM ensemble − LR)
= −0.00097, 95% paired bootstrap CI [−0.00649, +0.00455], paired
permutation two-sided p = 0.8047.

The linear model is now nominally ahead and the interval straddles zero.
The sentence cannot be repaired by changing digits. Per
`DECISIONS_ROUND3.md` the Attention-LSTM stays primary with tuned LR
reported as co-equal, which is what this interval supports.

Two housekeeping items in that JSON: the `description` string still says
`n=2035` (the three-country test size; actual 3,036), and
`numbers_of_record.csv` labels 0.8088 as the value entering the difference
when the difference uses the ensemble 0.8099. Both corrected in the results
table; the stale JSON string is cosmetic and left as produced.

---

## 4. Events-only vs. full across model families (L524–L526)

| family | submitted events / full / Δ | new events / full / Δ | status |
|---|---|---|---|
| Attention-LSTM | 0.745 / 0.741 / +0.004 | 0.8140 / 0.8088 / +0.0052 | SWAP |
| Tuned LR | 0.743 / 0.738 / +0.005 | 0.8115 / 0.8109 / +0.0006 | SWAP |
| XGBoost | 0.733 / 0.723 / +0.010 | 0.8054 / 0.7920 / +0.0134 | SWAP |

All three deltas remain positive, so **L513 and the corresponding sentence
in L707 hold: the network null replicates across LSTM, logistic regression
and XGBoost.** This is the paper's central claim and the expanded sample
strengthens it. UNCHANGED as a claim, SWAP on the digits.

Note the LR delta shrank to +0.0006, which is within rounding of zero. The
claim is that network features add nothing, so a delta at zero supports it;
just avoid describing the events-only model as outperforming for the LR
family.

---

## 5. SHAP table (L546–L549)

Events Only model, mean |SHAP| over all timesteps.
Source `results/exp5_shap/shap_summary.json`.

| feature | submitted | new | status |
|---|---|---|---|
| Prior disappearances | 0.220 | 0.3125 | SWAP |
| Arrests | 0.018 | 0.0205 | SWAP |
| Violence | 0.029 | 0.0173 | SWAP |
| Fatalities | 0.046 | 0.0162 | SWAP |

**The order of the three minor features reversed completely** — submitted
had fatalities > violence > arrests, the new run has arrests > violence >
fatalities. All three now fall within 0.0043 of each other. Presenting them
as ranked requires a seed-level stability check that has not been run. Safer
text: prior disappearances dominate at roughly 15× the next feature, and the
remaining three are not separable. REWRITE.

- L535 — "mean |SHAP| 0.220" → 0.3125; and "approximately five times the
  next most important feature" → approximately 15 times. SWAP
- L554 — most recent month t contributes 0.1333. SWAP

### 5a. Timestep gradient

Mean |SHAP| per timestep: t−5 0.0793, t−4 0.0738, t−3 0.0721, t−2 0.0808,
t−1 0.1106, t 0.1333.

The recency ratio t / t−5 is 1.68, not the 1.5 stated at L707. SWAP.

The gradient is not monotonic: the three oldest months sit within 0.007 of
each other and t−5 is slightly above t−4 and t−3. The rise is confined to
the recent half of the window. "Recency gradient" (L136) is still defensible
as written, but a claim of a monotonic decay across all six months is not.
- Table "Direction" column (Positive / Mixed) — not recomputed. Either
  re-derive the signs from `shap_values.npy` or drop the column.

---

## 6. Country-level performance, Table 8 (L609–L613)

Submitted: Nigeria 37 / 740 / 41.2% / 0.758; Mexico 32 / 672 / 53.6% /
0.741; Myanmar 18 / 623 / 50.4% / 0.736; Overall 87 / 2,035 / 48.3% / 0.745.

Nothing in the codebase computed this breakdown.
`exp9_country_breakdown.py` was written for it: one pooled Events Only
Attention-LSTM per seed, scored separately within each country's test
sequences. Not per-country models.

| country | regions | test seq | test prevalence | AUPRC | lift over own prevalence |
|---|---|---|---|---|---|
| Afghanistan | 34 | 674 | 0.232 | 0.5712 ± 0.0004 | +0.340 |
| Mexico | 32 | 757 | 0.362 | 0.5954 ± 0.0016 | +0.233 |
| Myanmar | 18 | 419 | 0.637 | 0.8935 ± 0.0009 | +0.256 |
| Nigeria | 37 | 861 | 0.571 | 0.8040 ± 0.0002 | +0.233 |
| Syria | 14 | 325 | 0.634 | 0.9599 ± 0.0002 | +0.326 |
| Overall | 135 | 3,036 | 0.460 | 0.8144 ± 0.0000 | +0.355 |

### 6a. Raw AUPRC is not comparable across these countries — REWRITE

Raw AUPRC spans 0.3887, from Afghanistan at 0.5712 to Syria at 0.9599. The
largest within-country seed SD is 0.0016. Unlike the network categories,
this spread is real — roughly 240 times the noise.

But it is mostly a prevalence effect. Test positive rates run from 0.232
(Afghanistan) to 0.637 (Myanmar), and AUPRC's baseline is the positive
rate. Measured as lift over each country's own prevalence, the spread falls
from 0.3887 to 0.1071, and the ordering changes: Afghanistan goes from worst
on raw AUPRC to best on lift.

Consequences for the text:

- L598 — "performance is relatively consistent across three distinct
  conflict contexts (range 0.736–0.758)" is **false on raw AUPRC** for five
  countries (range 0.571–0.960). It is defensible on lift (range
  0.233–0.340). REWRITE, and state which quantity is being compared.
- L618 — "2.2-percentage-point spread" → 38.9 points raw, 10.7 points on
  lift. REWRITE.
- Table 8 needs a prevalence column. Without it the table invites the
  reading that the model fails in Afghanistan and Mexico, when both clear
  their own baselines by more than Nigeria does.

---

## 7. Poisson appendix (L633–L634, L639)

Source `results/exp6_poisson_regression.json`. Count-vs-binary target match
rate is 1.0 after the alignment fix.

| metric | Full submitted | Full new | Events submitted | Events new |
|---|---|---|---|---|
| MAE | 1.259 | 1.7752 | 1.242 | 1.6860 |
| RMSE | 2.450 | 7.4121 | 2.422 | 6.4769 |
| Deviance | 2.741 | 2.8915 | 2.454 | 2.9018 |
| McFadden R² | 0.123 | 0.3615 | 0.186 | 0.3599 |
| Lin's CCC | 0.143 | 0.4636 | 0.240 | 0.4695 |

**L639 is now false. REWRITE.** The submitted sentence says Events Only
beats Full on every metric. On the new sample it wins MAE, RMSE and Lin's
CCC, and loses deviance (2.9018 vs. 2.8915) and McFadden R² (0.3599 vs.
0.3615). Three of five, with the two losses inside the third decimal.

Also note the count model fits far better than it did on three countries
(McFadden R² 0.36 against 0.12–0.19). Any argument that leans on counts
being hard to forecast no longer holds — see `DECISIONS_ROUND3.md` v2.

- L622 — recomputed. Test-period region-months: 3,036 (matches the test
  sequence count exactly). Counts range 0 to 47 (submitted 0–33), mean 1.69
  (1.30), s.d. 4.20 (2.55), 54.1% zero (51.7%). SWAP.
  Source `results/test_count_distribution_5c.json`.

---

## 8. Onset / zero-history regions (L511)

Submitted: of 87 regions, only 2 (Ayeyarwady, …) had no training-period
disappearance history.

New (`results/exp4_zero_history_subset.json`): 8 of 135 regions, 160 test
sequences, subset positive rate 0.175, subset AUPRC 0.1387 ± 0.0100, while
the same model scores 0.7191 ± 0.0101 on the full test set.

The subset AUPRC is **below** its own prevalence baseline. On genuinely
new-onset regions the model does not beat chance. This is the reviewer's
early-warning objection and the number now answers it directly — against us
on that subset, which is the honest finding and is reportable as a scope
limit. REWRITE, and it also governs L675 (currently commented out) and any
sentence claiming forecasting value where the lagged disappearance feature
is withheld. L443's version of that claim is true for the pooled test set
(0.7225 vs. 0.459) and false for the zero-history subset; the text must say
which.

---

## 8a. Abstract (L136) and conclusion (L707) — highest-visibility edits

Both carry the full set of headline numbers, and both carry the two claims
that change character rather than value.

Numbers: 202,347 → 360,530; 87 → 135; three countries → five; 0.741 →
0.8088; random 0.483 → 0.459; persistence 0.570 → 0.5967; 0.666 → 0.7225;
"9.6 points above persistence" → 12.6; and in L707 the recency ratio 1.5 →
1.68.

Two claims need more than that:

1. **"Tuned non-temporal baselines match LSTM accuracy but generalize worse
   under distributional shift, motivating the temporal model for
   deployment."** This is the sentence that keeps the LSTM primary, and it
   now rests entirely on the train–test gap signs rather than on accuracy,
   because tuned LR is nominally ahead on accuracy (§3d). **The gaps have
   now landed and they do not support it** — every model has a negative gap
   and the two logistic regressions have the largest ones (§3c). The
   sentence has to be deleted, not reworded. What replaces it is a decision,
   not a wording choice; see §12.
2. **"A No Past Disappearances ablation provides evidence of forecasting
   value beyond naive persistence"** (L136), and in L707 the same result
   billed as the strongest methodological finding. It holds on the
   pooled test set. It fails on the eight regions with no training-period
   history (§8), which is the subset the claim implies. The abstract and the
   conclusion are the two places a reviewer will check this, so the
   qualifier belongs in both: retained performance is measured on regions
   that mostly do have prior history.

---

## 9. Sections needing argument, not numbers

- **L271, country selection.** Must now justify five countries. The
  reviewer's scope objection also has a date-range half — the new export
  spans the same 2018–2024 window — and no experiment addresses that. It
  needs a written defense.
- **L200, study-country contexts.** Afghanistan and Syria need the same
  treatment Mexico, Myanmar and Nigeria get. Related: Syria now contributes
  the most disappearance events (4,841) while the introduction foregrounds
  Mexico. Which case leads is a narrative choice the new sample forces.
- **L414–L422**, LR and XGBoost setup prose — check against the corrected
  class weight once the reruns land.

---

## 10. Defects found during this audit

1. **Stale class weight in `exp1` and `exp2`.** Both hardcoded
   `1.683`, derived from the three-country training prevalence of 0.373.
   Every other script computes `w+ = (1-p)/p` from `y_train`. Fixed by
   recomputing from the training split at the same point the other scripts
   do; both experiments are rerunning. The affected values are the tuned LR
   and XGBoost rows.
2. **Opposite `gap` conventions.** `models.py` returns train − test;
   `exp1`/`exp2` report test − train. Left as written, flagged here.
3. **`train.py` cannot run in this tree.** It imports
   `from .models import ...` while `src/` has no `__init__.py`. It holds the
   original Table 4 comparison, so that table was not reproducible.
   `exp8_architecture_comparison.py` reproduces it by calling `models.py`
   directly, as `run_ablations.py` and `exp4` do. `train.py` left untouched.
4. **Model selection touches the test set.** `train_lstm` early-stops on
   test loss and restores the best-by-test-loss state. This is in the
   original publication code and is unchanged, but it is a reviewer-exposed
   choice and it explains the negative generalization gaps in Table 4.

---

## 11. One decision this audit forces

`DECISIONS_ROUND3.md` recorded "keep the Attention-LSTM primary, report
tuned LR as co-equal." That decision was made when the only problem was
that LR had drawn level on accuracy. It now has to survive a second
problem: the generalization argument that justified the choice does not
hold (§3c), and on the full feature set even an untuned LR is ahead (§3b).

Three ways forward.

**A. Keep the Attention-LSTM primary, justified on the events-only feature
set and on interpretability.** On four features the LSTM leads: 0.8140
against tuned LR 0.8115, a margin of +0.0025. The temporal-attention SHAP
analysis (§5, §5a) is something no flat model provides, and it is what
answers the reviewer on temporal structure. Weakness: +0.0025 has not been
tested. exp7's paired bootstrap was run on the *full* feature set, not on
the recommended model. Running the same design on events-only is one
script change and it decides whether this option is defensible.

**B. Make parsimony the headline.** Report that a four-feature model
reaches 0.814 and that a tuned linear model on the same four features
reaches 0.8115, so neither network features nor sequence architecture is
required. This is the strongest empirical reading and it turns the LR
result from a threat into a second null finding alongside the network null.
Cost: the largest rewrite, and the title's framing of an LSTM analysis
becomes odd.

**C. Minimal edit.** Delete the generalization sentence, report the models
as statistically indistinguishable per exp7, keep everything else. Cheapest,
and leaves the paper open to the question of why a sequence model at all.

### 11a. The events-only test was run, and it rules out option A

`exp10_bootstrap_events_only.py` applies exp7's design — same seeds, same
LSTM configuration, same GridSearchCV grid, same B=10,000 and P=10,000,
same paired resampling and label-swap null — to the four-feature set. Only
the feature subset differs.

| quantity | events-only (exp10) | full model (exp7) |
|---|---|---|
| Attention-LSTM ensemble | 0.8143 | 0.8099 |
| Tuned LR | 0.8115 | 0.8109 |
| Difference (ensemble − LR) | +0.0028 | −0.0010 |
| Bootstrap 95% CI | [−0.0002, +0.0059] | [−0.0065, +0.0045] |
| Bootstrap P(diff ≤ 0) | 0.0366 | 0.6291 |
| Permutation two-sided p | 0.4444 | 0.8047 |

Test set n=3,036 in both. LSTM seed spread is 0.0003, so the +0.0028
margin is about nine seed SDs — it is not seed noise. It is test-set
sampling noise: the bootstrap CI includes zero, with a lower bound of
−0.0002, and the permutation test is nowhere near significance at p=0.44.

The two tests differ in emphasis and both should be reported. The
one-sided bootstrap quantity P(diff ≤ 0) = 0.0366 is the most favourable
reading available, and it is not enough to carry a primary-model claim
when the interval it comes from includes zero and the permutation null
(sd 0.0036) places the observed difference well inside it.

Conclusion: on the paper's recommended feature set the Attention-LSTM and
a tuned linear model are not distinguishable. Option A is not available.

Three model families, four features, and now a formal test all agree:
neither the network features nor the sequence architecture is required.
That is option B, and it is the position the evidence supports.

---

## 11b. The three original countries also changed, because the ACLED pull is newer

Regenerating Figure 1 exposed this. The submitted analysis used
`ACLED Data_2025-12-31_Nigeria_Mexico_Myanmar.csv`; the five-country
analysis uses `ACLED Data_2026-09-06_...`. Same date range in both
(2018-01-31 to 2024-12-31), but ACLED backfills, so the same three
countries carry more disappearances in the newer pull:

| Country | Submitted (Fig 1 panel label) | Regenerated | Change |
|---|---|---|---|
| Nigeria | 2,827 | 2,899 | +72 (+2.5%) |
| Mexico | 1,449 | 1,472 | +23 (+1.6%) |
| Myanmar | 1,683 | 1,744 | +61 (+3.6%) |
| **Three-country total** | **5,959** | **6,115** | **+156** |

Afghanistan adds 527 and Syria 4,841, for 11,483 disappearances across the
five countries. That total reconciles exactly with the loader's own count.

Two consequences. The data section must state the access date, because a
reviewer holding the submitted version next to the revision will find
different counts for countries we did not change. And the added-countries
description cannot be phrased as "we added two countries to the existing
sample" — the existing sample moved too. The accurate phrasing is that the
data were re-collected for five countries on 2026-09-06.

Loader-reported totals for the five-country pull: 360,530 events after
excluding 6,339 maritime and Taiwan records, 11,483 disappearances, 8,371
arrests, 252,808 violence events, 438,917 fatalities.

---

## 11b. Figures — all five regenerated

Every figure `paper/sn-article.tex` includes now comes from the five-country
run, and every one of them is produced by a script in `src/`:

| Figure | Producer | Status |
|---|---|---|
| `pipeline_diagram.png` (L262) | `create_pipeline_diagram.py` | **new script** — was hand-drawn, no source existed |
| `figure1_time_series.png` (L280) | `create_figures.py` | five panels; also fixed a stale import that meant the script had never run since migration |
| `ablation_chart.png` (L442) | `create_ablation_chart.py` | **new script** — was hand-drawn, no source existed |
| `fig2_shap_heatmap.png` (L558) | `exp5_shap.py` | five-country run, n = 3,036 test sequences |
| `fig3_shap_direction.png` (L567) | `exp5_shap.py` | five-country run, n = 3,036 test sequences |

`fig1_shap_bar.png` is generated but not included by the manuscript.

Three claims carried by figures changed, and the text has to follow:

1. **The ablation chart no longer names a best configuration.** The seven
   configurations that keep past-disappearance history span 0.8080 to 0.8140.
   That 0.0060 range is smaller than the ±0.0071 chance variation on this test
   set (1.96 × the paired permutation null SD of 0.003614, `exp10`). Events Only
   has the highest mean and remains the model to recommend on parsimony
   grounds — four features against sixteen — but the ranking among the seven
   cannot be read off these numbers.
2. **The pipeline diagram's attention annotation flips sign.** It read "+0.1%
   AUPRC". Attention scores 0.8081 against the plain LSTM's 0.8089, so it now
   reads −0.1%.
3. **Ablation percentages are quoted against the Full Model throughout.**
   −10.7% for `No Past Disappearances` and −14.3% for `Network Only`. Both
   configurations are the full 16-feature set minus one block, so the Full Model
   is their comparator; this matches `delta_from_full` in `ablations_5c.json`
   and the manuscript text. Quoting them against Events Only would give −11.2%
   and −14.9%.

Details and the full cell-by-cell record for the diagram: `PIPELINE_DIAGRAM_EDITS.md`.

---

## 11c. Text tranche B — the framing rebuild (reviewer points A, B, C)

Eleven edits to `paper/sn-article.tex`. Nothing here changes a result; the
edits change what the paper claims about its results.

**Abstract (rewritten).** Now built on the two nulls. The generalization-gap
sentence ("Tuned non-temporal baselines match LSTM accuracy but generalize
worse under distributional shift") is gone — that defense was retired in
step 1. Parsimony is stated as the finding. The closing sentence bounds
scope: recurrence forecasting, not first onset.

**Research questions.** RQ1 no longer leans on event-level rarity. RQ3 asks
whether skill extends to regions with no recorded history, rather than
asserting that we predict new outbreaks.

**Introduction — rarity paragraph.** Separates the two prevalence figures
(3.2% of events; 35.8% of region-months) and states plainly that the
operationalized target is not a rare-event problem.

**Introduction — contributions.** Rewritten as three items: the network
null replicated across three model families; the architecture null with its
paired bootstrap CI and permutation p; the measured boundary of skill.
The old second contribution (AUPRC 0.666 "evidence of early-warning
capability") is gone.

**§2.1 ViEWS passage.** The onset-difficulty finding is now treated as a
constraint on claims, not a problem our models solve.

**§2.3 retitled** from "Rare Events in Conflict: Methodological Foundations"
to "Event-Level Rarity, Class Balance, and Metric Choice". King and Zeng and
the rare-event corrections are explicitly scoped to event-level incidence
and explicitly disclaimed for our estimator. AUPRC is justified on
operational ranking grounds (precision at absorbable recall), not on
imbalance.

**§2.3 validation paragraph.** Notes that withholding the lagged feature is
necessary but not sufficient, and motivates the zero-history evaluation.

**§3.1 spatial unit (the flagged contradiction).** §2.1 correctly described
ViEWS as country-month and PRIO-GRID; §3.1 claimed to follow prior work at
admin1. The citation-based justification is replaced by a measured one:

| | admin1 | admin2 |
|---|---|---|
| units | 135 | 3,044 |
| median events per unit-month | 14 | 2 |
| share of unit-months with <5 events | 24.8% | 76.1% |
| observed / possible unit-months | 10,506 / 11,340 | 61,753 / 255,696 |

At admin2 the actor co-occurrence networks would be empty or trivial for
most of the panel. The passage now states admin1 as our choice, positioned
between the two levels the literature uses. Computed from the extract; rows
added to `numbers_of_record.csv` under section "Spatial unit".

**§3.2 prevalence passage and §3.2 count-target defense.** Prevalences
updated to five countries. The RMSE argument ("standard loss functions
tend to favor models that predict near-zero values") is replaced: the
Poisson appendix reaches McFadden R² 0.36 and CCC 0.47, so counts here are
not unpredictable and that defense was false. The defense is now
operational — RMSE 6.5–7.4 against MAE 1.7–1.8 shows error concentrated in
high-count months — and notes the count target binarizes to the
classification target at match rate 1.0.

**Two figure captions and the dataset composition sentence** updated to the
five-country extract: 360,530 events, 135 regions, 11,483 disappearances
(3.2%), 8,371 arrests (2.3%), 252,808 violent events (70.1%), 438,917
fatalities.

One discrepancy to note for the response letter: the No Past Disappearances
configuration scores 0.7225 in `ablations_5c.json` and 0.7191 ± 0.0101 in
`exp4_zero_history_subset.json` (separate runs of the same configuration).
The manuscript quotes the ablation table for the ablation claim and exp4's
own pair (0.7191 full test vs 0.1387 subset) for the zero-history contrast,
so each comparison stays internal to one run.

---

## 11b. Text tranche A — Reviewer 2's five points

**Provenance finding, recorded before anything else.** The repository `sn-article.tex` is
not the submitted version. The commit `4377913` ("Changes in paper and running Poisson")
already applied most of Reviewer 2's round-2 requests. Verified by grep against the exact
strings the assessment quoted as present in the submission:

| R2 point | State found in repo `.tex` | Action taken this tranche |
|---|---|---|
| 1 — rename the outcome | Partial. Abstract and conclusion already renamed; title, keywords, intro and §2 opening still said "enforced disappearances" | Completed the renaming; added a paragraph reserving each term |
| 2 — causal language, 3 passages + §5.2 title | **Already done.** All three passages already predictive-association wording; §5.2 already carries the reviewer's drop-in title | None needed |
| 3 — narrow the network null | Partial. 6 of 9 sites already carried "in this dataset and design"; 3 did not (Fig. 1 caption, Poisson passage, conclusion) | Applied at the 3 remaining sites |
| 4 — soften generalisability | Partial. The "does not imply identical precursors" note was present; the surrounding claim was still three-country and false on five | Rewrote the claim (see reversal below) |
| 5 — ACLED sourcing and coding | Largely done. Sourcing, actor taxonomy, broader-than-legal qualification and no-filtering statement all present | Added the two coding boundary rules |

### Point 1 — outcome renaming

- **Title** is now "Predicting Abduction and Forced Disappearance Events: A Machine
  Learning Analysis of Temporal and Network Features". Co-author sign-off wanted.
- **Keywords** gained "Abduction and forced disappearance" ahead of the legal term.
- **Intro ¶1** gained a paragraph stating that the measured outcome is narrower than the
  legal definition, naming ACLED's category, and reserving "abduction/forced-disappearance
  events" for the measurement and "enforced disappearance" for the legal concept.
- **§2 opening** rewritten; also dropped its self-summarising final sentence.

### Point 5 — the two coding rules

Taken from the ACLED codebook, not from memory. Both work against our outcome:

1. **Event-type hierarchy.** The most severe reported outcome sets the sub-event type, so
   an abduction whose victim is later reported killed is coded `Attack` and never enters
   our outcome. Cases ending in confirmed death are systematically absent.
2. **Arrests boundary.** Deprivation of liberty by a non-state group operating a judicial
   or penal system is coded `Arrests` — which is one of our four predictor features. There
   is a coding boundary between the target and one of its own inputs.

Both are stated as residual risk, per the reviewer's instruction to qualify rather than
argue away.

### Two claim reversals found while updating numbers in these passages

**(a) Poisson appendix — "outperforms on every metric" is now false.**
On three countries Events Only won all five metrics. On five countries neither dominates:

| Metric | Full Model | Events Only | Winner |
|---|---|---|---|
| test MAE | 1.7752 | 1.6860 | Events Only |
| test RMSE | 7.4121 | 6.4769 | Events Only |
| test deviance | 2.8915 | 2.9018 | Full Model |
| McFadden $R^2$ | 0.3615 | 0.3599 | Full Model |
| Lin's CCC | 0.4636 | 0.4695 | Events Only |

The two Full-Model wins are third-decimal-place. Rewritten as a no-difference result,
which is what the classification null says too. The sentence claiming network features
"measurably degrade out-of-sample count fit" is **removed** — it was true on three
countries and is not true on five. Also: McFadden $R^2$ is now ~0.36, not 0.12–0.19, so
the passage no longer describes count fit as poor.

**(b) Country-level performance — "consistent performance" is now false as stated.**
The submitted claim was a 2.2-percentage-point spread. On five countries the raw spread is
**38.9 points** (Afghanistan 0.571 → Syria 0.960) because test prevalence ranges 0.232 to
0.637. `exp9_country_breakdown.json`'s own description warns against exactly this
comparison: *"Read each country's AUPRC against its own test positive rate, not the pooled
prevalence."*

Lift over own prevalence is the comparable quantity and spans **10.7 points**:

| Country | Regions | Test seq. | Pos. rate | AUPRC | Lift |
|---|---|---|---|---|---|
| Afghanistan | 34 | 674 | 23.2% | 0.571 | +0.340 |
| Mexico | 32 | 757 | 36.2% | 0.595 | +0.233 |
| Myanmar | 18 | 419 | 63.7% | 0.894 | +0.256 |
| Nigeria | 37 | 861 | 57.1% | 0.804 | +0.233 |
| Syria | 14 | 325 | 63.4% | 0.960 | +0.326 |
| **Overall** | **135** | **3,036** | **46.0%** | **0.814** | **+0.355** |

Seed-to-seed SD within country is at most 0.0016, so the between-country differences are
real rather than noise. Table~`tab:country` gained a Lift column and a caption stating
that these are one pooled model scored per country, not per-country models.

### Conclusion opening rewritten

Now: AUPRC 0.814 against persistence 0.597 and prevalence-equivalent 0.459; recency ratio
1.7 (was stated as 1.5, actual 1.682 from `timestep_importance`); and the no-past-
disappearances result is reported with its zero-history bound (0.723 pooled, 0.139 on the
eight no-history regions against a 0.175 no-skill baseline) rather than as the paper's
strongest finding.

### Open: §4 still carries three-country numbers

Not in this tranche's scope, and it needs its own pass. 27 non-comment lines still hold
submitted three-country values — the whole ablation table and its surrounding text
(lines 462–526), the tier summary (505), sequence counts (354, 535), and the
country-selection rationale (273–275, 699). Values affected: 0.741, 0.745, 0.740, 0.664,
0.666, 0.483, 0.570, 2,035.

---

## 11c. Section 4 numbers pass (five-country run)

Not a reviewer point. Every submitted three-country number in §3--§6 replaced
with its five-country value from `results/`. Three claims reversed and one
error in the submitted tables was found.

### Error in the submitted Table 4: mixed sign conventions

The submitted Gap column used **two different conventions in the same table**.
Verified against the printed train/test values:

| Row | Test | Train | Printed gap | test-train | train-test | Convention used |
|---|---|---|---|---|---|---|
| Attention-LSTM | 0.741 | 0.705 | -0.036 | +0.036 | -0.036 | train - test |
| Standard LSTM | 0.740 | 0.701 | -0.039 | +0.039 | -0.039 | train - test |
| Tuned LR | 0.738 | 0.689 | **+0.049** | **+0.049** | -0.049 | **test - train** |
| XGBoost | 0.723 | 0.704 | **+0.018** | **+0.019** | -0.019 | **test - train** |
| LR untuned | 0.717 | 0.626 | -0.091 | +0.091 | -0.091 | train - test |

The caption said "Gap = Test - Train AUPRC", which matches only two of the five
rows. **All five models scored higher on test than on train**, in both the
three- and five-country runs. The discussion built a model-selection argument
on the apparent contrast ("LSTM models exhibit negative train-test gaps ...
while the tuned LR and XGBoost show positive gaps, suggesting weaker
generalization"). That contrast was an artefact of the two flipped rows. The
argument is withdrawn in §4.1 and §6; the caption now states the convention
and its direction explicitly. This also completes the retirement of the
generalization-gap defence decided in tranche B.

### Reversal 1 - the LSTM no longer leads Table 4

Five-country, 16-feature set, sorted by test AUPRC: tuned LR 0.811, standard
LSTM 0.809, **untuned** LR 0.809, Attention-LSTM 0.808, XGBoost 0.792. The
submitted table led with Attention-LSTM. Rows are still sorted by test AUPRC,
so a linear model now heads the paper's main model table. exp7 on the full
feature set: observed LSTM - LR = -0.001, bootstrap 95% CI [-0.006, +0.005],
permutation p = 0.805. exp10 on the events-only set: +0.003, CI
[-0.000, +0.006], p = 0.444. Neither feature set separates the families, so
the reordering does not change the conclusion - but it changes which row is
first, and co-authors should see it.

The architecture justification is now the attention weights (§4.4), not
accuracy. Consistent with the exp8 architecture null already in the abstract.

### Reversal 2 - "predicts new outbreaks" was false

Discussion §5.x claimed "The model predicts new outbreaks without prior
history of disappearance (AUPRC = 0.666)". The No Disappearances ablation is
trained and scored on a panel dominated by regions that *do* have history, so
that number never supported the claim. exp4 evaluates the same fitted models
on the eight regions with no training-period disappearance (Badghis, Laghman,
Nimruz, Paktika, Ayeyarwady, Nay Pyi Taw, Lattakia, Tartous; 160 test
sequences, 17.5% positive): **AUPRC 0.139 +- 0.010 against a no-skill
baseline of 0.175** - at or below chance. Removed, and the zero-history
evaluation promoted to its own subsection (`sec:zero_history`) since three
other passages now point at it.

Note the submitted version reported only 2 zero-history regions (47
sequences) and called the subset too small to evaluate. Five countries make it
evaluable, and the answer is negative.

### Reversal 3 - network categories, sign of the per-category effect

Submitted: "structure +0.0, clustering +0.0, weights -0.002, centrality
-0.005" relative to Events Only. Five-country: clustering -0.001, structure
-0.001, centrality -0.001, weights -0.002. All four now cost the same small
amount rather than splitting in sign. Events Only - Network Only widens from
8.1 to 12.1 points.

### Values replaced

| Quantity | Submitted (3c) | Five-country | Sites |
|---|---|---|---|
| Best AUPRC (Events Only) | 0.745 | 0.814 | 495, 596, 653, 671 |
| Full Model AUPRC | 0.741 | 0.809 | Tables 4, 5, 6 |
| No Disappearances | 0.666 | 0.723 | 445, 499, 653 |
| Network Only | 0.664 | 0.693 | Table 5 |
| Random baseline | 0.483 | 0.459 | 382, 445, 455, Table 5 |
| Persistence | 0.570 | 0.597 | Table 4 |
| Train / test positive rate | 37.3 / 48.3% | 31.1 / 46.0% | 354, 368, 382, 455 |
| Sequences (train / test) | 4,258 / 2,035 | 6,660 / 3,036 | 354, 392, 475, 541 |
| Total sequences | 6,293 | 9,696 | 354, 392 |
| Drop from removing disappearances | 7.5 pt / 10.1% | 8.6 pt / 10.7% | 445, 509 |
| Margin over persistence w/o lag | 9.6 pt | 12.6 pt | 509 |
| XGBoost network penalty | -0.010 | -0.013 | 533 |

Ablation ranks 1--7 now span 0.808--0.814 (range 0.006), so the submitted
"three performance tiers" description was replaced with one band plus two
isolated points: rank 8 at 0.723 and rank 9 at 0.693 are 3 points apart, not
0.002 as in the three-country run.

### Stale string inside a result file

`exp7_bootstrap_lr_vs_lstm.json` has `"identical test set (n=2035, 2023-01
onward)"` in its description field, but its metrics are five-country
(lstm_mean_auprc 0.8088 matches `ablations_5c.json` Full Model exactly). The
description string was not updated when the experiment was rerun. Manuscript
quotes n=3036 from `exp8`'s sample block. No code change needed; recorded so
nobody re-derives 2,035 from that file.

---

## 11c. Text tranche C — model comparison, related work, motivation (R3 points D, E, F, I)

### Two of the four points were already applied by the author

Same provenance finding as tranche A. Before editing, each point was checked against
the repo `.tex`:

| Point | Ask | State found | Action |
|---|---|---|---|
| D | Which model is primary; retire the generalization defense | Paired-test paragraph and table already rewritten in the §4.1 numbers pass; the ceiling argument and the LSTM's role were missing | Added to §5.3 |
| E | Rewrite the opening for motivation | Opening was abstract, no operational stake | Rewrote ¶1, added a forecasting-rationale paragraph |
| F | Count/regression trend and why binary | **Already present** — §2.1 carries the trend, the operational defense of the binary target, and a pointer to the Poisson appendix | none |
| I | Why LSTM over GRU/Transformer | **Already present** — §2.4 gives parameter efficiency on a small input space plus precedent in the closest prior work | none |

### D — the argument added to §5.3

The retired generalization defense left the choice of the Attention-LSTM unjustified,
since on five countries the tuned LR edges both recurrent models. Two paragraphs replace it:

1. **Data-limited ceiling.** A regularised linear model on flattened lags, a
   gradient-boosted ensemble, and an attention-LSTM land within 0.02 AUPRC on the same
   four features. The constraint is the information in four monthly event counts, not
   the function class.
2. **The LSTM as capacity test.** The network null needs a model that *can* represent
   interactions among centrality, clustering, and edge weights. A linear model cannot,
   so a null from it alone is answerable in one line; from an attention model it is not.
   The linear result closes the argument from the other side — a signal made of deep
   nonlinear temporal interaction could not have been matched by a linear model on lags,
   and it was matched.

### Third claim reversal: the 10.1 / 89.9 decomposition was not a valid share

§5.4 read the lagged-feature ablation delta as a share of performance: past
disappearances "contribute 10.1% of performance," the other three features "the
remaining 89.9%." AUPRC has a floor at the positive class rate, not at zero, so
dividing by the raw score credits the remaining features with the baseline as though
they had earned it.

Recomputed against the baseline: the full model gains 0.349 over the random baseline
(0.809 vs 0.459); withholding the lagged feature gives up 0.086 of that gain, **24.7%**,
leaving **75.3%** on arrests, violence, and fatalities. The passage now reports shares
of the gain and says why.

### SHAP: values, ordering, and a disagreement worth reporting

Every value in §4.x's attribution passage and Table `tab:shap_importance` was stale, and
the ordering changed. Source `results/exp5_shap/shap_summary.json`.

| Feature | Submitted | Five-country |
|---|---|---|
| Prior disappearances | 0.220 | **0.313** |
| Fatalities | 0.046 | **0.016** |
| Violence | 0.029 | **0.017** |
| Arrests | 0.018 | **0.021** |

The three non-lag features now span 0.016–0.021. Their order flipped (fatalities from
2nd to 4th), and the range is too narrow to support any ranking among them — the text
now says so rather than listing them "in descending order."

The dominance ratio rose from ~5× to 15×, which sets up a real tension with the ablation:
attribution puts the lagged feature 15 times ahead of anything else, yet withholding it
costs only a quarter of the gain over baseline. Both are correct. Attribution magnitude
and non-substitutable information are different quantities, and the gap means the other
three features carry much of the same signal in usable form. This is now stated in both
§4.x and §5.5 instead of the two numbers being presented as mutually corroborating.

Recency gradient corrected: 0.439 at $t$ against 0.270 at $t-5$ for disappearances
(1.6×, was 1.5× on 0.304/0.200). Summed across features the gradient is **not monotone**
— it dips at $t-3$ (0.072) before rising through the last two months — so the claim that
the gradient is "direct evidence the model learns genuine temporal structure" was
softened to a recency tilt across the window.

### E — motivation

¶1 of the introduction dropped the abstract concealment/intimidation pair for the
measurement consequence: a disappearance is arranged to leave a question, which is both
what makes it work as intimidation and what makes it hard to count. A new paragraph gives
the forecasting rationale, which the submitted introduction never stated: documentation is
retrospective by construction, monitoring capacity has to be committed before the period
it covers, and a ranking of region-months is therefore an allocation instrument. No
numbers were moved into ¶1 — the dataset scale stays where it was.

### Prose sweep (tranche C portion)

`Additionally,` at §2.x removed. Full banned-vocabulary scan over non-comment lines now
returns nothing for the whole file.

---

## 11d. Text tranche D — scope answers (R3 points G, H)

### G, country half — answered with data, not argument

§3.1's selection paragraph still described three countries and still summed to 87 admin1
units. Rewritten around the five-country extract:

| Country | Disappearance events | admin1 units | Region |
|---|---|---|---|
| Syria | 4,841 | 14 | Middle East |
| Nigeria | 2,899 | 37 | West Africa |
| Myanmar | 1,744 | 18 | Southeast Asia |
| Mexico | 1,472 | 32 | North America |
| Afghanistan | 527 | 34 | Central Asia |
| **Total** | **11,483** | **135** | five regions |

Source `results/descriptives_5c.json`. The paragraph now leads with recorded outcome
volume as the selection criterion rather than with country narratives, which is the form
the reviewer's "why these" actually asks for. Mexico and Myanmar keep their existing
citations; no new country-narrative citations were invented for Syria or Afghanistan,
whose justification rests on extract-measurable properties and conflict type.

### G, country half — the sampling frame is now stated

New §3.1 paragraph, sourced from the named constants in `src/data_prep.py`:

- **Maritime areas** (`EXCLUDE_MARITIME`, four ocean/sea pseudo-entities, 149 events).
  Not administrative units — no population, no admin1 geography — so the region-month
  unit is undefined for them.
- **Taiwan** (`EXCLUDE_OUT_OF_FRAME`). Entered as an artifact of the ACLED pull: 6,190
  events, 21 admin1 units, all 84 months, of which 6,093 are protests and **zero** are
  abduction/forced disappearance.

The Taiwan figures are stated in the paper with their direction of effect, because the
direction favours us: retaining 21 all-negative regions (1,006 sequences of easy
negatives) leaves AUPRC at 0.8081 against 0.8088 but lifts AUROC 3.1 points (0.8174 →
0.8483) and widens the persistence margin 1.4 points. **Excluding Taiwan lowers every
number we report.** Saying so pre-empts the obvious suspicion about a discretionary
exclusion.

### G, date half — defended on coverage comparability

The export still spans 2018-01 to 2024-12, so this half needed an argument rather than a
rerun. The argument is not "that is what we had": ACLED back-coded its Latin America and
Caribbean file to the start of 2018, so **January 2018 is the first month in which all
five countries are covered on a comparable sourcing basis**. An earlier start would give
a panel whose first years mix ACLED's monitoring expansion in some countries with real
variation in others, and the lagged counts carrying most of the signal would absorb that
as trend. New bib entry `acledlac2020` (ACLED press release, Dec 2020) — verified against
ACLED's own coverage documentation, not asserted.

### H — the Poisson appendix table was stale; the prose was not

Decision S3 (include the count model as an appendix with a full metric suite) was already
implemented in §3.9 and §4.8. Checking it against `results/exp6_poisson_regression.json`
found the **table** carrying three-country values while the **prose** carried the correct
five-country ones — the reverse of the usual failure, and it had the two configurations
ranked the wrong way round on two of five metrics.

| Metric | Full, was | Full, now | Events, was | Events, now |
|---|---|---|---|---|
| MAE | 1.259 | **1.775** | 1.242 | **1.686** |
| RMSE | 2.450 | **7.412** | 2.422 | **6.477** |
| Deviance | 2.741 | **2.892** | 2.454 | **2.902** |
| McFadden $R^2$ | 0.123 | **0.362** | 0.186 | **0.360** |
| CCC | 0.143 | **0.464** | 0.240 | **0.470** |

The old table bolded Events Only as the winner on all five metrics. On the real numbers
neither configuration dominates — Events Only wins MAE, RMSE and CCC; the Full Model wins
deviance and $R^2$ in the third decimal — which is what the prose already said. Bolding
removed. Test count distribution also corrected: range 0–47 (was 0–33), mean 1.69 (1.30),
s.d. 4.20 (2.55), 54.1% zeros (51.7%), n = 3,036.

### Syria concentration, addressed before a reviewer raises it

Syria is 42.2% of disappearance events but only 10.4% of regions and 10.7% of test
sequences, because training and evaluation run on region-month sequences rather than on
events. It is also the highest-lift country, so a sample without it would score lower.
The Limitations section now states all of this, states that the other four countries each
retain at least +0.233 lift, and **says plainly that a leave-one-country-out retrain
would settle the question more directly and that we have not run one.** The plan made
that check conditional on approval; it remains unrun and open.

### Two more unsupported claims removed

Both flagged by a banned-vocabulary scan on "robust", and both were empirical claims we
never tested:

- §4.6: "more robust to data quality variation" → now states what the configuration does
  buy (no network-construction step, four counts readable off an event feed) and says
  explicitly that sensitivity to data quality was not tested.
- §6: "simpler event-based models are sufficient and more robust under distributional
  shift" → replaced with the tested claim, that the null appears in all three model
  families.

---

## 11f. Response letter and changes list (2026-09-12)

Two new deliverables, neither of which changes `sn-article.tex`:

- `paper/RESPONSE_TO_REVIEWERS_R3.md` — point-by-point response. Reviewer 2's five
  points, Reviewer 3's nine, then four sections that the reviewers did not ask for:
  where the two reports interact, claims withdrawn, requests not implemented, and a
  marked internal note for Constantinos and Nikandros to delete before submission.
- `paper/CHANGES_ROUND3.md` — every manuscript change ordered by section, each mapped
  to a reviewer point label or marked `self`.

**Disclosure decision.** Section 5 of the letter states the three claims withdrawn
during this revision: the train-to-test generalisation-gap argument, the
feature-contribution shares computed against a zero floor, and the monotone recency
claim. The alternative was to let the changed tables speak for themselves. We judged
that worse — the numbers moved in Tables 3, 4, 6 and 9, and a reviewer comparing
versions would see it and ask.

**Constraint honoured.** No sentence in either document characterises what the previous
round's numbers represented, and neither claims the pipeline was rerun unchanged on
additional countries. Section 1 of the letter states the sample change and that all
numbers were recomputed on it; it does not explain each movement.
`docs/INTERNAL_NOTE_target_index.md` is not referenced, per decision S2.

**Numbers.** Every figure quoted in the letter was re-read from `results/` in the same
session, not copied from the manuscript: exp7 ensemble difference −0.0009747 with
bootstrap CI [−0.006492, +0.004550] and permutation *p* = 0.8047; exp10 difference
+0.0027988 with CI [−0.000246, +0.005863] and *p* = 0.4444; ablation events-only 0.8140
against full 0.8088; tuned LR 0.8109 full and 0.8115 events-only; XGBoost replication
delta +0.0134. Table and figure numbers in both documents were taken from the `.tex`
ordering, not assumed — the model-comparison table is Table 3, not Table 4 as the
working notes in `numbers_of_record.csv` still label it.

**Open item.** The Taiwan exclusion figures disagree between `docs/DECISIONS_ROUND3.md`
(2.9 AUROC points) and `src/data_prep.py` (3.1 points). The paper and the letter use the
`data_prep.py` figures. Annotated in the decisions log; to be resolved in the
verification pass.

---

## 12. Bottom line

95 of the manuscript's 763 lines carry a number. No cell is pending any
more — every table has a regenerated value traceable to a file in
`results/`. The great majority are straight swaps.

The paper's central claim survives and strengthens: network features add
nothing beyond four event counts, replicated across three model families
(§4), on a sample 78% larger with two additional countries.

Eight places need the claim rewritten rather than the digits replaced:

1. §3b, L471 — the model ranking inverts; untuned LR beats both LSTMs
2. §3c, L136 and L471 — the generalization argument does not hold for any
   model, and the gap comparison was never valid across unequal prevalence
3. §3d, L473 — LSTM vs. tuned LR, the sign reverses
4. §6a, L598 and L618 — country consistency is false on raw AUPRC, true on
   lift; Table 8 needs a prevalence column
5. §7, L639 — Poisson "every metric" becomes three of five
6. §5, L546–549 — SHAP minor-feature order is not stable
7. §2a — network-category comparisons, the spread is inside the noise
8. §8, L136 / L511 / L675 / L707 — onset performance is below baseline on
   new-onset regions, stated unqualified in both abstract and conclusion

Plus two sections needing argument rather than numbers: country selection
including the unaddressed date range, and the study-country narrative now
that Syria carries the largest share of the outcome.

The primary-model decision (§11) has been settled by measurement rather
than preference. exp10 tested the events-only margin that option A
depended on: +0.0028 with a bootstrap CI of [−0.0002, +0.0059] and a
permutation p of 0.4444. Option A is out. Items 1, 2 and 3 above should
be written under option B — parsimony as the finding, with the linear
result reported as a second null alongside the network null.
