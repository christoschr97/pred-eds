# Response to Reviewers

- **Manuscript:** Predicting Abduction and Forced Disappearance Events: A Machine Learning
  Analysis of Temporal and Network Features
- **Journal:** EPJ Data Science
- **Revision round:** Major-revision comments dated 31 August 2026

---

We thank both reviewers for their careful and constructive comments. All five of Reviewer
2's points have been addressed, including the two replacement formulations proposed in the
report. Of Reviewer 3's nine points, seven are addressed in the main text, one is addressed
with a new supplementary count-regression analysis, and one is addressed through a clearer
scope justification rather than an expansion of the dataset. Section 6 identifies the
optional alternatives and additional analyses that were not undertaken and explains why.

During reanalysis, we also identified three claims that required correction. We have
withdrawn or qualified them and explain the revised interpretations transparently in
Section 5 of this response.

Section, table, and figure numbers below refer to the revised 39-page manuscript; page
numbers refer to pages in the compiled PDF. All quantitative values in this response were
checked against the project result files. The relevant file is identified where useful,
and `docs/numbers_of_record.csv` records all 183 reported quantities and their provenance.

---

## 1. Scope of this revision

The analysis now uses a five-country ACLED extract covering Afghanistan, Mexico, Myanmar,
Nigeria, and Syria: 135 admin1 regions from January 2018 through December 2024. The version
under review reported three countries. Every quantitative result in the manuscript was
recomputed on the five-country sample, so values differ from the previous version
throughout. Where a reviewer quotes a value from the earlier version, this response gives
the corresponding revised value and its source file.

Section 3.1 (pp. 10–13) now also states the sampling frame, which the previous version did
not. Two exclusions apply at load time, both as named constants in `src/data_prep.py`:

- Four maritime pseudo-entities (149 events). ACLED assigns events at sea to ocean and sea
  areas rather than to countries; these have no resident population and no admin1
  geography, so the region-month unit of analysis is undefined for them.
- Taiwan (6,190 events, 21 admin1 units, all 84 months, of which 6,093 are protests and
  none are abduction/forced disappearance). It entered the extract as an artifact of how
  the ACLED pull was specified.

We report the direction of the Taiwan exclusion explicitly. Retaining those 21 regions,
negative in every month, would add 1,006 sequences of easily
classified negatives: AUPRC is effectively unchanged (0.8081 against 0.8088) while AUROC
rises 3.1 points, from 0.8174 to 0.8483, and the margin over the persistence baseline
widens by 1.4 points. Excluding Taiwan therefore leaves AUPRC effectively unchanged while
producing the more conservative AUROC.

---

## 2. Reviewer 2

### 2.1 Point 1 — the dependent variable is ACLED's sub-event type, not the legal definition

**Comment.** The paper defines enforced disappearance in the narrow legal sense but
operationalises it through ACLED's combined "Abduction/forced disappearance" sub-event
type with no further filtering.

**Response.** Accepted. We have taken the second of the two options offered. Narrowing
the extract to the legal definition would require manual re-coding of every event against
the perpetrator and acknowledgment criteria, which is not available to us for this
revision. Renaming the outcome removes the mismatch while preserving the reproducible ACLED
coding rule used in the analysis.

**Changes.** The outcome is now named "abduction/forced-disappearance events" consistently
in the title, abstract, research questions, Section 3.2 (pp. 13–14), Section 4, and the
conclusions. Section 3.1 (pp. 10–13) states the coding boundary explicitly: the sub-event
type combines abduction and forced disappearance, does not condition on perpetrator
identity, and does not require a subsequent refusal to acknowledge the victim's fate. The
outcome is therefore broader than the 1992 Declaration and the 2006 Convention definition
in both respects.

### 2.2 Point 2 — residual causal language

**Comment.** Four passages remain causal, including the Section 5.2 heading.

**Response.** Accepted in full. All four are rewritten as statements about predictive
association, and the heading uses the reviewer's suggested replacement.

**Changes.** Section 5.2 (pp. 26–27) is now titled "Features Contributing to Predictive
Performance".
The "primary driver" formulation, the "create the operational environment" formulation and
the "specifies this mechanism" formulation are replaced with statements about what the
features contribute to prediction. Section 5.5 (p. 28) now states that the predictive design
cannot separate shared antecedents from context-specific ones.

### 2.3 Point 3 — "no predictive value" claimed too broadly

**Comment.** The claim appears in five places and overstates what a single dataset and
design can establish. Replacement wording supplied.

**Response.** Accepted. All reviewer-identified instances, together with related language
identified during our final audit, are now scoped to the tested co-occurrence-network
features and to the present dataset and design. The relevant locations are the abstract
(p. 1), Sections 4.2 (pp. 21–22), 5.1 (pp. 26–27), and 5.4 (p. 28), the Figure 3 caption
(p. 38), and the conclusion (p. 30).

**Evidence unchanged by the rewording.** Events-only 0.8140 ± 0.0003 against full-model
0.8088 ± 0.0006 AUPRC, a difference of 0.0052 in favour of dropping the network features
(`results/ablations_5c.json`); the four network-category configurations span 0.0012
(0.8118 to 0.8130), which is within seed variation.

### 2.4 Point 4 — generalisability overclaimed

**Comment.** "Demonstrates that our data-driven indicators generalize across diverse
conflict settings" is not supported by a purposive sample, and similar country-level AUPRC
does not imply the model uses the same precursors in each country.

**Response.** Accepted, and the second half of the point changed what we report. Section
4.6 now scores the pooled model within each country and reports lift over that country's
own test prevalence, because raw AUPRC is not comparable across countries whose no-skill
baselines range from 0.232 to 0.637.

**Evidence.** Raw AUPRC spans 38.9 points across the five countries (0.571 in Afghanistan
to 0.960 in Syria) and broadly tracks test prevalence. Measured as lift over own
prevalence the spread narrows to 10.7 points: +0.233 in Mexico and Nigeria, +0.340 in
Afghanistan (`results/exp9_country_breakdown.json`). Within-country seed standard deviation
is at most 0.0016, so the between-country differences are not seed noise.

**Changes.** The "demonstrates ... generalize" sentence is gone. Section 4.6 (pp. 24–25)
now states
that comparable lift does not establish that the model reads the same substantive
precursors in each setting, since an aggregate score cannot separate shared antecedents
from context-specific ones that happen to yield similar accuracy. The claim we make is
that positive lift is observed in every country in the sample; external validity beyond
these five countries is untested, as stated in Sections 5.6–5.7 (pp. 28–29).

### 2.5 Point 5 — ACLED coding and verification; the "less problematic" claim

**Comment.** The paper should briefly explain ACLED's coding and verification for this
category, and the claim that reporting bias is "less problematic for early warning
applications where monitoring organizations operate within the same information
environment" needs qualifying rather than asserting.

**Response.** Accepted. Section 3.1 (pp. 10–13) now describes the sourcing and coding
procedure and
names the two consequences that are measurement caveats rather than reassurances: ACLED
codes from reported incidents, so an event enters the record only if a monitor was present
and able to report it, and the sub-event type is assigned from the reported act rather than
from a subsequent determination of the perpetrator's acknowledgment.

**Changes.** The "less problematic" sentence is qualified: it now states the residual risk
that remains rather than claiming it away, and says that reporting environments are part of
the measurement process.

---

## 3. Reviewer 3

Several of these comments identified genuine internal inconsistencies in the submitted
text, and we are grateful for the opportunity to correct them.

### 3.1 Point A — the ViEWS citation does not support the admin1 justification

**Comment.** Section 2 correctly describes ViEWS as forecasting at country-month and
PRIO-GRID-month level, while Section 3 justifies admin1 aggregation by "following the
spatial granularity employed by leading conflict forecasting systems" citing that same
ViEWS paper. The manuscript contradicts itself.

**Response.** The reviewer is right and this was our error. We have dropped the appeal to
ViEWS and justified the spatial unit on the data directly.

**Evidence now given in Section 3.1 (pp. 12–13).** Our extract has 135 admin1 units against
3,044
admin2 units. Median events per unit-month is 14 at admin1 and 2 at admin2; the share of
unit-months with fewer than five events is 24.8% at admin1 and 76.1% at admin2; 10,506 of
11,340 possible admin1 region-months are observed, against 61,753 of 255,696 at admin2.
The network features in particular cannot be estimated from a handful of events per
unit-month. We also cite Oswald (2026), which does use admin0/admin1 units, as the
reviewer suggested.

### 3.2 Point B — the "onset" framing does not match the class balance

**Comment.** The paper motivates itself on ViEWS's difficulty forecasting onset in
previously peaceful locations, but the target is a high-prevalence region-month indicator.
The task is recurrence prediction in already-affected regions.

**Response.** Accepted. In addition to revising the framing, we added a targeted analysis
of the subset identified by the reviewer. Section 4.3 (p. 22) now evaluates regions with
no recorded abduction/forced-disappearance event during the training period.

**Evidence.** Eight of 135 regions record no abduction/forced-disappearance event anywhere
in the training period. They contribute 160 test sequences at a 17.5% positive rate. The
Attention-LSTM without past-disappearance features scores 0.139 ± 0.010 AUPRC on that
subset against a no-skill baseline of 0.175, while the same model scores 0.719 ± 0.010 on
the full test set (`results/exp4_zero_history_subset.json`). With 160 sequences and 28
positives the estimate is necessarily imprecise, so the claim we make is that predictive
skill in previously unaffected regions is not demonstrated — not that it is absent.

**Changes.** The target is now framed as incidence and recurrence rather than first-onset
prediction. The abstract and introduction (pp. 1–3) and Section 2.1 (pp. 4–5) state that
the demonstrated use case is recurrence forecasting in known-affected regions, with
first-onset prediction scoped out explicitly and quantified in Section 4.3 (p. 22). Train
and test prevalence (31.1% and 45.9%) are stated where the task is defined.

### 3.3 Point C — the rare-events framing rests on the wrong prevalence figure

**Comment.** Section 2.3 justifies the metric choice, the King–Zeng citation and the
comparison to 1–2% civil-war-onset prevalence using the event-level 2.9% figure, while the
classification target sits at 37–48% positive. A clarifying footnote does not resolve it.

**Response.** Accepted, and restructured rather than footnoted. Section 2.3 (pp. 6–7) is
now titled
"Event-Level Rarity, Class Balance, and Metric Choice" and separates the two quantities:
event-level rarity is why disappearances are hard to study and motivated the initial
framing, and the region-month target is what the models predict. AUPRC is justified on its
own terms for this target — it is the metric whose no-skill baseline is the positive rate,
which is what makes per-country comparison possible in Section 4.6 — and not by an appeal
to rare-event methodology.

### 3.4 Point D — logistic regression is nearly indistinguishable from the LSTM

**Comment.** The gap is 0.003 AUPRC, inside the LSTM's own seed variance. This deserves
central discussion. A paired test would give the question a formal answer.

**Response.** Accepted, and it changed what the paper claims. We ran the paired test the
reviewer suggested, on both feature sets, and we now report parsimony as a finding rather
than defending the sequence model's ranking.

**Evidence.** Paired bootstrap with B = 10,000 and a label-swap permutation null with
P = 10,000, on ensemble-averaged probabilities over the three seeds:

| Feature set | Tuned LR | LSTM ensemble | Difference (LSTM − LR) | 95% CI | Permutation *p* |
|---|---|---|---|---|---|
| Full, 16 features | 0.8109 | 0.8099 | −0.0010 | [−0.0065, +0.0045] | 0.8047 |
| Events only, 4 features | 0.8115 | 0.8143 | +0.0028 | [−0.0002, +0.0059] | 0.4444 |

Sources: `results/exp7_bootstrap_lr_vs_lstm.json`,
`results/exp10_bootstrap_events_only.json`, `results/exp1_tuned_lr.json`. On the full
feature set the top four models sit within 0.0028 of each other and the untuned linear
model is ahead of both sequence models (`results/exp8_architecture_comparison.json`).

**What the paper now says.** The twelve network features provide no incremental predictive
value beyond the four event counts in any of the three tested model families (LSTM
+0.0052, tuned LR +0.0006, XGBoost +0.0134 when the network features are removed). On the
same four event features, no tested sequence architecture improves on tuned logistic
regression. The nonlinear sequence model makes the network-feature result less dependent
on the limitations of a linear specification, while the linear model's parity shows that
the observed predictive signal does not require a recurrent architecture. The
Attention-LSTM is retained for the timestep-level attribution in Section 4.4 (pp. 23–24),
not as the best-scoring model.

Section 5.3 (pp. 27–28) interprets the convergence of three model families with different
inductive biases as consistent with a data-limited performance ceiling, rather than as
evidence that a more complex function class improves prediction.

### 3.5 Point E — the paper lacks motivation

**Comment.** The introduction does not establish why disappearances and why prediction
before it moves into methodology.

**Response.** Accepted. The opening paragraph (p. 2) is rewritten around the property that
makes
the violation what it is: the absence of a record. That is what makes it effective as
intimidation and what makes it hard to count.

**Changes.** A new second paragraph (pp. 2–3) states why this is a forecasting problem
rather than a
documentation problem. Documentation is retrospective by construction, entry into a
database depends on whether a monitor was present and able to work, and monitoring capacity
has to be committed before the period it covers. A ranking of region-months is what that
allocation decision consumes.

### 3.6 Point F — related work misses the shift toward count and regression targets

**Comment.** Reference [12] is a count-based fatality-prediction competition, yet the paper
cites it only for its onset-prediction finding and never discusses the trend it exemplifies.

**Response.** Accepted. Section 2.1 (pp. 4–5) now acknowledges the field's movement toward
magnitude
targets and states why we retain the binary framing: the operational use case is a
monitoring-allocation trigger, not a magnitude estimate. It points forward to the
supplementary count analysis added for point H in Sections 3.10 and 4.7 (pp. 19 and
25–26), so the binary choice is defended rather than assumed.

### 3.7 Point G — why only these countries, and why not the full date range

**Comment.** Country selection is partially justified; the date range is not. Nigeria has
ACLED coverage from 1997 and the data run to late 2025.

**Response.** Both halves are now answered in Section 3.1 (pp. 10–13): the country selection
is explained from the extract, and the date range is explained by the binding coverage
constraint.

**Countries.** The selection paragraph now leads with recorded outcome volume rather than
with country narratives: Syria 4,841 abduction/forced-disappearance events, Nigeria 2,899,
Myanmar 1,744, Mexico 1,472, Afghanistan 527, across 135 admin1 units in five world regions
(`results/descriptives_5c.json`). The sample is purposive and Section 3.1 says so.

**Date range.** The window starts in January 2018 for coverage comparability, not
availability. ACLED built its Latin America and Caribbean file later than its Africa, Asia
and Middle East files and back-coded the region to the beginning of 2018, so January 2018
is the first month in which all five countries are covered on a comparable sourcing basis.
An earlier start would give a panel whose first years mix the expansion of ACLED's
monitoring in some countries with actual variation in violence in others, and the lagged
event counts that carry most of the predictive signal would absorb that difference as
trend. December 2024 is the end of our extract.

**Scope of the revision.** Model retraining itself is inexpensive: the nine-configuration,
three-seed sweep completes in minutes on CPU. Extending the sample, however, requires
re-collecting and validating the raw extract and then re-verifying every reported result.
We therefore treat further temporal or geographic expansion as future work. Section 5.6
(pp. 28–29) reports the present concentration directly: Syria supplies 42.2% of recorded
abduction/forced-disappearance events while contributing 10.7% of test sequences and 10.4%
of regions. Table 8 shows that Syria has the highest raw AUPRC (0.960) and the
second-highest lift over its own prevalence (+0.326), while each of the other four countries
retains at least +0.233 lift. A leave-one-country-out retraining analysis would address the
effect of country composition more directly than within-country scoring of the pooled
model. We have not run that analysis, and Section 5.6 states this limitation explicitly.

### 3.8 Point H — predict counts, with a proper regression metric suite

**Comment.** Binary classification discards magnitude. "RMSE is no excuse."

**Response.** Accepted as a supplementary analysis, reported with the metric suite the
reviewer specified. Sections 3.10 (p. 19) and 4.7 (pp. 25–26) are new.

**Evidence.** L2-penalised Poisson GLM with a log link, tuned by grid search over the
penalty with five-fold `TimeSeriesSplit` on Poisson deviance, identical window construction
and temporal split to the classifier (`results/exp6_poisson_regression.json`):

| | Full, 16 feat. | Events only, 4 feat. |
|---|---|---|
| Test MAE | 1.775 | 1.686 |
| Test RMSE | 7.412 | 6.477 |
| Test Poisson deviance | 2.892 | 2.902 |
| Test McFadden *R²* | 0.362 | 0.360 |
| Test Lin's CCC | 0.464 | 0.470 |

Events-only wins MAE, RMSE and CCC; the full model wins deviance and *R²* in the third
decimal. Neither configuration dominates, which is the same conclusion the classification
ablation reaches, on a different target with a different model class. The count target
binarises exactly to the classification target: the sanity check on the threshold match
rate returns 1.0.

**What this does not let us claim.** The count model is moderately predictive here
(McFadden *R²* 0.36, Lin's CCC 0.47), so we cannot defend the binary target by arguing that
counts are unpredictable in this setting, and the paper does not. The defence is
operational: RMSE of 6.5 to 7.4 against MAE of 1.7 indicates a heavy right tail, so the
count model's error is dominated by a few large months, and what a monitoring allocation
decision consumes is a threshold-crossing ranking rather than a magnitude estimate.

### 3.9 Point I — why LSTM rather than GRU or a Transformer

**Comment.** Section 2.4 reviews the LSTM literature at length without justifying the
architecture against other sequence models.

**Response.** Accepted as a writing fix, as recommended. Section 2.4 (pp. 7–8)
now states the reasons: the input space is small (four to sixteen features over six
timesteps), which is where the parameter economy of a recurrent cell over an attention-only
model matters, and the closest prior conflict-forecasting work uses LSTM variants. Section
4.1 (pp. 20–21) already compares the standard and attention-augmented LSTM directly
(0.809 ± 0.001 against 0.808 ± 0.002), and point D above establishes that the open question is
temporal-versus-non-temporal rather than which recurrent cell. We therefore did not add a
GRU-versus-Transformer sweep, which would expand the architecture search without changing
the central comparison requested here.

---

## 4. Where the two reports interact

Reviewer 3's point C and point H are linked, and resolving them separately would have left
an inconsistency. The version under review declined a regression target partly by appeal to
rare-event framing, and point C shows that framing rested on the event-level figure rather
than on the prevalence of the actual target. We therefore did not keep that defence. The
count model is reported as a supplementary analysis in Sections 3.10 and 4.7, and the
binary framing is defended on operational grounds instead.

Reviewer 2's point 4 and Reviewer 3's point G also overlap on external validity. Section
4.6 reports per-country lift, and Section 5.6 states the sample's event-volume imbalance
and the check we did not run.

---

## 5. Additional claims corrected or withdrawn

These were identified while recomputing the manuscript's numbers, not raised by either
reviewer. We list them so the reviewers are not left to infer why particular passages
changed.

**The train-to-test generalisation-gap argument is withdrawn.** The version under review
defended the sequence model over logistic regression on a smaller train-to-test AUPRC gap.
On this sample all five models score higher on test than on train — Attention-LSTM by
0.0439, standard LSTM by 0.0505, XGBoost by 0.0505, untuned LR by 0.0714, tuned LR by
0.0747 — and the two logistic regressions show the largest margins, the opposite of the
direction the argument needs. Training prevalence is 31.1% against 45.9% in test, and
AUPRC's no-skill baseline is the positive rate, so the quantity tracks the prevalence shift
rather than generalisation. The argument is gone from Sections 4.1 and 5.3, and Table 3
now reports the gap with a single stated convention (train − test) for all five models
where two of five rows previously used the opposite sign.

**The feature-contribution shares are recomputed relative to the appropriate no-skill
baseline.** The previous
version reported the lagged outcome's contribution and the remaining features' contribution
as complementary percentage shares of performance, which takes zero as the floor of an
AUPRC. For a random classifier, the expected AUPRC equals the positive rate. Sections 4.4
and 5.2 now report the ablation as a share of the gain over that random baseline: the full
model's gain over baseline is 0.349 AUPRC, and withholding the lagged disappearance feature
costs 0.086 of it — approximately one quarter. Section 5.2 states the reason for the change
of denominator explicitly.

**The recency claim is weaker than stated.** Mean absolute SHAP per timestep across the
six-month window is 0.079, 0.074, 0.072, 0.081, 0.111, 0.133 from oldest to most recent.
Across all four features, the most recent month carries approximately 1.7 times the
attribution of the oldest; for the lagged disappearance feature alone, the ratio is 1.6.
The sequence dips mid-window rather than rising monotonically, so Section 4.4 now describes
a recency tilt across the window rather than direct evidence of learned temporal structure.
The attribution ordering of the three non-lagged features also changed on this sample. The
values are sufficiently close (0.021, 0.017, 0.016) that the text states their ordering
should not be read as a ranking.

Two further sentences were removed as unsupported rather than stale: a claim that the
simpler model is less sensitive to data quality variation, which we never tested, and a
claim that it is more stable under distributional shift, which the numbers above do not
support.

---

## 6. Alternatives and additional analyses not undertaken

For transparency, the following alternatives or additional analyses were not undertaken in
this revision:

1. **Re-coding the extract to the legal definition of enforced disappearance** (Reviewer 2,
   point 1). This would require manual coding of perpetrator identity and subsequent
   acknowledgment for every event. We adopted the reviewer's alternative and renamed the
   outcome instead.
2. **Further extending the date range or expanding beyond five countries** (Reviewer 3,
   point G). The extract was fixed when it was assembled; further extension would require
   re-collection and re-validation of the raw data, not merely additional training. This is
   treated as future work, with the current window's start defended on coverage
   comparability.
3. **A GRU-versus-Transformer architecture sweep** (Reviewer 3, point I). We instead added
   an explicit rationale for the LSTM and emphasized the more consequential comparison
   between temporal and non-temporal models, which Sections 4.1 and 5.3 address directly.
4. **A leave-one-country-out retraining analysis.** This was not requested, but it would
   provide a more direct test of sensitivity to country composition. It was not run and is
   stated as a limitation in Section 5.6.

We appreciate the reviewers' careful reading. Their comments led us to clarify the outcome,
correct the rare-event and first-onset framing, strengthen the model comparisons, add the
count-regression analysis, and narrow the conclusions to what the revised evidence
supports.
