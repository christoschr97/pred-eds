# Response to Reviewers — Round 2

Manuscript: "Predicting Enforced Disappearances: A Machine Learning Analysis of
Temporal and Network Features," EPJ Data Science.
Reviewer letter: *EPJ Data Science – Major Revision Comments – 31 Aug 2026*.

We thank both reviewers for a careful second reading. Reviewer 2's five points are
all textual/framing issues and are resolved by consistent renaming and reframing,
with no change to any reported number. Reviewer 3 raised nine points; four
identify a genuine internal inconsistency or an unaddressed statistical question
and are resolved below with either a rewrite or a new supplementary analysis; two
(motivation, related-work coverage) are addressed by rewriting; two (country/date
scope, LSTM-vs-alternatives) we address by explanation and citation rather than
new experiments, for reasons given below. Section references below are to the
revised manuscript; `sec:` labels can be cross-checked directly against
`sn-article.tex`.

---

## Reviewer 2

### Comment 1 — Clarify the dependent variable

> "The manuscript defines enforced disappearance in the narrow legal sense but
> operationalises the outcome using ACLED's combined category 'Abduction/forced
> disappearance.' [...] It makes sense to either explain whether ACLED events
> were further filtered [...] or consistently describe the outcome as
> abduction/forced-disappearance events throughout the title, abstract,
> analysis, and conclusion."

**Response.** No additional filtering to the narrow legal definition was applied
to the ACLED events; we have not represented otherwise. We adopt the second
option and rename the outcome as "abduction/forced-disappearance events" (ACLED's
own sub-event-type label) at every point where we describe the measured
quantity: abstract, Introduction (research question 1, contribution statements),
Methods (outcome-variable definition), Results, Discussion, and Conclusions — 11
sites in total. We have also added two sentences to the data-collection
description (Methods) stating explicitly that ACLED codes actor type per event,
that this sub-event-type category is broader than the ICPPED legal definition,
and that no perpetrator-type filter is applied — so the modeled outcome is
transparently the operational ACLED category, not a verified legal
determination.

We have kept the paper's **title** unchanged ("Predicting Enforced
Disappearances..."). We read the reviewer's suggested scope as extending
through "the title," and considered changing it to match exactly. We decided
against it: the title functions as the paper's motivating, human-rights framing,
and the abstract's first sentence now states the operational outcome precisely
enough that the distinction is clarified at the reader's first point of contact
with the paper's substance. We flag this explicitly here in case the reviewer
or editor considers the title itself still in scope for point 1 — we are willing
to revisit this specific choice if asked.

### Comment 2 — Remove remaining causal language

> "Examples include: 'If actor network configuration were a primary driver...'
> and: 'Violence and arrests create the operational environment for
> disappearances.' [...] The title of Section 5.2, 'What Drives Disappearance
> Prediction,' could also be changed to something like 'Features Contributing to
> Predictive Performance.'"

**Response.** All four passages identified are rewritten as predictive-association
statements, and the reviewer's suggested section title is adopted verbatim for
what is now Section 5.2, "Features Contributing to Predictive Performance."

### Comment 3 — Limit the network finding to the tested measures

> "The tested graph-theoretic metrics derived from monthly actor co-occurrence
> networks provide no incremental predictive value beyond the event-based
> features in this dataset and design."

**Response.** Adopted verbatim at all five sites previously reading "no
predictive value": the abstract, the Figure 1 caption, Section 5.1 (two
instances), and the Conclusions. We have also applied the same narrower wording
in two further locations that make an equivalent overclaim (Contribution 1 in
the Introduction, and the Discussion section "Why Network Features Fail") since
they restate the same finding.

### Comment 4 — Moderate claims of generalisability

> "Three purposively selected high-prevalence cases cannot establish this. The
> results show robustness within Nigeria, Mexico, and Myanmar. Broader external
> validity remains to be tested. Similar country-level AUPRC scores also do not
> establish that the model captures the same substantive precursors in each
> context."

**Response.** The generalizability sentence in Methods and the country-level
performance discussion in Results are both rewritten to state robustness
*within* the three selected countries rather than generalizability *beyond*
them, and the Results discussion now explicitly notes that similar AUPRC across
countries does not by itself establish that the model relies on the same
precursors in each setting — both points adopted directly from the reviewer's
wording.

### Comment 5 — Clarify ACLED reporting and coding limitations

> "It still seems to make sense to briefly explain how ACLED verifies and codes
> relevant events [...] The statement that reporting bias is 'less problematic'
> for early-warning applications should also be qualified."

**Response.** Added to Methods: a short description of ACLED's sourcing and
verification procedure for the Abduction/forced-disappearance sub-event-type
category (multi-source corroboration, actor-type coding). The Limitations
paragraph's "less problematic" claim is qualified with a clause stating the
residual risk explicitly: reporting gaps may themselves correlate with
repression intensity, which the "less problematic" framing does not rule out.

---

## Reviewer 3

We are grateful for the reviewer's candor about their unfamiliarity with the
outlet's conventions; we did not discount any comment on that basis and checked
each one directly against our own manuscript text before responding.

### Admin1/ViEWS citation

> "You cite the VIEWS project to justify the admin1 level, but they do not
> operate on the admin1 level; only admin0 and PRIO-GRID. (see eg Oswald 2026 in
> JCR who used Google Trends and Wikipedia data on the Admin0 and Admin1
> levels)"

**Response.** Confirmed as an internal inconsistency: our own Related Work
section correctly describes ViEWS as forecasting at country-month and
PRIO-GRID-month resolution, yet the Methods section's admin1 justification cited
the same ViEWS reference. We have replaced that citation with Oswald (2026,
*Journal of Conflict Resolution*), which the reviewer identifies as operating at
the admin0/admin1 level, and reworded the sentence accordingly.

### Admin1 vs. admin2/grid resolution

> "Working on the admin1 level is okay, but you could likewise go on admin2 or
> grid [...] especially given the focus on just three countries."

**Response.** We agree finer resolution is possible in principle and already
note it as future work. We have not changed the resolution in this round: admin1
was chosen for data sufficiency (event counts per admin2 cell over our six-month
input window would be sparse enough to undermine the temporal-sequence
construction the model depends on), and revisiting this would require rebuilding
the aggregation pipeline and re-validating every reported result, not a text
edit. We consider this future work rather than an omission in the current
design and say so explicitly in the manuscript.

### "Onset" is misleading given the actual class balance

> "The phrasing to predict 'onset' is misleading, you are really predicting
> incidence which is much more prevalent (look at your class (im)balance which
> is by and large quite well distributed)."

**Response.** Confirmed and corrected. "Onset" language is replaced with
"incidence" throughout (abstract included), and the paper is now explicit that
this is a persistence/recurrence-prediction problem in already-affected regions
(37–48% positive region-months, only 2 of 87 admin1 regions with zero
disappearances in the training period), not new-region onset detection. The
rare-event methodological framing (King–Zeng bias correction, comparison to
1–2% civil-war-onset prevalence) is reworded so it no longer implies the
operationalized classification task shares that prevalence; the 2.9% figure is
now explicitly labeled event-level, distinct from the region-month prevalence
that the classifier actually targets and that motivates AUPRC as the primary
metric.

### Why LSTM over alternatives

> "Why was temporal LSTM chosen over other alternatives? [...] there was no
> proper explanation or discussion provided."

**Response.** We have added an explicit justification in Related Work: parameter
efficiency for the short six-month, 4–16-feature input sequences used here, and
precedent in the closest prior conflict-forecasting work using LSTMs (Malone
2022, Radford 2022). We have not run a new LSTM-vs-GRU-vs-Transformer
architecture ablation. The comparison our results actually motivate more
strongly is temporal-vs-non-temporal (see the next point): tuned logistic
regression is statistically indistinguishable from the Attention-LSTM on test
AUPRC, which is a more consequential architectural question for this task than
the choice among recurrent variants.

### Logistic regression is nearly indistinguishable from LSTM

> "The real question to me here is: do we really need LSTM if a simple logistic
> regression seems to be doing just fine? This deserves much more discussion."

**Response.** We agree, and this was the one point that warranted new
computation rather than reframing. We ran a paired bootstrap (10,000 resamples
of the shared n=2,035 test set) and a paired permutation test (10,000 label
swaps) on the AUPRC difference between the tuned logistic regression (0.738)
and the Attention-LSTM (0.741 ± 0.003 across 3 seeds). The 95% bootstrap CI on
the difference is [-0.008, +0.014] (crosses zero); the permutation test gives
two-sided p = 0.708. We now state plainly in Results that the raw AUPRC
advantage is not statistically distinguishable from zero, and in Discussion that
model architecture is not decided by aggregate AUPRC on this task alone — our
existing generalization-gap finding (LSTM: −0.036/−0.039 train→test gap vs.
tuned LR: +0.049, under the 37.3%→48.3% distributional shift between the two
periods) is the basis for preferring the LSTM for deployment, not a raw AUPRC
edge that this test shows is not real.

### Paper lacks motivation

> "Why do we need this paper, why forced disappearances, why predicting them?
> [...] it reads like there was the application first which later was decided
> to be turned into a publication."

**Response.** The Introduction is reordered to lead with the paper's theoretical
motivation (competing network-centric and state-capacity accounts of where
disappearances occur) and its framing sentence before the disappearance-specific
background material, so the "why this question matters" argument now precedes
the definitional and data background rather than following it.

### Why only three countries, why not the full date range

> "Why only three countries, and why these three? [...] why not use the full
> range of data available, eg Nigeria since 1997 and data are available until
> last week [...] increasing the sample would [...] not bring massive
> computational obstacles with it."

**Response.** We agree the training compute is not the constraint — the full
27-run ablation sweep completes in minutes on a single CPU. The constraint is
data collection and validation: the three-country, 2018–2024 scope reflects the
ACLED extract assembled and validated for this study, and extending country
coverage or date range means re-extracting and re-validating a materially
larger raw dataset, then rerunning and re-verifying every reported number
against it — not a bounded addition to the current pipeline. We treat this as
a natural direction
for future work building on the current, validated three-country dataset rather
than a same-round extension, and say so explicitly in Limitations.

### Related-work review is thin; misses the move to numerical/regression prediction

> "There was also a distinct move away from binary and categorical to numerical
> variable prediction for a few years now already, which is not reflected in
> this paper."

**Response.** Related Work now includes an explicit acknowledgment of this
trend, with a stated reason for keeping the binary framing here: an
early-warning trigger use case (crossing an alert threshold) rather than a
magnitude-forecasting one. We also note that reference [12] (the ViEWS
challenge, already cited) is itself a count-based fatality-prediction
competition, and now discuss it in that light rather than only for its
onset-prediction finding.

### Predict counts/regression, not just binary classification

> "Especially given the absence of a strong imbalance, I would like to see
> predictions of the number of disappearances as well. The RMSE metric is no
> excuse to not do a regression task here; there are plenty of other useful
> metrics out there [...] (see [...] https://joss.theoj.org/papers/10.21105/
> joss.04655)"

**Response.** We agree, and add a supplementary Poisson regression analysis
(new Methods subsection, new Results subsection with Table 9) predicting the
raw monthly disappearance count with an L2-penalized Poisson GLM, on the same
input construction and temporal split as the main classifier, evaluated with
the metrics from the reviewer's cited framework (Correndo et al. 2022): MAE,
RMSE, mean Poisson deviance, McFadden pseudo-R², and Lin's concordance
correlation coefficient. The result is directionally the same as our
classification ablations: the four-feature Events Only configuration
outperforms the 16-feature Full Model on every metric (MAE 1.242 vs. 1.259,
deviance 2.454 vs. 2.741, McFadden R² 0.186 vs. 0.123, CCC 0.240 vs. 0.143). We
present this as a supplementary robustness check rather than replacing the
binary framing, for the early-warning-trigger reason given under the previous
point.

---

## Summary of new material added this round

| New/changed element | Location | Motivated by |
|---|---|---|
| Poisson regression baseline (Methods + Results + Table 9) | `sec:poisson_methods`, `sec:poisson_results` | R3 (count/regression request) |
| Bootstrap + permutation test on LSTM-vs-LR AUPRC gap | Results (Model Architecture Comparison), Discussion | R3 (LR-vs-LSTM discussion) |
| Oswald (2026) citation replacing misattributed ViEWS citation | Methods (admin1 justification) | R3 (admin1/ViEWS) |
| LSTM justification sentence | Related Work | R3 (why LSTM) |
| Regression-trend acknowledgment | Related Work | R3 (related-work coverage) |
| Introduction reordered around motivation | Introduction | R3 (motivation) |
| Outcome renamed to abduction/forced-disappearance events (body text) | Abstract, Methods, Results, Discussion, Conclusions | R2 (dependent variable) |
| Four causal-language passages + Section 5.2 title rewritten | Results, Discussion | R2 (causal language) |
| "No predictive value" narrowed to "no incremental predictive value..." | Abstract, Fig. 1 caption, Results (×2), Conclusions | R2 (network finding) |
| Generalizability claim softened to within-country robustness | Methods, Results | R2 (generalisability) |
| ACLED coding/verification description added; "less problematic" qualified | Methods, Limitations | R2 (ACLED coding) |

No changes were made to `data_prep.py`, `models.py`, `train.py`,
`run_ablations.py`, or any previously reported classification result — every
number in the paper's original tables is unchanged. The full line-level diff is
in `CHANGES_ROUND2.md`.
