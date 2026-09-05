# Round 2 Reviewer Assessment — EPJ Data Science
## Major Revision Comments, 31 Aug 2026

Source documents: `EPJ Data Science - Major Revision Comments - 31_8_2026.pdf` (this round),
`Summary of changes first review round - June 2026...pdf` (context only, already implemented).

Every claim below was checked against the actual submitted manuscript text
(`pred_eds_ml_forecasting_REVISIONS_SUB_READY_2_June_V1 (6).pdf`) rather than assumed from
the review letter alone. Where a manuscript passage is quoted, it was located by direct
text search on the extracted PDF, not recalled.

---

## Reviewer 2 (returning reviewer — five points, all wording/framing)

Reviewer 2's own closing line: *"I believe the paper will be completely ready for
publication after these minor revisions."* All five points are textual — none requires
new modeling or a pipeline rerun.

| # | Comment | Confirmed in manuscript? | Recommendation |
|---|---|---|---|
| 1 | Dependent variable: paper defines ED narrowly (legal sense) but operationalises via ACLED's combined "Abduction/forced disappearance" sub-event type with no further filtering | **Yes.** Intro defines ED as "the arrest, detention, or abduction of persons by state agents or organized groups followed by a refusal to acknowledge their fate"; Methods codes the outcome directly from `sub_event_type == Abduction/forced disappearance`, no additional filter | Adopt reviewer's second option: rename the outcome consistently as "abduction/forced-disappearance events" in title, abstract, analysis, conclusion. Re-filtering ACLED events to the narrower legal definition would require new manual coding — out of scope for this round. |
| 2 | Remaining causal language in three passages + Section 5.2 title | **Yes, all four confirmed verbatim:** "If actor network configuration were a primary driver of disappearances, removing network features would materially reduce predictive performance"; "Violence and arrests create the operational environment for disappearances"; "the near-zero contribution of fatalities specifies this mechanism"; section title "What Drives Disappearance Prediction" | Rewrite all four as predictive-association statements. Reviewer's suggested replacement title ("Features Contributing to Predictive Performance") is a direct drop-in. |
| 3 | "No predictive value" claimed too broadly (5 occurrences) | **Yes**, confirmed in abstract, Fig. 1 caption, Section 5.1 (×2), and the conclusion. Reviewer supplies the exact replacement wording. | Adopt reviewer's suggested sentence verbatim: "provide no incremental predictive value beyond the event-based features in this dataset and design." Apply consistently across all 5 instances. |
| 4 | Generalisability overclaimed — "ensures"/"demonstrates" language from a 3-country purposive sample | **Yes.** Found: "multi-contextual validation demonstrates that our data-driven indicators generalize across diverse conflict settings rather than being artifacts of a specific geographic or political environment." | Soften to: results are robust *within* Nigeria, Mexico, Myanmar; external validity beyond these three is untested. Also flag that similar country-level AUPRC does not imply the model uses the same precursors per reviewer's note. |
| 5 | ACLED coding/verification process for the disappearance category should be briefly explained; "less problematic" claim needs qualifying | **Yes**, found: "this distinction is less problematic for early warning applications where monitoring organizations operate within the same information environment." No explanation of ACLED's verification/coding process is present. | Add 2–3 sentences on ACLED's sourcing and coding procedure for this sub-event type; qualify "less problematic" (state what residual risk remains, don't claim it away). |

**Triage: all 5 are text edits.** No rerun of any experiment needed.

---

## Reviewer 3 (first-time reviewer for this journal — self-flagged uncertainty about fit)

Reviewer 3 explicitly says: *"perhaps some of my comments or expectations are off... I urge
the editor to ignore unfitting comments."* Several points are well-grounded and confirmed
against our own text; others read as a different subfield's conventions (VIEWS/PRIO-GRID
political-science forecasting norms) rather than errors, and a few amount to requests for
new experiments substantially beyond revision scope. Triaged below.

### Points confirmed as real, actionable problems

**A. VIEWS citation does not support the admin1 justification — internal inconsistency, not just a citation quibble.**
The manuscript's own Related Work section states: *"The ViEWS (Violence Early Warning
System) consortium... forecast[s] state-based conflict and one-sided violence at
country-month and PRIO-GRID-month levels"* — correctly describing ViEWS as **not**
admin1-based. Yet the Methods section separately justifies admin1 aggregation by
*"following the spatial granularity employed by leading conflict forecasting systems [10]"*
— and reference [10] is that same ViEWS paper (Hegre et al. 2019). The manuscript
contradicts itself between Section 2 and Section 3. Reviewer 3 is factually correct.
**Fix:** either cite a source that actually uses admin1 (reviewer points to Oswald 2026,
JCR, using admin0/admin1 with Google Trends/Wikipedia data) or drop the appeal-to-ViEWS
framing and justify admin1 directly on its own merits (data sufficiency vs. resolution
tradeoff, which the sentence already states independently of the citation).

**B. "Onset" framing is misleading given the actual class balance — confirmed.**
Section 2.1 frames the paper's motivation around ViEWS's documented difficulty
*"forecast[ing] conflict onset in previously peaceful locations"*, and other passages use
onset-adjacent language. But the actual target is 37.3% (train) / 48.3% (test) positive
region-months — not rare, and the paper's own robustness check found only 2 of 87 admin1
regions had zero disappearances during the training period. The task is overwhelmingly
**persistence/recurrence prediction within already-affected regions**, not onset
prediction in new locations. This is a framing problem that goes past word choice — it
shapes which literature (King-Zeng rare-events correction, civil-war onset comparisons)
the paper leans on to justify its methodology.
**Fix:** replace "onset" with "incidence"/"recurrence" throughout; be explicit early that
this is a persistence-prediction problem in known-affected regions, not new-region
detection — and adjust the 2.3 framing accordingly (see next point).

**C. "Rare events" framing conflates two different prevalence figures — confirmed.**
Section 2.3 justifies the entire methodological apparatus (AUPRC-over-AUROC, King-Zeng
bias correction citation, comparison to 1–2% civil-war-onset prevalence) using the
**event-level** figure: *"only 2.9% of conflict events in our data involve
disappearances."* But the actual classification target is the **region-month** aggregate,
at 37–48% positive — not rare by any standard. The manuscript has one clarifying sentence
("References to rarity in this paper refer to the event-level figure unless otherwise
stated") but this doesn't resolve the deeper issue: Section 2.3's theoretical
justification (why AUPRC, why not standard logistic regression, why this is methodologically
like civil-war-onset work) is built on a prevalence figure that does not describe the
actual prediction task.
**Fix:** this needs more than a footnote. Either drop the rare-event framing for the
main classification task and justify AUPRC on its own terms (imbalance is still real at
Nigeria/regional level, just not "2.9%-rare"), or restructure 2.3 to be explicit that
event-level rarity motivated the initial framing while the operationalized task is more
balanced — and explain why AUPRC is still the right metric regardless.

**D. Logistic regression is nearly indistinguishable from LSTM — deserves more discussion, not new modeling.**
Confirmed in Table 3: tuned LR = 0.738 AUPRC vs. Attention-LSTM = 0.741 ± 0.003 — a
0.003-point gap, inside the LSTM's own seed variance. The manuscript's current answer is
the train→test generalization-gap argument (LSTM: −0.036/−0.039 gap; tuned LR: +0.049;
under the 37.3%→48.3% distribution shift), which is a real and already-computed finding,
not a new one. Reviewer wants this made central rather than a passing note.
**Fix:** this is a writing/emphasis fix using numbers already in hand. Consider adding a
paired bootstrap or permutation test over the 3 LSTM seeds vs. the single LR run to give
the "is this gap real" question a formal answer — small additional computation, not a
new experiment.

### Points that are partially reasonable / lower priority

**E. "Paper lacks motivation" (why disappearances, why prediction at all).**
Fair critique of framing/emphasis — the intro doesn't open with a strong "why this
matters" hook before getting into methodology. This is a rewrite of the opening
paragraphs, not new work, but is a judgment call about tone the co-authors should weigh
in on.

**F. Related-work review "thin," missing the shift toward numerical/regression conflict prediction.**
Partially fair. Section 2 is actually substantial — four subsections, ~3 pages, covers
conflict prediction, rare-event methodology, temporal modeling (LSTM literature: Beck/King/Zeng,
Malone, Radford, Von der Maase, Brandt et al.), and network analysis. But reviewer's
specific point stands: reference [12] (the ViEWS 2023/24 challenge, already cited) is
literally a **count-based fatality-prediction** competition, yet the manuscript only
cites it for its onset-prediction-failure finding, never discussing the broader shift
toward regression/count targets it exemplifies. **Fix:** add 2-3 sentences acknowledging
this trend and stating explicitly why this paper stays with binary framing (early-warning
trigger use case vs. magnitude estimation).

### Points that amount to requests for new experiments — flag for editor/co-author discussion

**G. Why only three countries; why not the full date range (Nigeria since 1997, data to Dec 2025/last week)?**
Country selection is *already* justified in the manuscript on four stated grounds
(historical significance per country, conflict-type diversity, subnational variation,
consistent ACLED coverage) — this partially answers "why these three," though not "why
not more." Extending the date range or adding countries means re-collecting ACLED data,
rerunning the full pipeline (`data_prep.py` → `train.py` → all ablations), and
re-verifying every reported number — a substantial undertaking, not a text fix. Reviewer
guesses this wouldn't be a large computational lift; based on the pipeline's actual
runtime (full 27-run ablation sweep completes in minutes on CPU), that part is likely
true, but re-collecting and re-validating a new, larger raw dataset is the real cost, not
the training compute. **Recommend:** respond that data collection was fixed at the point
the underlying dataset (HuRiViRe) was assembled, and treat any extension as future work
rather than a same-round rerun, unless the co-authors want to commit to it.

**H. Predict counts/regression instead of (or in addition to) binary classification, with a proper regression metric suite.**
This is the highest-effort ask in either review — a genuinely new modeling target,
new evaluation code, and new results tables, not a reframing of existing numbers.
Reviewer's own framing ("RMSE is no excuse") anticipates the paper's existing rare-event
justification for avoiding regression (Section 3.2), which point C above shows is itself
built on a shaky premise (event-level vs. region-month prevalence) — so the regression
objection and the rare-event-framing problem are linked. **Recommend:** decide with
co-authors whether to (a) add a small supplementary regression baseline (e.g., Poisson
or linear regression on monthly counts, reported with 2–3 metrics from the cited
JOSS framework) as an appendix result, or (b) explicitly scope the paper as
early-warning/binary-trigger by design and defend that choice on operational grounds
(a threshold-crossing alert, not a magnitude forecast) without adding a new pipeline.

**I. Why LSTM over alternatives (GRU, Transformer, etc.)?**
Section 2.4 reviews LSTM conflict-prediction literature at length but doesn't explicitly
justify LSTM vs. other sequence architectures. Table 3 already benchmarks LSTM against
non-temporal baselines (LR, XGBoost, persistence) — that comparison exists. A specific
LSTM-vs-GRU-vs-Transformer ablation would be new work with unclear payoff, given point D
shows the *bigger* open question is temporal-vs-non-temporal, not which temporal
architecture. **Recommend:** add one or two sentences citing why LSTM specifically (parameter
efficiency, established precedent in the cited conflict-forecasting literature) rather
than running new architecture comparisons.

---

## Summary triage

| Effort tier | Items |
|---|---|
| **Text-only fixes, no rerun** | Reviewer 2, all 5 points; Reviewer 3 A, B (partially — reframing), C (partially — reframing), E, F, I |
| **Small additional analysis (numbers already exist or are cheap to compute)** | Reviewer 3 D (bootstrap/permutation test on existing seeds) |
| **Requires co-author/editor decision before proceeding** | Reviewer 3 G (data extension/more countries), H (regression task) |

## Open questions for Constantinos and Nikandros

1. Reviewer 2's fixes and most of Reviewer 3's framing fixes (A, B, C, E, F, I) look
   straightforward to implement this round — propose drafting the text changes directly.
2. Reviewer 3's data-scope requests (G) look like future-work material given the
   HuRiViRe collection was already fixed at a point in time — agree on wording for the
   response letter here.
3. Reviewer 3's regression-task request (H) is the one item that could change the
   paper's scope. Decide: appendix-only supplementary regression baseline, or a written
   defense of the binary/early-warning framing with no new pipeline run.
