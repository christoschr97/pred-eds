# Co-Author Briefing — Round 2 Revision (for Constantinos Djouvas and Nikandros Ioannidis)

This note summarizes what changed in this revision round, which decisions were
made without consulting you first (and why), and which specific items need your
sign-off before submission. Full detail is in `RESPONSE_TO_REVIEWERS.md` (letter
to the editor/reviewers) and `CHANGES_ROUND2.md` (line-level diff against the
June submission).

## Where things stand

Both reviewers' comments are addressed. Reviewer 2 (returning reviewer, five
points, all wording/framing) closed with: *"I believe the paper will be
completely ready for publication after these minor revisions."* Reviewer 3
(first-time reviewer for this journal) raised nine points; two required new
computation, the rest are addressed by rewriting or by an explained decision not
to expand scope.

Manuscript recompiles clean: 35 pages, zero undefined references, zero bibtex
warnings. No previously reported result changed — every table number from the
June submission is untouched.

## Three decisions I made without asking you first, flagged here for review

I made these calls unilaterally to keep the revision moving, following the
project's standing rule of preserving existing results and scope unless a
reviewer point specifically requires otherwise. All three are reversible before
submission if you disagree.

**1. Declined to extend country coverage or date range (Reviewer 3).**
The reviewer asked why only three countries and why not the full available date
range (e.g., Nigeria back to 1997, data through last week). This is not a text
fix — it means re-extracting and re-validating a larger raw ACLED dataset and
rerunning the full pipeline against it. I responded that the ACLED extract's
scope was fixed at collection time and framed this as future work rather than a
same-round rerun. **Flag:** if either of you has appetite for this as a
follow-up paper or wants a stronger commitment in the response letter (e.g., a
timeline), let me know before I finalize the letter.

**2. Added a bounded Poisson regression supplement rather than declining the
regression request (Reviewer 3).** The reviewer's request to add a count-based
regression task alongside the binary classifier was the one item that could
plausibly change the paper's scope. I chose the middle path: a supplementary
Poisson regression analysis (new Methods + Results subsections, Table 9),
reported alongside — not replacing — the binary classification framing, with
an explicit stated reason (early-warning-trigger use case vs. magnitude
forecasting) for keeping binary as primary. **Flag:** if you'd rather the
regression result NOT appear in the main text at all (e.g., pushed to
supplementary material only, or omitted with a written defense instead), this
is easy to move — it's self-contained in `sec:poisson_methods` /
`sec:poisson_results`.

**3. Kept the paper's title unchanged.** Reviewer 2's dependent-variable point
(see below) technically extends to "the title, abstract, analysis, and
conclusion." I renamed the outcome to "abduction/forced-disappearance events"
everywhere in the body but kept the title as "Predicting Enforced
Disappearances..." for the human-rights framing and searchability, relying on
the abstract's first sentence to clarify the operational outcome. This is
flagged explicitly in the response letter to the editor rather than presented
as full compliance. **Flag:** this is the single highest-visibility remaining
gap between the reviewer's literal request and what we did — worth a quick
read of the new abstract opening sentence to confirm you're comfortable with
this framing before submission.

## What changed, by reviewer point

**Reviewer 2 — all five points, text-only, no rerun:**
1. Outcome renamed to "abduction/forced-disappearance events" throughout the
   body (title excluded — see decision 3 above); added a Methods clarification
   that no perpetrator-type filter was applied to the ACLED data.
2. Four causal-language passages rewritten as predictive-association
   statements; Section 5.2 retitled "Features Contributing to Predictive
   Performance" (reviewer's own suggested wording).
3. "No predictive value" replaced with the reviewer's suggested wording ("no
   incremental predictive value beyond the event-based features in this
   dataset and design") at 5 sites, plus 2 further sites making the same claim.
4. Generalizability language softened to "robust within Nigeria, Mexico, and
   Myanmar" rather than "generalizes"/"ensures."
5. Added ACLED sourcing/verification description; qualified the "less
   problematic" reporting-bias claim.

**Reviewer 3 — confirmed problems, resolved by rewrite or citation fix:**
- VIEWS/admin1 citation was internally inconsistent (our own Related Work
  correctly describes ViEWS as country/PRIO-GRID-level, but Methods cited it to
  justify admin1) — fixed by citing Oswald (2026, JCR) instead.
- "Onset" framing was misleading given the actual 37–48% region-month
  prevalence — replaced with "incidence" throughout; the rare-event
  methodological framing (King–Zeng, civil-war-onset comparison) reworded so it
  no longer implies the classification task itself is rare.
- Introduction reordered to lead with motivation; added an LSTM-choice
  justification and a related-work paragraph acknowledging the shift toward
  regression/count-based conflict forecasting.

**Reviewer 3 — required new computation:**
- Paired bootstrap (10,000 resamples) and permutation test (10,000 label
  swaps) on the LSTM-vs-LR AUPRC gap: 95% CI [-0.008, +0.014], p = 0.708 — the
  0.003-point raw AUPRC advantage is not statistically distinguishable from
  zero. This does not undercut our existing generalization-gap argument for
  preferring the LSTM (LSTM: −0.036/−0.039 train→test gap vs. LR: +0.049); we
  now state both findings together and let the generalization-gap argument,
  not the raw AUPRC, carry the case for the LSTM.
- Poisson regression supplement (decision 2 above): Events Only beats Full
  Model on every metric, same direction as the existing classification
  ablations — network features add no value here either.

**Housekeeping:** an old duplicate submission PDF (`pred_eds_ml_theory_testing_
SUBMISSION_READY.pdf`) sitting in `paper/` from before this round was deleted
during the initial code/document migration — unrelated to the reviewer
response, noted here only so its absence isn't a surprise.

## What I need from you

1. Sign off on decisions 1–3 above (or tell me to change them) before I treat
   the response letter as final.
2. A read of `RESPONSE_TO_REVIEWERS.md` — it's structured as the actual
   point-by-point letter to the editor and is close to submission-ready, but
   you should both read it since it commits to specific positions (especially
   on the country-scope and title decisions) on your behalf.
3. Once approved, I can produce the final EPJ Data Science submission package
   (revised PDF + response letter + any required cover-letter template) —
   let me know if the journal has a specific format for the response letter
   beyond a plain document.
