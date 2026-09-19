# Round 3 revision plan — status as of 2026-09-12

**Plan:** EPJ Data Science major revision on the five-country sample, step by step
(created 2026-09-08). Nine steps, eight complete, one open.

**Deliverables the plan committed to**

1. Revised `sn-article.tex` and compiled PDF — .tex done, PDF pending step 9
2. Point-by-point response letter for round 3 — done
3. `results/` and figures regenerated on the five-country sample — done
4. Verification log with every manuscript number checked — pending step 9

**Sequencing constraint the plan was built around.** Every reported number moves on the
five-country sample, so results and figures had to be regenerated before any text was
written, or the text gets written twice. Steps 2 and 3 therefore precede all four text
tranches.

---

## 1. Lock the remaining decisions — DONE

Two decisions were already fixed going in: report the five-country sample only, and say
nothing to the editor about the target-construction change. Two were open — whether to
answer Reviewer H with a Poisson appendix or an operational defence, and whether to promote
tuned LR over the Attention-LSTM as the primary model.

**Outcome.** Both resolved in `docs/DECISIONS_ROUND3.md` as S1–S5. H answered with the
Poisson appendix *and* the operational defence, since the count model turned out to be
moderately predictive and so could not support a "counts are unpredictable" argument. D
answered by reporting the two models as statistically indistinguishable rather than
promoting either. `results/exp4_zero_history_subset.json` was checked first as planned and
does score the no-prior-history subset, which is what let point B be answered with a
measurement instead of a word change.

**Produced.** `docs/DECISIONS_ROUND3.md`, `docs/INTERNAL_NOTE_target_index.md`
(internal only, not referenced in the letter).

## 2. Regenerate every number of record — DONE

**Produced.** `results/ablations_5c.json`, `descriptives_5c.json`,
`test_count_distribution_5c.json`, and `exp1` through `exp10`:
tuned LR, XGBoost, persistence, zero-history subset, SHAP, Poisson, paired bootstrap,
architecture comparison, country breakdown, events-only bootstrap. Plus
`docs/numbers_of_record.csv` — 183 rows, one per quantity the manuscript states, each with
its source file. Every text edit in steps 4–7 cites that file.

**Deviation from plan.** Three experiments were added beyond exp1–exp7: `exp8`
architecture comparison, `exp9` country breakdown, `exp10` events-only paired bootstrap.
exp9 was needed because raw per-country AUPRC is not comparable across countries with
different no-skill baselines; exp10 because the full-feature bootstrap alone left the
events-only comparison untested.

## 3. Rebuild the manuscript figures — DONE

**Produced.** `paper/figures/`: `ablation_chart`, `figure1_time_series`,
`fig1_shap_bar`, `fig2_shap_heatmap`, `fig3_shap_direction`, `pipeline_diagram` (PNG and
PDF each), plus `exp7_bootstrap_diff.png` and `exp10_bootstrap_events_only.png` — the
LR-versus-LSTM null figures the plan asked for when point D was promoted to a main-text
finding.

## 4. Text tranche A — Reviewer 2's five points — DONE

All five implemented in `paper/sn-article.tex`. Both of the reviewer's drop-in replacement
formulations adopted verbatim. Edits logged with reasoning in `docs/MANUSCRIPT_DELTA.md`.

## 5. Text tranche B — the framing rebuild (A, B, C) — DONE

The admin1 justification no longer appeals to ViEWS; it rests on the extract's own
sufficiency numbers. Onset framing replaced with recurrence and incidence. §2.3
restructured to separate event-level rarity from the region-month target.

**Finding that changed the rest of the work.** The `.tex` on disk is not the submitted
version — several reviewer requests had already been absorbed into it in an earlier commit.
From this step on, every tranche checked what the file already said before editing it.

## 6. Text tranche C — model comparison and related work (D, E, F, I) — DONE

D promoted to a results finding with the paired bootstrap and the data-limited-ceiling
argument. Points F and I needed no edit — already applied by the author. Two claims
withdrawn here rather than revised: the generalization-gap defence and the
feature-contribution shares.

## 7. Text tranche D — scope answers (G, H) — DONE

G answered in both halves: countries on recorded outcome volume, date range on ACLED
coverage comparability (new bib entry `acledlac2020`). Sampling frame stated for the first
time, with the direction of the Taiwan exclusion — which lowers every number we report —
stated explicitly. H implemented as the Poisson appendix, §3.10 and §4.7.

**Not done, by decision.** The leave-one-country-out check the plan offered conditionally
("if approved") was not run. The Syria concentration is instead addressed with event,
region and test-sequence shares in §5.6, which states that a LOCO retrain would settle it
more directly and was not performed.

## 8. Assemble the point-by-point response letter — DONE

**Produced.** `paper/RESPONSE_TO_REVIEWERS_R3.md` (14 reviewer points, plus sections on
where the two reports interact, three claims withdrawn, four requests not implemented, and
a marked internal note to delete before submission) and `paper/CHANGES_ROUND3.md` (every
change by manuscript section, each mapped to a reviewer point label or marked as
self-identified).

Constraint honoured: no characterisation of what the previous round's numbers represented,
and no claim that the pipeline was rerun unchanged.

## 9. Final verification pass — OPEN

Extract every numeric claim from the revised `sn-article.tex`, check each against
`results/` programmatically, confirm reviewer-point coverage, recompile the PDF, log
residual discrepancies.

**Deliverable.** `docs/verification_log_round3.md` with a pass/fail row per claim, plus
`paper/sn-article.pdf`.

**Known open item to close in this step.** The Taiwan exclusion figures disagree:
`docs/DECISIONS_ROUND3.md` records 2.9 AUROC points (0.823 → 0.852), `src/data_prep.py`
lines 52–53 record 3.1 points (0.8174 → 0.8483). Neither has a file in `results/`. The
manuscript and the letter quote the `data_prep.py` figures. Resolution is either one
with-Taiwan run written into `results/`, or dropping the point estimates from §3.1 and
stating the direction only.
