# Summary of Changes — Round 3

Companion to `paper/RESPONSE_TO_REVIEWERS_R3.md`. Ordered by manuscript section so
Constantinos and Nikandros can review the paper and this list side by side. Reviewer point
labels are those used in `docs/ROUND2_REVIEWER_ASSESSMENT.md`: **R2-1** to **R2-5** for
Reviewer 2, **A** to **I** for Reviewer 3. Changes marked **self** were not requested by
either reviewer.

Section numbers are the revised manuscript's. Table and figure numbers are as they appear
in the revised paper: Table 3 model comparison, Table 4 feature ablation, Table 5
events-only across model families, Table 6 SHAP attribution, Table 7 recommended model,
Table 8 country breakdown, Table 9 Poisson appendix.

---

## Whole-document changes

| Change | Point | Where |
|---|---|---|
| Outcome renamed "abduction/forced-disappearance events" | R2-1 | title, abstract, §1.1, §3.2, §4, §6 |
| "No predictive value" replaced with the reviewer's wording, all 5 sites | R2-3 | abstract, Fig. 3 caption, §5.1 (×2), §6 |
| Onset language replaced with incidence/recurrence | B | §1, §2.1, §2.3, §3.2 |
| Country lists updated from three countries to five | G | §1, §3.1, §5.6 |
| Every reported number recomputed on the five-country sample | — | all of §4, §5 |
| Remaining absolutist network-null wording ("add nothing", "Why Network Features Fail" heading, "Network features consistently fail") scoped to the reviewer's adopted wording — 4 sites missed in the R2-3 pass | R2-3, self | abstract, §5.4 (×2), §5.4 heading |
| Recency wording ("recency gradient", "confirms temporal structure") aligned with §4.4's non-monotonic finding — 3 sites missed in the earlier pass | self | abstract, Fig. 4 caption, §5.5 |
| Zero-history overclaim ("does not anticipate onset", "skill does not extend") softened to "predictive skill was not demonstrated on this limited subset" — 2 sites missed in the B pass | B, self | §4.3, §5.4 |
| Gain-denominator convention (share of Full Model AUPRC) replaced with share of the gain over the random baseline, matching §4.4/§5.2 — 2 sites missed in the self pass | self | §4.2, Fig. 3 caption |

---

## §1 Introduction

| Change | Point |
|---|---|
| Opening paragraph rewritten around the absence of a record as the defining property, replacing the abstract concealment/intimidation characterisation | E |
| New paragraph on why this is a forecasting rather than a documentation problem: documentation is retrospective, entry depends on monitor presence, monitoring capacity is committed before the period it covers | E |
| Country list now Afghanistan, Mexico, Myanmar, Nigeria, Syria | G |
| Research questions reworded to the renamed outcome | R2-1 |

## §2 Related Work

| Change | Point |
|---|---|
| §2.1 now acknowledges the field's shift toward count and magnitude targets, and states why the binary framing is retained (monitoring-allocation trigger, not magnitude estimate), pointing forward to §4.7 | F |
| §2.1 reframed as recurrence forecasting in known-affected regions rather than onset detection | B |
| §2.3 retitled "Event-Level Rarity, Class Balance, and Metric Choice"; event-level rarity and region-month prevalence separated; AUPRC justified on its own terms for this target rather than by appeal to rare-event methodology | C |
| §2.4 now justifies the recurrent architecture explicitly: small input space (4–16 features over 6 timesteps) favours parameter economy over an attention-only model, and the closest prior conflict-forecasting work uses LSTM variants | I |
| One transitional word removed from a related-work paragraph; sentence now opens on the substantive claim | self |

## §3 Data and Methods

| Change | Point |
|---|---|
| §3.1 selection paragraph rewritten around recorded outcome volume: Syria 4,841 events, Nigeria 2,899, Myanmar 1,744, Mexico 1,472, Afghanistan 527; 135 admin1 units across five world regions | G |
| §3.1 sampling frame stated for the first time: four maritime pseudo-entities (149 events, unit of analysis undefined) and Taiwan (6,190 events, 21 units, 6,093 protests, zero disappearances), with the direction of effect — excluding Taiwan lowers reported AUROC from 0.8483 to 0.8174 | G, self |
| §3.1 date window defended on coverage comparability: ACLED back-coded its Latin America file to the start of 2018, making January 2018 the first month with comparable sourcing across all five countries. New citation `acledlac2020` | G |
| §3.1 admin1 justification no longer appeals to ViEWS; now rests on the data — 135 admin1 vs 3,044 admin2 units, median 14 vs 2 events per unit-month, 24.8% vs 76.1% of unit-months below five events, 10,506 of 11,340 possible admin1 region-months observed. Oswald (2026) cited as an admin0/admin1 precedent | A |
| §3.1 ACLED sourcing and coding procedure described; sub-event-type boundary stated (combines abduction and forced disappearance, does not condition on perpetrator, does not require subsequent non-acknowledgment) | R2-1, R2-5 |
| §3.1 "less problematic" claim qualified: residual risk stated, reporting environments named as part of the measurement process | R2-5 |
| §3.1 design paragraph now states the sample is purposive and points to §4.6 for per-country results | R2-4, G |
| §3.2 train and test prevalence (31.1%, 46.0%) stated where the task is defined | B, C |
| §3.10 new: supplementary regression method — L2-penalised Poisson GLM, log link, grid search over the penalty with five-fold `TimeSeriesSplit` on Poisson deviance, identical window construction and split to the classifier | H |
| §3.6 and §3.8 early-stopping wording corrected: the code keys patience, checkpointing, and the LR scheduler on evaluation loss, not test AUPRC as previously stated. Neither section now names a validation split, since none exists in the reported runs | self |
| Class weight corrected from 1.683 (stale three-country prevalence, 0.627/0.373) to 2.213 (actual five-country training prevalence, 0.311), matching Fig. 1 (`pos_weight` 2.2127) and every logged LSTM run — 6 sites | self | §3.6, §3.8 (×3), §3.9 (×2) |
| §3.6 formula qualifier changed from "$p = 0.311$" to "the unrounded training-set positive rate ($p \approx 0.311$)" — the rounded value implies 2.215, not 2.213 | self |
| §3.6 dataset size corrected from "≈6,000" to "9,696" sequences, matching the count already stated in §3.2 and Table 4's caption | self |

## §4 Results

| Change | Point |
|---|---|
| §4.1 train-to-test generalisation-gap argument removed | D, self |
| Table 3 rebuilt: single gap convention (train − test) across all five models, replacing two rows that used the opposite sign. All five models score higher on test than train | self |
| §4.1 now reports the top four models within 0.0028 of each other, with the untuned linear model ahead of both sequence models | D |
| §4.2 network null stated with the reviewer's wording and the three-family replication (LSTM +0.0052, tuned LR +0.0006, XGBoost +0.0134) | R2-3, D |
| §4.3 zero-history subset: 8 of 135 regions, 160 test sequences at 17.5% positive, AUPRC 0.139 ± 0.010 against a 0.175 no-skill baseline, same model 0.719 ± 0.010 on the full test set. Claim limited to skill not demonstrated, not skill absent | B |
| §4.4 feature-contribution shares recomputed against the gain over the random baseline rather than against zero: full-model gain 0.349, withholding the lagged feature costs 0.086 of it — a quarter, with three quarters carried by the other three event counts | self |
| §4.4 attribution and non-substitutable information distinguished: the lagged feature's attribution advantage is 15.2× the next largest, while withholding it costs 0.086 of the 0.349 gain | self |
| Table 6 rebuilt; the three non-lagged features reordered and now separate by 0.021/0.017/0.016, so the text states their ordering should not be read as a ranking | self |
| §4.4 recency claim softened: per-timestep attribution is 0.079, 0.074, 0.072, 0.081, 0.111, 0.133 from oldest to most recent, a 1.6× tilt that dips mid-window rather than rising monotonically | self |
| §4.6 rewritten to report lift over each country's own test prevalence, since no-skill baselines range 0.232–0.637. Raw AUPRC spread 38.9 points, lift spread 10.7 points, within-country seed SD ≤ 0.0016 | R2-4 |
| §4.6 states that comparable lift does not establish that the model reads the same precursors in each setting | R2-4 |
| §4.7 new: Poisson results with the full five-metric suite for both feature sets; neither configuration dominates; binary-threshold match rate 1.0. Test count distribution corrected to range 0–47, mean 1.69, s.d. 4.20, 54.1% zeros, n = 3,036 | H |
| Table 9 rebuilt from `results/exp6_poisson_regression.json`; bolding removed, since the previous version marked one configuration as winning on all five metrics | H, self |
| §4.5 Events Only vs Full Model improvement corrected from "+0.004" to "+0.005 AUPRC", matching Table 5's own $\Delta$ column (0.814 − 0.809) | self |
| §4.2 "the next subsection shows it does not extend" reworded to "tests whether that margin extends" — missed in the earlier zero-history pass | B, self |
| §4.3 "at or below chance" sentence replaced: point estimate below the no-skill baseline, skill not demonstrated on 160 sequences / 28 positives; the persistence-margin clause kept as a separate sentence | B, self |
| Figure 3 (`ablation_chart.png`/`.pdf`, `src/create_ablation_chart.py`) regenerated: panel-a title scoped to "Tested network features do not improve on four event counts in this design"; the two drop labels changed from a percent-of-Full-Model figure to raw AUPRC lost plus percent of the gain over the random baseline (−0.086 AUPRC, ≈25% of gain above random; −0.116 AUPRC, ≈33%), matching the §4.2 prose denominator. Caption's "does not extend" replaced with "was not demonstrated" | self |

## §5 Discussion

| Change | Point |
|---|---|
| §5.1 network-null wording adopted verbatim from the reviewer, both sites | R2-3 |
| §5.2 retitled "Features Contributing to Predictive Performance" | R2-2 |
| "Primary driver", "create the operational environment" and "specifies this mechanism" formulations rewritten as predictive association | R2-2 |
| §5.3 data-limited-ceiling argument added: three model families with different inductive biases converging within a narrow band on the same features indicates a ceiling set by the information in monthly event counts, not by the function class | D |
| §5.3 states the Attention-LSTM's role as the capacity test that makes the network null credible, with the linear model's parity closing the converse argument | D |
| §5.3 unsupported sensitivity-to-data-quality claim replaced; the untested dimension is named as untested | self |
| §5.5 now states the predictive design cannot separate shared antecedents from context-specific ones | R2-2 |
| §5.6 Syria concentration stated: 42.2% of disappearance events, 10.7% of test sequences, 10.4% of regions; highest raw AUPRC; the four others each retain ≥ +0.233 lift; leave-one-country-out retrain named as the more direct check and as not run | G, self |
| §5.6 reporting-bias paragraph updated to five countries, adding government-held Syria alongside post-2021 Myanmar | G |
| §5.7 external-validity paragraph rewritten; Syria removed from the future-validation list since it is now in the sample (mislabeled §5.6 in an earlier version of this list) | G |
| §5.4 both network-null statements ("event-based features outperform...", "the absence of predictive value...") scoped to "the tested co-occurrence-network features ... in this dataset and design" — missed in the R2-3/§5.1 pass | R2-3, self |
| §5.5 "temporally structured way" claim replaced with the tested non-monotonic recency tilt, matching §4.4 — missed in the earlier recency-wording pass | self |
| §5.6 corrected: Syria has the highest raw AUPRC (0.960) but the *second*-highest lift over own prevalence (+0.326), behind Afghanistan (+0.340) — an earlier pass had called it the highest-lift country, which `results/exp9_country_breakdown.json` does not support | self |
| §4.6 "that ordering follows each country's test prevalence" reworded to "raw AUPRC broadly tracks test prevalence" | self |
| §5.2 "leaves three quarters standing on arrests, violence, and fatalities alone" reworded to "approximately three quarters of the gain remains in the remaining-feature configuration" | self |
| §5.2 "AUPRC has a floor at the positive rate" reworded to "the expected AUPRC of a random classifier equals the positive rate" | self |
| Test-period positive rate standardized to 45.9% (exact: 45.9486%) — 6 sites previously read 46.0%: §3.4 (×2), §3.7, §4.1 body, Table 3 caption, §4.6 Table 8 total row | self |
| §5.2 "leaving three quarters standing on arrests, violence, and fatalities alone" was factually wrong — the 15-feature no-disappearances configuration retains all 12 network features too. Reworded to state the 0.723 AUPRC without attributing it to any single feature family | self |
| §4.6 "these between-country differences are not sampling noise" reworded to "not explained by seed-to-seed training variation" — seed spread measures training variation, not a sampling-uncertainty estimate | self |
| §5.6 removed the unsupported inference that a sample without Syria "would score lower overall" and that no country "carries the pooled result"; kept only the four-country +0.233 lift floor and named leave-one-country-out retraining as the untested direct check | self |
| Figure 1 caption: "prevents information leakage" reworded to "separates training on 2018--2022 from evaluation on 2023--2024" (temporal split controls information flow, it does not by itself prove leakage is absent); "within sampling noise of the seven best-performing configurations" reworded to "within 0.006 AUPRC," matching the §4.2 rank 1--7 span | self |

## §6 Conclusions

| Change | Point |
|---|---|
| Renamed outcome and the reviewer's network-null wording | R2-1, R2-3 |
| Unsupported distributional-shift claim replaced with the tested claim — the null appears in all three model families | self |
| "SHAP attribution shows temporal structure rather than a single lag" reworded to "shows a non-monotonic recency tilt ... rather than weight concentrated in a single lag" — missed in the earlier recency-wording pass | self |

## Bibliography

| Change | Point |
|---|---|
| `acledlac2020` added — ACLED's December 2020 Latin America back-coding release, supporting the January 2018 window start | G |
| `oswald2026googletrends` cited in §3.1 as an admin0/admin1 precedent | A |

---

## Provenance

- `docs/MANUSCRIPT_DELTA.md` — the running record of every edit, with the reasoning and
  the source file behind each number, including the three withdrawn claims.
- `docs/numbers_of_record.csv` — 183 reported quantities, each with its `results/` source.
- `docs/DECISIONS_ROUND3.md` — decisions S1–S5 and the evidence behind each.
- `docs/ROUND2_REVIEWER_ASSESSMENT.md` — the reviewer points as triaged, with each claim
  checked against the submitted manuscript text.
