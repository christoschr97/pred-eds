# pipeline_diagram.png — every value that changes on five countries

> **Resolved.** The diagram was redrawn in matplotlib as
> `src/create_pipeline_diagram.py`, reproducing the submitted layout with every
> number read from `results/`. It is no longer hand-maintained, so this list is
> now a record of what changed rather than a set of edits to apply. One further
> difference from the submitted version, beyond the table below: the submitted
> diagram drew **eight** ablation tiles under a caption reading "9
> configurations" — Network Only was missing. All nine are now drawn.

`pipeline_diagram.png` has no producing script and no vector source. Searched
both trees for `*.svg`, `*.drawio`, `*.excalidraw`, `*.pptx`, `*.key`, `*.ai`
and for any script that writes the filename: nothing. The only artefacts are
two byte-identical PNGs (md5 `547bb4a9`), one in
`DISACT-GNN/lstm_new_features/sn-article-template/figures/` and the copy in
`paper/figures/`. It was drawn by hand in a tool that left no file behind.

Values below were read off the submitted PNG and checked against the result
files. Ten cells change; the rest of the diagram is still correct.

## Changes

| Box | Submitted | Five-country | Source |
|---|---|---|---|
| ACLED Raw Events | 202,347 events | **360,530 events** | loader stdout; `descriptives_5c.json` |
| ACLED Raw Events | Nigeria · Mexico · Myanmar | **Nigeria · Mexico · Myanmar · Afghanistan · Syria** | data file |
| Region-Month Aggregation | 87 regions × 84 months | **135 regions × 84 months** | `exp8...json` → `sample.n_regions` |
| Temporal Sequences | 6,293 sequences total | **9,696 sequences total** | 6,660 + 3,036, `exp8` → `sample` |
| Training band | 4,258 sequences (37.3% positive) | **6,660 sequences (31.1% positive)** | `exp8` → `sample.n_train_sequences`, `train_positive_rate` |
| Test band | 2,035 sequences (48.3% positive) | **3,036 sequences (45.9% positive)** | `exp8` → `sample`; `exp10` → `test_positive_rate` = 0.4595 |
| Attention box | (+0.1% AUPRC) | **−0.1% AUPRC** | `exp8` → `summary`: attention 0.8081 vs standard 0.8089 |
| Weighted BCE Loss | w₊ = 1.683 | **w₊ = 2.213** | `exp8` → `config.pos_weight` = 2.2127 |
| Ablation, Events Only tile | 4 feat · **best** | **4 feat · highest mean** | see note below |
| Ablation, No Disappear. tile | −10.1% | **−10.7%** | `ablations_5c.json`: 0.7225 vs full 0.8088 |

## Unchanged — do not touch

Event Features (4 features: arrests, violence, disappearances, fatalities);
Network Features (12 features: density, centrality, clustering, weights);
sequence window `t−5 … t (input) → t+1`; the split dates (Training Jul 2018 –
Dec 2022, Test Jan 2023 – Dec 2024); Input Tensor `(batch, 6, F)`, `F = 4 or
16`; 2-Layer LSTM, 64 hidden units, dropout 0.2; FC Layers 64→32→1 + sigmoid;
Adam, lr = 0.001, batch = 64; early stopping patience 10, max 50 epochs; AUPRC
as primary metric; the nine ablation tiles and `9 configurations × 3 seeds = 27
experiments`.

The window annotation already reads `t+1`, so the diagram needs no change on
that point.

## Two wording changes, not number changes

**"best" on the Events Only tile.** On five countries the seven configurations
that keep past-disappearance history span 0.8080 to 0.8140. That 0.0060 range
is smaller than the ±0.0071 chance variation on this test set (1.96 × the
paired permutation null SD of 0.003614, `exp10`). Events Only has the highest
mean and is the right model to recommend on parsimony, but it cannot be called
best. `ablation_chart.png` was rebuilt on the same reasoning.

**The attention box.** It is already drawn dashed, as optional. The parenthetical
now has to read as a cost rather than a gain: attention scores 0.0008 below the
plain LSTM. Since the attention weights carry the temporal-recency analysis, the
box stays — but the diagram should not imply it buys accuracy.

## Percentages: which denominator

`No Past Disappearances` (15 features) and `Network Only` (12) are each the full
16-feature set minus one block, so both are quoted against the Full Model
(0.8088), matching the `delta_from_full` field in `ablations_5c.json` and the
percentage in the manuscript text: −10.7% and −14.3%. Do not quote them against
Events Only (which would give −11.2% and −14.9%).
