# Internal note — target index in the sequence builder

**For Constantinos and Nikandros. Internal only; not part of the response letter.**

## What we found

During the code audit for round 3, the target construction in `create_temporal_sequences`
was checked against the horizon the manuscript states. With a six-month input window the
sequence builder took its label from one position past the intended target month. Inputs
were never affected — months $t-5 \ldots t$ throughout, exactly as described. Only the
label was displaced, so the model was scored against month $t+2$ while the manuscript,
the abstract and the Table 3 persistence definition all state month $t+1$.

The reason this went unnoticed is worth recording: the recorded `target_month` field held
the correct value, so every log file and every results JSON agreed with the manuscript.
The discrepancy was visible only in the index arithmetic.

It was verified two independent ways — by reading the arithmetic in source, and by pushing
a synthetic single-region series with controlled ground truth through the real pipeline
functions.

## What it does and does not affect

- **No data leakage.** The label always came from a month strictly after the input window.
- **No change to inputs, features, network construction, splitting or normalization.**
- **Relative model ordering unchanged.** All nine ablations were rerun across three
  dataset conditions, 81 runs in total. The paper's central finding — that the 12 network
  structure features add nothing beyond the 4 event-count features — holds in every
  condition, and the events-versus-network gap widens after the correction.
- **Absolute values move.** The naive persistence baseline was a two-month carry-forward
  described as one-month; corrected, it rises. Model AUPRC rises as well.
- One incidental defect closes with the same one-token change: no sequence now references
  a target month absent from the data.
- One claim in the manuscript needs correcting on its own merits regardless: the
  per-category network numbers were never a ranking. The spread between categories is the
  same order as the seed-to-seed standard deviation, and the nominally best category
  shuffles between conditions.

## What we are doing about it

The fix is in `src/data_prep.py` and `src/run_ablations.py` — both files carried their own
copy of the sequence builder, so the change was made twice, with a comment at each site.
Round 3 reports the five-country sample only, so all reported numbers are regenerated from
the corrected pipeline and the manuscript's stated one-month horizon is what the code does.

Per S2 in `docs/DECISIONS_ROUND3.md`, this is not raised with the editor: the manuscript is
unpublished, and the version that carried the mismatched statement does not survive
review. Nothing we write will characterise what the round-2 numbers represented, and
nothing will state that the pipeline was rerun unchanged on additional countries. If asked
directly about the target definition, we answer plainly.

Raise it if either of you would rather we handled it differently — the decision is
reversible right up to resubmission.
