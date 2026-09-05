# Run Guide

How to execute the migrated pipeline and reproduce the numbers in the
submitted manuscript. Every command below was run in this session on a
local machine (8 CPU, 16 GB RAM, no GPU) and its output is logged in
`verification_log.md`.

## 1. Environment

The paper's stated environment (Methods, reproducibility paragraph):
PyTorch 2.0.1, NumPy 1.24.3, Pandas 2.0.2, NetworkX 3.1, scikit-learn 1.3.0.
This is exactly what was installed and used for the verification run below:

```bash
conda create -n disact-repro python=3.10 \
    numpy=1.24.3 pandas=2.0.2 scikit-learn=1.3.0 networkx=3.1 \
    matplotlib seaborn tqdm pytorch=2.0.1 cpuonly \
    -c pytorch -c conda-forge
conda activate disact-repro
pip install xgboost shap   # needed for revision_experiments/exp2 and exp5 only
```

## 2. Data placement

The raw ACLED CSV is **not** in this repository (redistribution
restriction, see `AUDIT.md` §5). Place your own licensed copy at:

```
disact-pred-eds/data/ACLED Data_2025-12-31_Nigeria_Mexico_Myanmar.csv
```

`data/` is already in `.gitignore`. Several scripts (`verify_statistics.py`,
`revision_experiments/exp1-4`) hard-code this exact relative/absolute path
pattern — this is the original code's own path plumbing, unmodified.
For this session's verification a symlink to the source repo's copy
(`projects/DISACT-GNN/data/...`) was used at that location.

## 3. Confirm the raw data matches the paper's stated dataset

```bash
cd src
python verify_statistics.py
```

Checks event count, disappearance count, admin1 region count, arrests,
fatalities, and date range against the numbers quoted in the manuscript
text (Sec. 3). No model training; runs in seconds.

## 4. Reproduce Table 4 (nine-configuration ablation, 27 runs)

```bash
cd src
python run_ablations.py \
    --data "../data/ACLED Data_2025-12-31_Nigeria_Mexico_Myanmar.csv" \
    --seeds 3 --epochs 50 --device cpu
```

Measured runtime this session: ~10 minutes total (9 configs × 3 seeds).
The paper's own text estimates "2–3 minutes per random seed" — this
machine trains faster than that per run once data loading is amortized,
so total wall time undercuts the paper's own estimate.

This script always runs all nine configurations (no CLI flag to select
one); to run a single configuration/seed for a quick check, import
`run_ablation_experiment` directly from a Python shell (see
`verification_log.md` for the exact snippet used in this session).

## 5. Reproduce the Table 3 non-LSTM baselines and reviewer-response experiments

```bash
cd src/revision_experiments
python exp3_persistence.py      # no training, seconds
python exp1_tuned_lr.py         # grid search, ~20s
python exp2_gbm_baseline.py     # grid search, ~2-3 min
python exp4_zero_history.py     # LSTM, 3 seeds, ~1 min
```

Each writes its own JSON to `results/`. **Sandbox note**: in this
environment, `GridSearchCV(..., n_jobs=-1)` in exp1/exp2 raised
`PermissionError: [Errno 1] Operation not permitted` from joblib's `loky`
backend trying to check POSIX semaphore limits (a container/sandbox
restriction, not a code bug). Worked around **without editing the
scripts** by running them inside a `joblib.parallel_backend("threading",
n_jobs=1)` context via a tiny external wrapper
(`run_with_threading_backend.py`, not part of the pipeline, not
migrated) — same computation, single-process. On an unrestricted
machine `python exp1_tuned_lr.py` directly works as-is.

## 6. Not runnable standalone as migrated (packaging gaps, flagged not fixed)

- **`train.py`** — uses relative imports (`from .data_prep import`,
  `from .models import`), which requires it to be imported as part of a
  Python package (`__init__.py` + `python -m <pkg>.train`), not run as a
  bare script. It also has no CLI/`argparse` block, only library
  functions (`run_comparison`, `print_comparison_table`) — the original
  repo's `publication_code/README.md` documents a `--config`/`--seed` CLI
  for it that does not exist in the actual file. It is not called by
  `run_ablations.py`, so it is not required to reproduce Table 3 or Table 4
  — cross-check the Standard-LSTM row (0.740 in Table 3) by importing its
  functions directly in a script that adds `src/` to `sys.path`, mirroring
  what `revision_experiments/*.py` already do.
- **`create_figures.py`** — imports `from lstm_new_features.data_prep import
  ...`, i.e. it expects the *original* package name from the source repo,
  not a standalone import. It was not re-run this session (the already-
  generated figure PNGs were migrated directly instead); re-running it
  requires either restoring that package name or running it from within
  `projects/DISACT-GNN` where the package already resolves.

Per the "don't refactor" instruction, neither file was edited to fix
these import paths — this section exists so the packaging gap is visible
rather than silently patched.
