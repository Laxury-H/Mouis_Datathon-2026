# History Archive

This folder stores archived competition artifacts that are no longer part of the active working set.

## What is archived here

- `Results/history/submissions/v17/` contains the archived v17 submission files.
- `Results/history/submissions/v18/` contains the archived v18 sweep files.
- `Results/submissions/v18/` contains the current v18 submission candidates.

## How to reproduce the archived versions

Run from the repository root with the same deterministic environment used during model generation:

```bash
PYTHONHASHSEED=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python Data/model_v17_1.py
PYTHONHASHSEED=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python Data/model_v17_1b.py
PYTHONHASHSEED=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python Data/model_v17_2.py
PYTHONHASHSEED=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python Data/model_v17_dl_anchor.py
PYTHONHASHSEED=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python Data/model_v18_dl_stack.py
```

The latest v18 sweep files were generated from `Data/model_v18_dl_stack.py`, with the strongest observed anchor in the final micro-sweep being `submission_v18_dl_stack_anchor_a12.csv`.

## Notes

- The active default submission file is `Data/submission.csv`.
- Archived submission files are stored by version family so you can compare public leaderboard behavior over time.
- For the latest candidate set, use `Results/submissions/v18/`.
