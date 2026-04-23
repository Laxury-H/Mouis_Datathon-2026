# Archived v18 Submission Files

This folder stores the v18 DL stack sweep files that were generated from `Data/model_v18_dl_stack.py` and then moved out of `Data/` to keep the active workspace clean.

## Current v18 archive

- `submission_v18_dl_stack_raw.csv`
- `submission_v18_dl_stack_anchor_a05.csv`
- `submission_v18_dl_stack_anchor_a08.csv`
- `submission_v18_dl_stack_anchor_a09.csv`
- `submission_v18_dl_stack_anchor_a10.csv`
- `submission_v18_dl_stack_anchor_a11.csv`
- `submission_v18_dl_stack_anchor_a12.csv`
- `submission_v18_dl_stack_anchor_a13.csv`
- `submission_v18_dl_stack_anchor_a14.csv`
- `submission_v18_dl_stack_anchor_a15.csv`
- `submission_v18_dl_stack_anchor_a16.csv`
- `submission_v18_dl_stack_anchor_a20.csv`

## Reproduce the sweep

Run from the repository root:

```bash
PYTHONHASHSEED=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python Data/model_v18_dl_stack.py
```

The current file set keeps the raw DL stack output plus the anchored variants around the sweet spot near `a11` and `a12`.