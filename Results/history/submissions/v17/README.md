# Archived v17 Submission Files

This folder contains the v17 submission CSVs that were generated during the modeling process and later moved out of `Data/` to keep the workspace tidy.

## Archived files

- `submission_v17_1.csv`
- `submission_v17_1b.csv`
- `submission_v17_2.csv`
- `submission_v17_2_rescue.csv`
- `submission_v17_3_bal75.csv`
- `submission_v17_3_bal80.csv`
- `submission_v17_3_push70.csv`
- `submission_v17_3_safe85.csv`
- `submission_v17_3_safe90.csv`
- `submission_v17_4_joint_n00.csv`
- `submission_v17_4_joint_n02.csv`
- `submission_v17_4_joint_n04.csv`
- `submission_v17_4_joint_n06.csv`
- `submission_v17_4_joint_n08.csv`
- `submission_v17_4_ratio_mid_b20.csv`
- `submission_v17_4_ratio_mid_b35.csv`
- `submission_v17_4_ratio_mid_b50.csv`
- `submission_v17_4_ratio_season_g15.csv`
- `submission_v17_4_ratio_season_g25.csv`
- `submission_v17_4_ratio_season_g35.csv`
- `submission_v17_4_ratio_season_g45.csv`
- `submission_v17_5_ratio_doy_g20.csv`
- `submission_v17_5_ratio_doy_g22.csv`
- `submission_v17_5_ratio_doy_g24.csv`
- `submission_v17_5_ratio_doy_g26.csv`
- `submission_v17_5_ratio_doy_g28.csv`
- `submission_v17_5_ratio_doy_g30.csv`
- `submission_v17_5_ratio_mix_a15.csv`
- `submission_v17_5_ratio_mix_a20.csv`
- `submission_v17_5_ratio_mix_a25.csv`
- `submission_v17_6_ratio_doy_g27.csv`
- `submission_v17_dl_anchor_a05.csv`
- `submission_v17_dl_anchor_a10.csv`
- `submission_v17_dl_anchor_a15.csv`
- `submission_v17_dl_anchor_a20.csv`
- `submission_v17_dl_raw.csv`

## Reproduce with the active scripts

Run from the repository root:

```bash
PYTHONHASHSEED=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python Data/model_v17_1.py
PYTHONHASHSEED=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python Data/model_v17_1b.py
PYTHONHASHSEED=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python Data/model_v17_2.py
PYTHONHASHSEED=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python Data/model_v17_dl_anchor.py
```

Each script writes its own submission file back into `Data/` when executed.
