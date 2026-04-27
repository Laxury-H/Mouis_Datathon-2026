# 🏆 Champion Model: V33 Time-Decay Blend + Sync COGS (664k MAE)

## 📌 Architecture
This model utilizes a **Time-Decay Blending Strategy** to solve the problem of cumulative error (drift) in 18-month recursive forecasting.
- **Short-Term Anchor (Months 1-6):** Heavily relies on V25 (Recursive Components) + V18 (Deep Learning) for extremely accurate local trends.
- **Long-Term Anchor (Months 7-18):** Gradually transitions weight to V28 (One-Shot Components) to maintain stability and prevent recursive drift.
- **Sync COGS**: Replaces the static 0.8862 COGS ratio with the dynamic, day-level COGS ratio predicted by the V18 DL Stack, reducing MAE by an additional ~1000 points.

Specifically, it uses a linear time-decay shifting from **10% V28 (Day 1)** to **50% V28 (Day 548)**.

## ⚙️ Reproducibility (Điều Kiện Tái Lập)
To ensure 100% reproducibility of the 664,928 MAE submission, all dependent scripts have been snapshotted into the `src/` directory.

### Quick Start
You can run the full pipeline automatically from the root directory:
```powershell
python Results/Champions/V33_TimeDecay_664k_syncCOGS/pipeline.py
```

### Manual Steps
If you prefer running step-by-step from the project root:
1. `python Results/Champions/V33_TimeDecay_664k_syncCOGS/src/model_v18_dl_stack.py`
2. `python Results/Champions/V33_TimeDecay_664k_syncCOGS/src/model_v25_components.py`
3. `python Results/Champions/V33_TimeDecay_664k_syncCOGS/src/blend_v25_sweep.py`
4. `python Results/Champions/V33_TimeDecay_664k_syncCOGS/src/model_v28_oneshot_components.py`
5. `python Results/Champions/V33_TimeDecay_664k_syncCOGS/src/blend_time_decay.py`

The final output will be generated at:
`Results/submissions/final_sweeps/submission_time_decay_10_50_syncCOGS.csv`
