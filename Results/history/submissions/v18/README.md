# V18 Deep Learning Stack Experiments

## Overview
This experiment family (V18) utilizes a stacked ensemble architecture:
- **Base Models**: XGBoost, LightGBM, and CatBoost (Tweedie Regression).
- **Meta-Model**: Linear Stacking (constrained positive weights).
- **Features**: Time-series lags, rolling windows, and dynamic multiplier tuning.
- **Goal**: Sub-680k MAE on Public LB.

## Results Summary
The experiments revealed a clear "U-shaped" performance curve on the Public Leaderboard relative to the anchoring alpha (blending with the stable V17.5 G26 model).

| Version | Alpha (New Model %) | Public LB MAE | Status |
|---------|---------------------|---------------|--------|
| `a12`   | 12%                 | 713,751       | Stable |
| `a18`   | 18%                 | 705,030       | Improving |
| `a22`   | 22%                 | 701,432       | Near Optimum |
| **`a25`** | **25%**           | **700,637**   | **Best V18** |
| `a30`   | 30%                 | 701,929       | Degrading |
| `a50`   | 50%                 | 861,965       | Overfit |
| `raw`   | 100%                | 981,882       | Failure |

## Insights
- **Overestimation**: The "Raw" model projected a Mean Revenue of ~5.1M, while the actual test distribution appears closer to ~4.2M. This caused the catastrophic performance of high-alpha versions.
- **Optimum**: The sweet spot for ensembling this complex stack with the baseline is approximately **25%**.
- **MAE vs Holdout**: Despite a very low Holdout MAE (~653k), the Public LB score was significantly higher, indicating a distribution shift between the holdout period and the test period.

## Files
- `submission_v18_dl_stack_anchor_a25.csv`: The champion of this series.
- `submission_v18_dl_stack_raw.csv`: The pure stack output.
- `model_v18_dl_stack.py`: The source script (kept in Data/ for active development).