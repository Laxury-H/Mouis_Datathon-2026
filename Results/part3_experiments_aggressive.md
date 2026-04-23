# Part 3 Aggressive Experiments (v24+)

- Timestamp: 2026-04-22 17:31:04 UTC
- Seed: 2026
- Trials run: 42
- Best submission version: submission_v24.csv

## Trial Ranking

| Rank | Trial | Phase | Family | Seed | Mean MAE | Mean RMSE | Mean R2 | Weighted MAE | Weighted RMSE | Objective | Blend a_rev | Blend a_cogs | Params |
|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 37 | refine | cat | 2024 | 574309.7436 | 747394.8789 | 0.573791 | 574309.7436 | 747394.8789 | 634889.5410 | 0.35 | 0.45 | iterations=992, learning_rate=0.06663, depth=8, l2_leaf_reg=5.231, bagging_temperature=0.614 |
| 2 | 38 | refine | cat | 2024 | 573170.5654 | 751803.8291 | 0.566685 | 573170.5654 | 751803.8291 | 635692.2077 | 0.25 | 0.80 | iterations=812, learning_rate=0.06892, depth=6, l2_leaf_reg=5.231, bagging_temperature=0.614 |
| 3 | 20 | coarse | cat | 42 | 573505.4305 | 759193.8471 | 0.573154 | 573505.4305 | 759193.8471 | 638496.3763 | 0.30 | 0.70 | iterations=463, learning_rate=0.0829, depth=9, l2_leaf_reg=6.215, bagging_temperature=1.334 |
| 4 | 22 | coarse | cat | 2024 | 575184.6253 | 759240.9147 | 0.566578 | 575184.6253 | 759240.9147 | 639604.3266 | 0.25 | 0.65 | iterations=899, learning_rate=0.08199, depth=6, l2_leaf_reg=5.231, bagging_temperature=0.614 |
| 5 | 1 | coarse | xgb | 42 | 589678.0842 | 772916.2653 | 0.553022 | 589678.0842 | 772916.2653 | 653811.4476 | 0.30 | 0.35 | n_estimators=634, max_depth=4, learning_rate=0.08399, subsample=0.8402, colsample_bytree=0.7797, reg_alpha=0.1775, reg_lambda=2.081, min_child_weight=7.289 |
| 6 | 35 | refine | cat | 42 | 591371.3592 | 780018.1934 | 0.555241 | 591371.3592 | 780018.1934 | 657397.7512 | 0.25 | 0.65 | iterations=571, learning_rate=0.0794, depth=9, l2_leaf_reg=6.215, bagging_temperature=1.334 |
| 7 | 33 | coarse | hgb | 2024 | 595700.5070 | 777304.8703 | 0.536537 | 595700.5070 | 777304.8703 | 659262.0341 | 0.35 | 0.30 | learning_rate=0.1221, max_depth=5, max_iter=690, max_leaf_nodes=100, min_samples_leaf=23, l2_regularization=0.9425 |
| 8 | 36 | refine | cat | 42 | 596927.5268 | 785166.2351 | 0.542800 | 596927.5268 | 785166.2351 | 662811.0747 | 0.30 | 0.55 | iterations=549, learning_rate=0.0905, depth=9, l2_leaf_reg=6.215, bagging_temperature=1.334 |
| 9 | 39 | refine | xgb | 42 | 600738.5047 | 785117.5450 | 0.545519 | 600738.5047 | 785117.5450 | 665271.1688 | 0.30 | 0.35 | n_estimators=751, max_depth=3, learning_rate=0.07756, subsample=0.8402, colsample_bytree=0.7797, reg_alpha=0.1775, reg_lambda=2.081, min_child_weight=7.289 |
| 10 | 29 | coarse | hgb | 2024 | 602903.0068 | 789911.5636 | 0.533740 | 602903.0068 | 789911.5636 | 668356.0017 | 0.30 | 0.25 | learning_rate=0.1358, max_depth=9, max_iter=263, max_leaf_nodes=86, min_samples_leaf=16, l2_regularization=0.6987 |
| 11 | 40 | refine | xgb | 42 | 601057.7868 | 793844.4402 | 0.536227 | 601057.7868 | 793844.4402 | 668533.1155 | 0.20 | 0.45 | n_estimators=765, max_depth=4, learning_rate=0.0796, subsample=0.8402, colsample_bytree=0.7797, reg_alpha=0.1775, reg_lambda=2.081, min_child_weight=7.289 |
| 12 | 2 | coarse | xgb | 42 | 602837.7948 | 791092.0061 | 0.534420 | 602837.7948 | 791092.0061 | 668726.7688 | 0.25 | 0.30 | n_estimators=577, max_depth=4, learning_rate=0.08528, subsample=0.7895, colsample_bytree=0.9884, reg_alpha=0.4599, reg_lambda=1.772, min_child_weight=6.145 |
| 13 | 15 | coarse | lgbm | 2024 | 603363.5964 | 793923.7698 | 0.532679 | 603363.5964 | 793923.7698 | 670059.6571 | 0.30 | 0.25 | n_estimators=893, learning_rate=0.06594, num_leaves=46, max_depth=9, subsample=0.7404, colsample_bytree=0.8654, reg_alpha=0.2014, reg_lambda=2.415, min_child_samples=34 |
| 14 | 31 | coarse | hgb | 2024 | 606324.5285 | 791349.3053 | 0.534052 | 606324.5285 | 791349.3053 | 671083.2004 | 0.30 | 0.30 | learning_rate=0.1324, max_depth=9, max_iter=283, max_leaf_nodes=114, min_samples_leaf=52, l2_regularization=0.2517 |
| 15 | 19 | coarse | cat | 42 | 606083.6292 | 794747.3853 | 0.535316 | 606083.6292 | 794747.3853 | 672115.9438 | 0.25 | 0.45 | iterations=622, learning_rate=0.09377, depth=7, l2_leaf_reg=4.311, bagging_temperature=0.8374 |
| 16 | 34 | coarse | hgb | 2024 | 607020.4067 | 794387.6700 | 0.523742 | 607020.4067 | 794387.6700 | 672598.9488 | 0.30 | 0.35 | learning_rate=0.1087, max_depth=10, max_iter=285, max_leaf_nodes=53, min_samples_leaf=43, l2_regularization=0.4823 |
| 17 | 28 | coarse | hgb | 42 | 608412.4071 | 792848.0621 | 0.528378 | 608412.4071 | 792848.0621 | 672964.8864 | 0.30 | 0.25 | learning_rate=0.04888, max_depth=8, max_iter=601, max_leaf_nodes=55, min_samples_leaf=28, l2_regularization=0.06464 |
| 18 | 4 | coarse | xgb | 42 | 607868.2812 | 794525.2077 | 0.535374 | 607868.2812 | 794525.2077 | 673198.2055 | 0.25 | 0.25 | n_estimators=300, max_depth=6, learning_rate=0.08632, subsample=0.7039, colsample_bytree=0.8067, reg_alpha=0.1826, reg_lambda=0.8908, min_child_weight=4.961 |
| 19 | 41 | refine | hgb | 2024 | 607298.0140 | 797334.5043 | 0.533861 | 607298.0140 | 797334.5043 | 673810.7856 | 0.25 | 0.20 | learning_rate=0.1204, max_depth=6, max_iter=822, max_leaf_nodes=100, min_samples_leaf=23, l2_regularization=0.9425 |
| 20 | 13 | coarse | lgbm | 42 | 611409.9521 | 797958.9287 | 0.530830 | 611409.9521 | 797958.9287 | 676702.0939 | 0.25 | 0.25 | n_estimators=516, learning_rate=0.06637, num_leaves=25, max_depth=11, subsample=0.8804, colsample_bytree=0.8474, reg_alpha=0.4452, reg_lambda=2.796, min_child_samples=23 |
| 21 | 24 | coarse | hgb | 42 | 611638.1264 | 799036.1657 | 0.529017 | 611638.1264 | 799036.1657 | 677227.4401 | 0.30 | 0.25 | learning_rate=0.03364, max_depth=10, max_iter=415, max_leaf_nodes=90, min_samples_leaf=15, l2_regularization=0.4008 |
| 22 | 42 | refine | hgb | 2024 | 610708.5712 | 801578.7002 | 0.530827 | 610708.5712 | 801578.7002 | 677513.1164 | 0.30 | 0.15 | learning_rate=0.1248, max_depth=7, max_iter=613, max_leaf_nodes=100, min_samples_leaf=23, l2_regularization=0.9425 |
| 23 | 27 | coarse | hgb | 42 | 614435.3281 | 796134.4267 | 0.530204 | 614435.3281 | 796134.4267 | 678030.0126 | 0.30 | 0.20 | learning_rate=0.04204, max_depth=10, max_iter=591, max_leaf_nodes=114, min_samples_leaf=37, l2_regularization=0.6572 |
| 24 | 30 | coarse | hgb | 2024 | 615110.3972 | 795789.8251 | 0.540105 | 615110.3972 | 795789.8251 | 678348.1970 | 0.30 | 0.25 | learning_rate=0.04494, max_depth=5, max_iter=400, max_leaf_nodes=74, min_samples_leaf=38, l2_regularization=1.437 |
| 25 | 9 | coarse | xgb | 2024 | 611974.3809 | 803099.4809 | 0.528974 | 611974.3809 | 803099.4809 | 678868.1659 | 0.25 | 0.20 | n_estimators=611, max_depth=7, learning_rate=0.06163, subsample=0.7005, colsample_bytree=0.9279, reg_alpha=0.2597, reg_lambda=1.153, min_child_weight=4.25 |
| 26 | 7 | coarse | xgb | 2024 | 616305.3639 | 795640.5390 | 0.526530 | 616305.3639 | 795640.5390 | 679072.6752 | 0.30 | 0.35 | n_estimators=294, max_depth=4, learning_rate=0.09008, subsample=0.7684, colsample_bytree=0.8226, reg_alpha=0.29, reg_lambda=0.8778, min_child_weight=5.984 |
| 27 | 18 | coarse | lgbm | 2024 | 615894.8792 | 799561.2157 | 0.529267 | 615894.8792 | 799561.2157 | 680178.0970 | 0.30 | 0.20 | n_estimators=863, learning_rate=0.07106, num_leaves=73, max_depth=11, subsample=0.8065, colsample_bytree=0.7615, reg_alpha=0.1568, reg_lambda=0.7946, min_child_samples=36 |
| 28 | 16 | coarse | lgbm | 2024 | 616020.1839 | 800022.7215 | 0.530371 | 616020.1839 | 800022.7215 | 680421.0720 | 0.25 | 0.25 | n_estimators=500, learning_rate=0.06658, num_leaves=30, max_depth=8, subsample=0.8885, colsample_bytree=0.7625, reg_alpha=0.3279, reg_lambda=1.345, min_child_samples=35 |
| 29 | 21 | coarse | cat | 2024 | 618737.5715 | 800676.1750 | 0.537579 | 618737.5715 | 800676.1750 | 682416.0827 | 0.25 | 0.20 | iterations=389, learning_rate=0.07851, depth=5, l2_leaf_reg=6.022, bagging_temperature=1.263 |
| 30 | 17 | coarse | lgbm | 2024 | 618261.9564 | 805403.4828 | 0.529269 | 618261.9564 | 805403.4828 | 683761.4906 | 0.25 | 0.20 | n_estimators=637, learning_rate=0.02591, num_leaves=88, max_depth=7, subsample=0.7098, colsample_bytree=0.6794, reg_alpha=0.2774, reg_lambda=1.079, min_child_samples=16 |
| 31 | 32 | coarse | hgb | 2024 | 619929.2380 | 808534.0811 | 0.523638 | 619929.2380 | 808534.0811 | 685940.9331 | 0.30 | 0.20 | learning_rate=0.1174, max_depth=13, max_iter=383, max_leaf_nodes=76, min_samples_leaf=45, l2_regularization=1.078 |
| 32 | 10 | coarse | xgb | 2024 | 621250.0391 | 808173.5094 | 0.528424 | 621250.0391 | 808173.5094 | 686673.2537 | 0.25 | 0.15 | n_estimators=393, max_depth=3, learning_rate=0.1105, subsample=0.9969, colsample_bytree=0.6706, reg_alpha=0.1791, reg_lambda=1.96, min_child_weight=2.857 |
| 33 | 23 | coarse | hgb | 42 | 620245.4526 | 812939.8344 | 0.526535 | 620245.4526 | 812939.8344 | 687688.4862 | 0.30 | 0.20 | learning_rate=0.0322, max_depth=4, max_iter=408, max_leaf_nodes=66, min_samples_leaf=58, l2_regularization=0.3248 |
| 34 | 11 | coarse | lgbm | 42 | 623107.5529 | 809326.6910 | 0.520608 | 623107.5529 | 809326.6910 | 688284.2512 | 0.30 | 0.20 | n_estimators=528, learning_rate=0.05041, num_leaves=78, max_depth=5, subsample=0.9875, colsample_bytree=0.9609, reg_alpha=0.3105, reg_lambda=0.9007, min_child_samples=32 |
| 35 | 3 | coarse | xgb | 42 | 622366.6209 | 817510.9288 | 0.517304 | 622366.6209 | 817510.9288 | 690667.1286 | 0.25 | 0.20 | n_estimators=311, max_depth=6, learning_rate=0.1026, subsample=0.8345, colsample_bytree=0.7686, reg_alpha=0.1389, reg_lambda=0.9527, min_child_weight=4.444 |
| 36 | 25 | coarse | hgb | 42 | 624702.9983 | 814301.0894 | 0.511569 | 624702.9983 | 814301.0894 | 691062.3302 | 0.20 | 0.35 | learning_rate=0.1297, max_depth=7, max_iter=393, max_leaf_nodes=35, min_samples_leaf=39, l2_regularization=0.3581 |
| 37 | 5 | coarse | xgb | 42 | 626038.3941 | 816550.1817 | 0.518268 | 626038.3941 | 816550.1817 | 692717.5198 | 0.20 | 0.20 | n_estimators=403, max_depth=6, learning_rate=0.05, subsample=0.7628, colsample_bytree=0.9561, reg_alpha=0.3987, reg_lambda=1.713, min_child_weight=3.088 |
| 38 | 26 | coarse | hgb | 42 | 628386.6712 | 812821.3137 | 0.516081 | 628386.6712 | 812821.3137 | 692938.7961 | 0.30 | 0.15 | learning_rate=0.1002, max_depth=5, max_iter=307, max_leaf_nodes=54, min_samples_leaf=49, l2_regularization=0.6344 |
| 39 | 8 | coarse | xgb | 2024 | 626872.9841 | 820057.6990 | 0.509507 | 626872.9841 | 820057.6990 | 694487.6343 | 0.20 | 0.20 | n_estimators=687, max_depth=6, learning_rate=0.08215, subsample=0.8116, colsample_bytree=0.7971, reg_alpha=0.2474, reg_lambda=1.44, min_child_weight=5.567 |
| 40 | 6 | coarse | xgb | 2024 | 632044.2121 | 822028.2651 | 0.507468 | 632044.2121 | 822028.2651 | 698538.6307 | 0.20 | 0.15 | n_estimators=463, max_depth=9, learning_rate=0.07634, subsample=0.8298, colsample_bytree=0.9652, reg_alpha=0.1597, reg_lambda=1.892, min_child_weight=2.854 |
| 41 | 12 | coarse | lgbm | 42 | 634582.5122 | 823234.0412 | 0.515043 | 634582.5122 | 823234.0412 | 700610.5473 | 0.25 | 0.15 | n_estimators=363, learning_rate=0.01701, num_leaves=114, max_depth=4, subsample=0.7845, colsample_bytree=0.8851, reg_alpha=0.2437, reg_lambda=0.7322, min_child_samples=17 |
| 42 | 14 | coarse | lgbm | 42 | 637128.6618 | 830018.8207 | 0.500058 | 637128.6618 | 830018.8207 | 704640.2174 | 0.20 | 0.15 | n_estimators=403, learning_rate=0.09508, num_leaves=42, max_depth=5, subsample=0.8917, colsample_bytree=0.8809, reg_alpha=0.273, reg_lambda=2.793, min_child_samples=33 |

## Best Model Rationale

Best single trial is #37 (cat) because it has the lowest weighted objective (0.65 * weighted MAE + 0.35 * weighted RMSE), with robust averaged R2 across all folds.

## Ensemble / Blend Formula

Per trial target blend:
- Revenue_hat = a_rev * model_hat + (1 - a_rev) * seasonal_hat
- COGS_hat = a_cogs * model_hat + (1 - a_cogs) * seasonal_hat

Final ensemble across top trials:
- y_final = sum_i (w_i * y_i)

Weights used:
- trial #37 (cat, seed=2024): weight=0.201744, a_rev=0.35, a_cogs=0.45
- trial #38 (cat, seed=2024): weight=0.201489, a_rev=0.25, a_cogs=0.80
- trial #20 (cat, seed=42): weight=0.200604, a_rev=0.30, a_cogs=0.70
- trial #22 (cat, seed=2024): weight=0.200257, a_rev=0.25, a_cogs=0.65
- trial #1 (xgb, seed=42): weight=0.195905, a_rev=0.30, a_cogs=0.35

## Reproducible Commands

```bash
python Data/model_part3_v24_aggressive.py
```
