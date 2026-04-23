from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit

DATA_DIR = Path(__file__).resolve().parent
BEST_BASE_FILE = DATA_DIR / "submission_v17_5_ratio_doy_g26.csv"
HOLDOUT_DAYS = 180
SEED = 2026

LAGS = (1, 2, 3, 7, 14, 21, 28, 56)
ROLL_WINDOWS = (3, 7, 14, 28)
EMA_SPANS = (3, 7, 14)


def _lag(values: list[float], lag: int) -> float:
    if len(values) >= lag:
        return float(values[-lag])
    return float(values[0])


def _rmean(values: list[float], window: int) -> float:
    chunk = values[-window:] if len(values) >= window else values
    return float(np.mean(chunk))


def _rstd(values: list[float], window: int) -> float:
    chunk = values[-window:] if len(values) >= window else values
    return float(np.std(chunk))


def _ema(values: list[float], span: int) -> float:
    if not values:
        return 0.0
    alpha = 2.0 / (span + 1.0)
    out = float(values[0])
    for v in values[1:]:
        out = alpha * float(v) + (1.0 - alpha) * out
    return float(out)


def calendar_features(date: pd.Timestamp) -> dict[str, float]:
    day_of_month = int(date.day)
    day_of_week = int(date.dayofweek)
    month = int(date.month)

    is_weekend = 1.0 if day_of_week >= 5 else 0.0
    is_mega_1111 = 1.0 if (month == 11 and day_of_month == 11) else 0.0
    is_mega_1212 = 1.0 if (month == 12 and day_of_month == 12) else 0.0
    is_payday = 1.0 if (day_of_month >= 25 or day_of_month <= 3) else 0.0

    # Payday trap interactions and non-linear cross features
    return {
        "day_of_year": float(date.dayofyear),
        "day_of_month": float(day_of_month),
        "day_of_week": float(day_of_week),
        "month": float(month),
        "week_of_year": float(date.isocalendar().week),
        "quarter": float(date.quarter),
        "is_weekend": is_weekend,
        "is_month_start": float(date.is_month_start),
        "is_month_end": float(date.is_month_end),
        "is_mega_1111": is_mega_1111,
        "is_mega_1212": is_mega_1212,
        "is_payday": is_payday,
        "doy_sin": float(np.sin(2.0 * np.pi * date.dayofyear / 365.25)),
        "doy_cos": float(np.cos(2.0 * np.pi * date.dayofyear / 365.25)),
        "dow_sin": float(np.sin(2.0 * np.pi * day_of_week / 7.0)),
        "dow_cos": float(np.cos(2.0 * np.pi * day_of_week / 7.0)),
        "month_sin": float(np.sin(2.0 * np.pi * month / 12.0)),
        "month_cos": float(np.cos(2.0 * np.pi * month / 12.0)),
        "dom_x_weekend": float(day_of_month) * is_weekend,
        "dom_x_mega1111": float(day_of_month) * is_mega_1111,
        "dom_x_mega1212": float(day_of_month) * is_mega_1212,
        "payday_x_weekend": is_payday * is_weekend,
        "payday_x_mega1111": is_payday * is_mega_1111,
        "payday_x_mega1212": is_payday * is_mega_1212,
    }


def build_revenue_level_frame(sales: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    rev = sales["Revenue"].astype(float)
    feats = pd.DataFrame(index=sales.index)

    for lag in LAGS:
        feats[f"rev_lag_{lag}"] = rev.shift(lag)

    shifted = rev.shift(1)
    for w in ROLL_WINDOWS:
        feats[f"rev_roll_mean_{w}"] = shifted.rolling(w).mean()
        feats[f"rev_roll_std_{w}"] = shifted.rolling(w).std()

    for span in EMA_SPANS:
        feats[f"rev_ema_{span}"] = shifted.ewm(span=span, adjust=False).mean()

    feats["rev_mom_1_7"] = feats["rev_lag_1"] / (feats["rev_lag_7"] + 1e-6)
    feats["rev_mom_1_28"] = feats["rev_lag_1"] / (feats["rev_roll_mean_28"] + 1e-6)
    feats["rev_diff_1_7"] = feats["rev_lag_1"] - feats["rev_lag_7"]

    cal = [calendar_features(d) for d in sales["Date"]]
    cal_df = pd.DataFrame(cal, index=sales.index)
    feats = pd.concat([feats, cal_df], axis=1)

    combo = pd.concat([feats, np.log1p(rev).rename("target")], axis=1).dropna().reset_index(drop=True)
    feature_cols = [c for c in combo.columns if c != "target"]
    return combo[feature_cols], combo["target"].to_numpy(dtype=float), feature_cols


def build_revenue_diff_frame(sales: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    rev = sales["Revenue"].astype(float)
    feats = pd.DataFrame(index=sales.index)

    for lag in LAGS:
        feats[f"rev_lag_{lag}"] = rev.shift(lag)

    shifted = rev.shift(1)
    for w in ROLL_WINDOWS:
        feats[f"rev_roll_mean_{w}"] = shifted.rolling(w).mean()
        feats[f"rev_roll_std_{w}"] = shifted.rolling(w).std()

    for span in EMA_SPANS:
        feats[f"rev_ema_{span}"] = shifted.ewm(span=span, adjust=False).mean()

    feats["rev_mom_1_7"] = feats["rev_lag_1"] / (feats["rev_lag_7"] + 1e-6)
    feats["rev_mom_1_28"] = feats["rev_lag_1"] / (feats["rev_roll_mean_28"] + 1e-6)
    feats["rev_diff_1_7"] = feats["rev_lag_1"] - feats["rev_lag_7"]

    cal = [calendar_features(d) for d in sales["Date"]]
    cal_df = pd.DataFrame(cal, index=sales.index)
    feats = pd.concat([feats, cal_df], axis=1)

    target = rev.diff()
    combo = pd.concat([feats, target.rename("target")], axis=1).dropna().reset_index(drop=True)
    feature_cols = [c for c in combo.columns if c != "target"]
    return combo[feature_cols], combo["target"].to_numpy(dtype=float), feature_cols


def build_ratio_log_frame(sales: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    rev = sales["Revenue"].astype(float)
    ratio = (sales["COGS"] / sales["Revenue"]).clip(0.72, 1.1).astype(float)

    feats = pd.DataFrame(index=sales.index)
    for lag in LAGS:
        feats[f"ratio_lag_{lag}"] = ratio.shift(lag)
    shifted = ratio.shift(1)
    for w in ROLL_WINDOWS:
        feats[f"ratio_roll_mean_{w}"] = shifted.rolling(w).mean()
        feats[f"ratio_roll_std_{w}"] = shifted.rolling(w).std()
    for span in EMA_SPANS:
        feats[f"ratio_ema_{span}"] = shifted.ewm(span=span, adjust=False).mean()

    for lag in (1, 7, 14, 28):
        feats[f"rev_lag_{lag}"] = rev.shift(lag)
    feats["rev_roll_mean_14"] = rev.shift(1).rolling(14).mean()
    feats["rev_roll_mean_28"] = rev.shift(1).rolling(28).mean()

    feats["ratio_mom_1_7"] = feats["ratio_lag_1"] / (feats["ratio_lag_7"] + 1e-6)
    feats["ratio_mom_1_28"] = feats["ratio_lag_1"] / (feats["ratio_roll_mean_28"] + 1e-6)

    cal = [calendar_features(d) for d in sales["Date"]]
    cal_df = pd.DataFrame(cal, index=sales.index)
    feats = pd.concat([feats, cal_df], axis=1)

    combo = pd.concat([feats, np.log(ratio).rename("target")], axis=1).dropna().reset_index(drop=True)
    feature_cols = [c for c in combo.columns if c != "target"]
    return combo[feature_cols], combo["target"].to_numpy(dtype=float), feature_cols


def build_models() -> dict[str, object]:
    return {
        "xgb": xgb.XGBRegressor(
            n_estimators=700,
            learning_rate=0.03,
            max_depth=6,
            min_child_weight=3,
            subsample=0.86,
            colsample_bytree=0.85,
            reg_alpha=0.08,
            reg_lambda=1.35,
            objective="reg:absoluteerror",
            random_state=SEED,
            n_jobs=1,
            verbosity=0,
        ),
        "lgb": lgb.LGBMRegressor(
            n_estimators=900,
            learning_rate=0.03,
            num_leaves=63,
            min_child_samples=20,
            subsample=0.88,
            colsample_bytree=0.86,
            reg_alpha=0.05,
            reg_lambda=1.20,
            objective="mae",
            random_state=SEED,
            n_jobs=1,
            verbose=-1,
        ),
        "cat": CatBoostRegressor(
            iterations=900,
            learning_rate=0.03,
            depth=7,
            loss_function="MAE",
            random_seed=SEED,
            verbose=False,
        ),
    }


def fit_model_family(X: pd.DataFrame, y: np.ndarray) -> dict[str, object]:
    models = build_models()
    for model in models.values():
        model.fit(X, y)
    return models


def row_revenue_features(date: pd.Timestamp, rev_hist: list[float], feature_cols: list[str]) -> pd.DataFrame:
    row: dict[str, float] = {}
    for lag in LAGS:
        row[f"rev_lag_{lag}"] = _lag(rev_hist, lag)
    for w in ROLL_WINDOWS:
        row[f"rev_roll_mean_{w}"] = _rmean(rev_hist, w)
        row[f"rev_roll_std_{w}"] = _rstd(rev_hist, w)
    for span in EMA_SPANS:
        row[f"rev_ema_{span}"] = _ema(rev_hist, span)
    row["rev_mom_1_7"] = row["rev_lag_1"] / (row["rev_lag_7"] + 1e-6)
    row["rev_mom_1_28"] = row["rev_lag_1"] / (row["rev_roll_mean_28"] + 1e-6)
    row["rev_diff_1_7"] = row["rev_lag_1"] - row["rev_lag_7"]
    row.update(calendar_features(date))
    return pd.DataFrame([[row[c] for c in feature_cols]], columns=feature_cols)


def row_ratio_features(date: pd.Timestamp, ratio_hist: list[float], rev_hist: list[float], feature_cols: list[str]) -> pd.DataFrame:
    row: dict[str, float] = {}
    for lag in LAGS:
        row[f"ratio_lag_{lag}"] = _lag(ratio_hist, lag)
    for w in ROLL_WINDOWS:
        row[f"ratio_roll_mean_{w}"] = _rmean(ratio_hist, w)
        row[f"ratio_roll_std_{w}"] = _rstd(ratio_hist, w)
    for span in EMA_SPANS:
        row[f"ratio_ema_{span}"] = _ema(ratio_hist, span)
    for lag in (1, 7, 14, 28):
        row[f"rev_lag_{lag}"] = _lag(rev_hist, lag)
    row["rev_roll_mean_14"] = _rmean(rev_hist, 14)
    row["rev_roll_mean_28"] = _rmean(rev_hist, 28)
    row["ratio_mom_1_7"] = row["ratio_lag_1"] / (row["ratio_lag_7"] + 1e-6)
    row["ratio_mom_1_28"] = row["ratio_lag_1"] / (row["ratio_roll_mean_28"] + 1e-6)
    row.update(calendar_features(date))
    return pd.DataFrame([[row[c] for c in feature_cols]], columns=feature_cols)


def recursive_predict_revenue_level(models: dict[str, object], history: pd.DataFrame, dates: pd.Series, feature_cols: list[str]) -> np.ndarray:
    hist = history["Revenue"].astype(float).tolist()
    all_model_preds: list[np.ndarray] = []

    for model in models.values():
        local_hist = list(hist)
        preds = []
        for d in dates:
            x = row_revenue_features(pd.Timestamp(d), local_hist, feature_cols)
            log_pred = float(model.predict(x)[0])
            y = max(float(np.expm1(log_pred)), 0.0)
            preds.append(y)
            local_hist.append(y)
        all_model_preds.append(np.asarray(preds, dtype=float))
    return np.column_stack(all_model_preds)


def recursive_predict_revenue_diff(models: dict[str, object], history: pd.DataFrame, dates: pd.Series, feature_cols: list[str]) -> np.ndarray:
    hist = history["Revenue"].astype(float).tolist()
    all_model_preds: list[np.ndarray] = []

    for model in models.values():
        local_hist = list(hist)
        preds = []
        for d in dates:
            x = row_revenue_features(pd.Timestamp(d), local_hist, feature_cols)
            diff_pred = float(model.predict(x)[0])
            y = max(local_hist[-1] + diff_pred, 0.0)
            preds.append(y)
            local_hist.append(y)
        all_model_preds.append(np.asarray(preds, dtype=float))
    return np.column_stack(all_model_preds)


def recursive_predict_ratio(models: dict[str, object], history: pd.DataFrame, dates: pd.Series, feature_cols: list[str], revenue_pred: np.ndarray) -> np.ndarray:
    ratio_hist = ((history["COGS"] / history["Revenue"]).clip(0.72, 1.1)).astype(float).tolist()
    rev_hist = history["Revenue"].astype(float).tolist()
    all_model_preds: list[np.ndarray] = []

    for model in models.values():
        local_ratio = list(ratio_hist)
        local_rev = list(rev_hist)
        preds = []
        for i, d in enumerate(dates):
            local_rev[-1] = local_rev[-1]
            x = row_ratio_features(pd.Timestamp(d), local_ratio, local_rev, feature_cols)
            log_ratio = float(model.predict(x)[0])
            ratio = float(np.exp(log_ratio))
            ratio = float(np.clip(ratio, 0.82, 0.98))
            preds.append(ratio)
            local_ratio.append(ratio)
            local_rev.append(float(revenue_pred[i]))
        all_model_preds.append(np.asarray(preds, dtype=float))
    return np.column_stack(all_model_preds)


def fit_stacker(pred_matrix: np.ndarray, y_true: np.ndarray) -> LinearRegression:
    meta = LinearRegression(positive=True)
    meta.fit(pred_matrix, y_true)
    return meta


def multiplier_features(dates: pd.Series, revenue_pred: np.ndarray) -> pd.DataFrame:
    rows = []
    for d, y in zip(dates, revenue_pred):
        row = calendar_features(pd.Timestamp(d))
        row["rev_pred"] = float(y)
        row["rev_pred_log1p"] = float(np.log1p(max(y, 0.0)))
        rows.append(row)
    return pd.DataFrame(rows)


def tune_and_fit_dynamic_multiplier(va_dates: pd.Series, rev_pred: np.ndarray, rev_true: np.ndarray) -> tuple[object, float]:
    feat = multiplier_features(va_dates, rev_pred)
    raw_mul = np.clip(rev_true / (rev_pred + 1e-6), 0.80, 1.25)

    tscv = TimeSeriesSplit(n_splits=4)
    oof = np.ones(len(feat), dtype=float)
    for tr_idx, te_idx in tscv.split(feat):
        model = xgb.XGBRegressor(
            n_estimators=220,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:absoluteerror",
            random_state=SEED,
            n_jobs=1,
            verbosity=0,
        )
        model.fit(feat.iloc[tr_idx], raw_mul[tr_idx])
        oof[te_idx] = model.predict(feat.iloc[te_idx])

    best_gamma = 0.0
    best_mae = float("inf")
    for gamma in np.arange(0.0, 1.01, 0.05):
        mul = 1.0 + gamma * (oof - 1.0)
        mul = np.clip(mul, 0.88, 1.15)
        mae = mean_absolute_error(rev_true, rev_pred * mul)
        if mae < best_mae:
            best_mae = float(mae)
            best_gamma = float(gamma)

    final_model = xgb.XGBRegressor(
        n_estimators=260,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:absoluteerror",
        random_state=SEED,
        n_jobs=1,
        verbosity=0,
    )
    final_model.fit(feat, raw_mul)
    return final_model, best_gamma


def main() -> None:
    np.random.seed(SEED)

    sales = pd.read_csv(DATA_DIR / "sales.csv", parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)
    sample = pd.read_csv(DATA_DIR / "sample_submission.csv")
    sample["Date"] = pd.to_datetime(sample["Date"])

    cut = len(sales) - HOLDOUT_DAYS
    tr = sales.iloc[:cut].copy().reset_index(drop=True)
    va = sales.iloc[cut:].copy().reset_index(drop=True)

    # Revenue level models (log target)
    X_lvl, y_lvl, f_lvl = build_revenue_level_frame(tr)
    lvl_models = fit_model_family(X_lvl, y_lvl)
    lvl_val_matrix = recursive_predict_revenue_level(lvl_models, tr, va["Date"], f_lvl)
    lvl_meta = fit_stacker(lvl_val_matrix, va["Revenue"].to_numpy(dtype=float))
    rev_lvl_val = np.maximum(lvl_meta.predict(lvl_val_matrix), 0.0)

    # Revenue difference models (diff target)
    X_diff, y_diff, f_diff = build_revenue_diff_frame(tr)
    diff_models = fit_model_family(X_diff, y_diff)
    diff_val_matrix = recursive_predict_revenue_diff(diff_models, tr, va["Date"], f_diff)
    diff_meta = fit_stacker(diff_val_matrix, va["Revenue"].to_numpy(dtype=float))
    rev_diff_val = np.maximum(diff_meta.predict(diff_val_matrix), 0.0)

    # Blend level + diff outputs
    best_beta = 0.0
    best_rev_mae = float("inf")
    best_rev_val = rev_lvl_val.copy()
    for beta in np.arange(0.0, 1.01, 0.05):
        blend = (1.0 - beta) * rev_lvl_val + beta * rev_diff_val
        mae = mean_absolute_error(va["Revenue"], blend)
        if mae < best_rev_mae:
            best_rev_mae = float(mae)
            best_beta = float(beta)
            best_rev_val = blend

    # Dynamic multiplier
    dyn_model, dyn_gamma = tune_and_fit_dynamic_multiplier(
        va["Date"],
        best_rev_val,
        va["Revenue"].to_numpy(dtype=float),
    )
    dyn_feat_val = multiplier_features(va["Date"], best_rev_val)
    dyn_mul_val = dyn_model.predict(dyn_feat_val)
    dyn_mul_val = np.clip(1.0 + dyn_gamma * (dyn_mul_val - 1.0), 0.88, 1.15)
    rev_val = best_rev_val * dyn_mul_val

    # Ratio models
    X_ratio, y_ratio, f_ratio = build_ratio_log_frame(tr)
    ratio_models = fit_model_family(X_ratio, y_ratio)
    ratio_val_matrix = recursive_predict_ratio(ratio_models, tr, va["Date"], f_ratio, rev_val)
    ratio_meta = fit_stacker(ratio_val_matrix, (va["COGS"] / va["Revenue"]).to_numpy(dtype=float))
    ratio_val = np.clip(ratio_meta.predict(ratio_val_matrix), 0.82, 0.98)

    cogs_val = rev_val * ratio_val
    print("holdout_revenue_mae", float(mean_absolute_error(va["Revenue"], rev_val)))
    print("holdout_cogs_mae", float(mean_absolute_error(va["COGS"], cogs_val)))
    print("best_beta_diff_blend", best_beta)
    print("dynamic_multiplier_gamma", dyn_gamma)

    # Refit base models on full data
    X_lvl_full, y_lvl_full, f_lvl_full = build_revenue_level_frame(sales)
    lvl_models_full = fit_model_family(X_lvl_full, y_lvl_full)

    X_diff_full, y_diff_full, f_diff_full = build_revenue_diff_frame(sales)
    diff_models_full = fit_model_family(X_diff_full, y_diff_full)

    X_ratio_full, y_ratio_full, f_ratio_full = build_ratio_log_frame(sales)
    ratio_models_full = fit_model_family(X_ratio_full, y_ratio_full)

    future_dates = sample["Date"]
    lvl_future_matrix = recursive_predict_revenue_level(lvl_models_full, sales, future_dates, f_lvl_full)
    diff_future_matrix = recursive_predict_revenue_diff(diff_models_full, sales, future_dates, f_diff_full)

    rev_lvl_future = np.maximum(lvl_meta.predict(lvl_future_matrix), 0.0)
    rev_diff_future = np.maximum(diff_meta.predict(diff_future_matrix), 0.0)
    rev_future_pre = (1.0 - best_beta) * rev_lvl_future + best_beta * rev_diff_future

    dyn_feat_future = multiplier_features(future_dates, rev_future_pre)
    dyn_mul_future = dyn_model.predict(dyn_feat_future)
    dyn_mul_future = np.clip(1.0 + dyn_gamma * (dyn_mul_future - 1.0), 0.88, 1.15)
    rev_future = np.maximum(rev_future_pre * dyn_mul_future, 0.0)

    ratio_future_matrix = recursive_predict_ratio(ratio_models_full, sales, future_dates, f_ratio_full, rev_future)
    ratio_future = np.clip(ratio_meta.predict(ratio_future_matrix), 0.82, 0.98)

    out_raw = sample.copy()
    out_raw["Revenue"] = rev_future
    out_raw["COGS"] = rev_future * ratio_future

    ref = pd.read_csv(BEST_BASE_FILE)
    ref_ratio = ref["COGS"] / ref["Revenue"]

    outputs: dict[str, pd.DataFrame] = {
        "submission_v18_dl_stack_raw.csv": out_raw,
    }

    # Anchor DL outputs to today's sweet spot
    raw_ratio = out_raw["COGS"] / out_raw["Revenue"]
    for alpha in (0.05, 0.10, 0.15, 0.20):
        out = out_raw.copy()
        out["Revenue"] = (1.0 - alpha) * ref["Revenue"] + alpha * out_raw["Revenue"]
        ratio = (1.0 - alpha) * ref_ratio + alpha * raw_ratio
        ratio = ratio.clip(0.82, 0.98)
        out["COGS"] = out["Revenue"] * ratio
        outputs[f"submission_v18_dl_stack_anchor_a{int(alpha * 100):02d}.csv"] = out

    for name, df in outputs.items():
        out = df.copy()
        out["Date"] = pd.to_datetime(out["Date"]).dt.strftime("%Y-%m-%d")
        path = DATA_DIR / name
        out.to_csv(path, index=False)

        ratio = out["COGS"] / out["Revenue"]
        mae_rev_ref = float(np.mean(np.abs(out["Revenue"] - ref["Revenue"])))
        mae_cogs_ref = float(np.mean(np.abs(out["COGS"] - ref["COGS"])))
        mae_ratio_ref = float(np.mean(np.abs(ratio - ref_ratio)))

        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        print("\ncreated", path.name)
        print("sha256", sha)
        print("mean_revenue", float(out["Revenue"].mean()), "mean_cogs", float(out["COGS"].mean()), "mean_ratio", float(ratio.mean()))
        print("mae_vs_g26_revenue", mae_rev_ref, "mae_vs_g26_cogs", mae_cogs_ref, "mae_vs_g26_ratio", mae_ratio_ref)


if __name__ == "__main__":
    main()
