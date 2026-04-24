from __future__ import annotations

import hashlib
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error

DATA_DIR = Path(__file__).resolve().parent
SEEDS = [42, 100, 2024]
BASE_COGS_RATIO = 0.8862
HOLDOUT_DAYS = 180
RECENT_START = "2019-01-01"


# ----------------------------
# Feature helpers
# ----------------------------

def _calendar_features(dates: pd.Series, idx: np.ndarray) -> pd.DataFrame:
    frame = pd.DataFrame(index=dates.index)
    frame["t"] = idx.astype(float)
    frame["day_of_year"] = dates.dt.dayofyear
    frame["day_of_month"] = dates.dt.day
    frame["day_of_week"] = dates.dt.dayofweek
    frame["month"] = dates.dt.month
    frame["week_of_year"] = dates.dt.isocalendar().week.astype(int)
    frame["quarter"] = dates.dt.quarter
    frame["is_month_start"] = dates.dt.is_month_start.astype(int)
    frame["is_month_end"] = dates.dt.is_month_end.astype(int)
    frame["is_weekend"] = (dates.dt.dayofweek >= 5).astype(int)
    frame["is_tet_holiday"] = ((dates.dt.month == 1) | (dates.dt.month == 2)).astype(int)
    frame["is_11_11"] = ((dates.dt.month == 11) & (dates.dt.day == 11)).astype(int)
    frame["is_12_12"] = ((dates.dt.month == 12) & (dates.dt.day == 12)).astype(int)

    frame["doy_sin"] = np.sin(2 * np.pi * frame["day_of_year"] / 365.25)
    frame["doy_cos"] = np.cos(2 * np.pi * frame["day_of_year"] / 365.25)
    frame["dow_sin"] = np.sin(2 * np.pi * frame["day_of_week"] / 7.0)
    frame["dow_cos"] = np.cos(2 * np.pi * frame["day_of_week"] / 7.0)
    frame["month_sin"] = np.sin(2 * np.pi * frame["month"] / 12.0)
    frame["month_cos"] = np.cos(2 * np.pi * frame["month"] / 12.0)
    return frame


def _lag(values: list[float], lag: int) -> float:
    if len(values) >= lag:
        return float(values[-lag])
    return float(values[0])


def _rolling_mean(values: list[float], window: int) -> float:
    chunk = values[-window:] if len(values) >= window else values
    return float(np.mean(chunk))


def _rolling_std(values: list[float], window: int) -> float:
    chunk = values[-window:] if len(values) >= window else values
    return float(np.std(chunk))


def _ewm(values: list[float], span: int) -> float:
    if not values:
        return 0.0
    alpha = 2.0 / (span + 1.0)
    value = float(values[0])
    for cur in values[1:]:
        value = alpha * float(cur) + (1.0 - alpha) * value
    return float(value)


def fit_affine_calibration(preds: np.ndarray, actual: np.ndarray) -> tuple[float, float]:
    design = np.vstack([preds, np.ones_like(preds)]).T
    scale, bias = np.linalg.lstsq(design, actual, rcond=None)[0]
    return float(scale), float(bias)


def apply_affine_calibration(preds: np.ndarray, scale: float, bias: float) -> np.ndarray:
    return scale * preds + bias


# ----------------------------
# Revenue model
# ----------------------------

def revenue_training_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    df = df[["Date", "Revenue"]].copy().sort_values("Date").reset_index(drop=True)
    revenue = df["Revenue"].astype(float)
    idx = np.arange(len(df), dtype=float)
    feats = _calendar_features(df["Date"], idx)

    for lag in (1, 7, 14, 28, 56, 91, 182):
        feats[f"lag_{lag}"] = revenue.shift(lag)
    for window in (7, 14, 28, 56, 91):
        shifted = revenue.shift(1)
        feats[f"roll_mean_{window}"] = shifted.rolling(window).mean()
        feats[f"roll_std_{window}"] = shifted.rolling(window).std()
    for span in (7, 21, 63):
        feats[f"ewm_{span}"] = revenue.shift(1).ewm(span=span, adjust=False).mean()

    feats["diff_1_7"] = feats["lag_1"] - feats["lag_7"]
    feats["diff_1_28"] = feats["lag_1"] - feats["lag_28"]
    feats["mom_1_7"] = feats["lag_1"] / (feats["lag_7"] + 1e-6)
    feats["mom_1_28"] = feats["lag_1"] / (feats["roll_mean_28"] + 1e-6)
    feats["mom_7_28"] = feats["lag_7"] / (feats["lag_28"] + 1e-6)

    combo = pd.concat([feats, revenue.rename("target")], axis=1).dropna().reset_index(drop=True)
    return combo.drop(columns=["target"]), combo["target"]


def revenue_row(date: pd.Timestamp, idx: int, history: list[float]) -> pd.Series:
    row = _calendar_features(pd.Series([date]), np.asarray([idx], dtype=float)).iloc[0]
    row = row.copy()
    for lag in (1, 7, 14, 28, 56, 91, 182):
        row[f"lag_{lag}"] = _lag(history, lag)
    for window in (7, 14, 28, 56, 91):
        row[f"roll_mean_{window}"] = _rolling_mean(history, window)
        row[f"roll_std_{window}"] = _rolling_std(history, window)
    for span in (7, 21, 63):
        row[f"ewm_{span}"] = _ewm(history, span)
    row["diff_1_7"] = row["lag_1"] - row["lag_7"]
    row["diff_1_28"] = row["lag_1"] - row["lag_28"]
    row["mom_1_7"] = row["lag_1"] / (row["lag_7"] + 1e-6)
    row["mom_1_28"] = row["lag_1"] / (row["roll_mean_28"] + 1e-6)
    row["mom_7_28"] = row["lag_7"] / (row["lag_28"] + 1e-6)
    return row


def fit_revenue_models(df: pd.DataFrame) -> list[xgb.XGBRegressor]:
    X, y = revenue_training_frame(df)
    models: list[xgb.XGBRegressor] = []
    for seed in SEEDS:
        model = xgb.XGBRegressor(
            n_estimators=620,
            learning_rate=0.028,
            max_depth=5,
            min_child_weight=3,
            subsample=0.86,
            colsample_bytree=0.84,
            reg_alpha=0.06,
            reg_lambda=1.25,
            objective="reg:absoluteerror",
            random_state=seed,
            verbosity=0,
            n_jobs=1,
        )
        model.fit(X, y)
        models.append(model)
    return models


def predict_revenue(models: list[xgb.XGBRegressor], history: pd.DataFrame, future_dates: pd.Series) -> np.ndarray:
    hist = history["Revenue"].astype(float).tolist()
    preds: list[float] = []
    start_idx = len(hist)
    for step, date in enumerate(future_dates):
        row = revenue_row(pd.Timestamp(date), start_idx + step, hist)
        pred = float(np.mean([model.predict(pd.DataFrame([row]).values)[0] for model in models]))
        pred = max(pred, 0.0)
        preds.append(pred)
        hist.append(pred)
    return np.asarray(preds, dtype=float)


def revenue_seasonal_baseline(train: pd.DataFrame, future_dates: pd.Series) -> np.ndarray:
    tmp = train[["Date", "Revenue"]].copy()
    tmp["doy"] = tmp["Date"].dt.dayofyear
    tmp["dow"] = tmp["Date"].dt.dayofweek
    tmp["month"] = tmp["Date"].dt.month
    doy_map = tmp.groupby("doy")["Revenue"].mean().to_dict()
    dow_map = tmp.groupby("dow")["Revenue"].mean().to_dict()
    month_map = tmp.groupby("month")["Revenue"].mean().to_dict()
    x = np.arange(len(tmp), dtype=float)
    y = tmp["Revenue"].to_numpy(dtype=float)
    slope, intercept = np.polyfit(x, y, 1)

    out = []
    for step, date in enumerate(future_dates):
        trend = intercept + slope * (len(tmp) + step)
        base = 0.48 * doy_map.get(int(date.dayofyear), float(np.mean(y)))
        base += 0.32 * dow_map.get(int(date.dayofweek), float(np.mean(y)))
        base += 0.20 * month_map.get(int(date.month), float(np.mean(y)))
        pred = 0.80 * base + 0.20 * trend
        out.append(max(float(pred), 0.0))
    return np.asarray(out, dtype=float)


# ----------------------------
# Ratio / COGS model
# ----------------------------

def ratio_training_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    df = df[["Date", "Revenue", "COGS"]].copy().sort_values("Date").reset_index(drop=True)
    ratio = (df["COGS"] / df["Revenue"]).clip(0.35, 2.0)
    target = np.log(ratio)
    revenue = df["Revenue"].astype(float)
    idx = np.arange(len(df), dtype=float)
    feats = _calendar_features(df["Date"], idx)

    for lag in (1, 7, 14, 28, 56, 91, 182):
        feats[f"ratio_lag_{lag}"] = ratio.shift(lag)
    for window in (7, 14, 28, 56, 91):
        shifted = ratio.shift(1)
        feats[f"ratio_roll_mean_{window}"] = shifted.rolling(window).mean()
        feats[f"ratio_roll_std_{window}"] = shifted.rolling(window).std()
    for span in (7, 21, 63):
        feats[f"ratio_ewm_{span}"] = ratio.shift(1).ewm(span=span, adjust=False).mean()

    for lag in (1, 7, 14, 28, 56, 91, 182):
        feats[f"rev_lag_{lag}"] = revenue.shift(lag)
    for window in (7, 14, 28, 56, 91):
        shifted_rev = revenue.shift(1)
        feats[f"rev_roll_mean_{window}"] = shifted_rev.rolling(window).mean()
        feats[f"rev_roll_std_{window}"] = shifted_rev.rolling(window).std()
    for span in (7, 21, 63):
        feats[f"rev_ewm_{span}"] = revenue.shift(1).ewm(span=span, adjust=False).mean()

    feats["rev_curr"] = revenue
    feats["rev_ratio_1"] = feats["rev_lag_1"] / (feats["rev_lag_7"] + 1e-6)
    feats["rev_ratio_2"] = feats["rev_lag_1"] / (feats["rev_roll_mean_28"] + 1e-6)
    feats["ratio_mom_1_7"] = feats["ratio_lag_1"] / (feats["ratio_lag_7"] + 1e-6)
    feats["ratio_mom_1_28"] = feats["ratio_lag_1"] / (feats["ratio_roll_mean_28"] + 1e-6)

    combo = pd.concat([feats, target.rename("target")], axis=1).dropna().reset_index(drop=True)
    return combo.drop(columns=["target"]), combo["target"]


def ratio_row(date: pd.Timestamp, idx: int, ratio_hist: list[float], rev_hist: list[float], current_rev: float) -> pd.Series:
    row = _calendar_features(pd.Series([date]), np.asarray([idx], dtype=float)).iloc[0]
    row = row.copy()
    for lag in (1, 7, 14, 28, 56, 91, 182):
        row[f"ratio_lag_{lag}"] = _lag(ratio_hist, lag)
    for window in (7, 14, 28, 56, 91):
        row[f"ratio_roll_mean_{window}"] = _rolling_mean(ratio_hist, window)
        row[f"ratio_roll_std_{window}"] = _rolling_std(ratio_hist, window)
    for span in (7, 21, 63):
        row[f"ratio_ewm_{span}"] = _ewm(ratio_hist, span)

    for lag in (1, 7, 14, 28, 56, 91, 182):
        row[f"rev_lag_{lag}"] = _lag(rev_hist, lag)
    for window in (7, 14, 28, 56, 91):
        row[f"rev_roll_mean_{window}"] = _rolling_mean(rev_hist, window)
        row[f"rev_roll_std_{window}"] = _rolling_std(rev_hist, window)
    for span in (7, 21, 63):
        row[f"rev_ewm_{span}"] = _ewm(rev_hist, span)

    row["rev_curr"] = current_rev
    row["rev_ratio_1"] = row["rev_lag_1"] / (row["rev_lag_7"] + 1e-6)
    row["rev_ratio_2"] = row["rev_lag_1"] / (row["rev_roll_mean_28"] + 1e-6)
    row["ratio_mom_1_7"] = row["ratio_lag_1"] / (row["ratio_lag_7"] + 1e-6)
    row["ratio_mom_1_28"] = row["ratio_lag_1"] / (row["ratio_roll_mean_28"] + 1e-6)
    return row


def fit_ratio_models(df: pd.DataFrame) -> list[xgb.XGBRegressor]:
    X, y = ratio_training_frame(df)
    models: list[xgb.XGBRegressor] = []
    weights = np.exp(np.linspace(-0.8, 0.0, len(y)))
    for seed in SEEDS:
        model = xgb.XGBRegressor(
            n_estimators=720,
            learning_rate=0.022,
            max_depth=5,
            min_child_weight=3,
            subsample=0.88,
            colsample_bytree=0.82,
            reg_alpha=0.04,
            reg_lambda=1.30,
            objective="reg:absoluteerror",
            random_state=seed,
            verbosity=0,
            n_jobs=1,
        )
        model.fit(X, y, sample_weight=weights)
        models.append(model)
    return models


def predict_ratio(models: list[xgb.XGBRegressor], history: pd.DataFrame, future_dates: pd.Series, future_revenue: np.ndarray) -> np.ndarray:
    ratio_hist = ((history["COGS"] / history["Revenue"]).clip(0.35, 2.0)).astype(float).tolist()
    rev_hist = history["Revenue"].astype(float).tolist()
    preds: list[float] = []
    start_idx = len(history)
    for step, date in enumerate(future_dates):
        row = ratio_row(pd.Timestamp(date), start_idx + step, ratio_hist, rev_hist, float(future_revenue[step]))
        log_ratio_pred = float(np.mean([model.predict(pd.DataFrame([row]).values)[0] for model in models]))
        ratio_pred = float(np.exp(log_ratio_pred))
        ratio_pred = float(np.clip(ratio_pred, 0.72, 1.05))
        preds.append(ratio_pred)
        ratio_hist.append(ratio_pred)
        rev_hist.append(float(future_revenue[step]))
    return np.asarray(preds, dtype=float)


def ratio_seasonal_baseline(train: pd.DataFrame, future_dates: pd.Series) -> np.ndarray:
    ratio = ((train["COGS"] / train["Revenue"]).clip(0.35, 2.0)).astype(float)
    tmp = train[["Date"]].copy()
    tmp["ratio"] = ratio.values
    tmp["doy"] = tmp["Date"].dt.dayofyear
    tmp["dow"] = tmp["Date"].dt.dayofweek
    tmp["month"] = tmp["Date"].dt.month
    doy_map = tmp.groupby("doy")["ratio"].mean().to_dict()
    dow_map = tmp.groupby("dow")["ratio"].mean().to_dict()
    month_map = tmp.groupby("month")["ratio"].mean().to_dict()
    x = np.arange(len(tmp), dtype=float)
    y = tmp["ratio"].to_numpy(dtype=float)
    slope, intercept = np.polyfit(x, y, 1)

    out = []
    for step, date in enumerate(future_dates):
        trend = intercept + slope * (len(tmp) + step)
        base = 0.50 * doy_map.get(int(date.dayofyear), float(np.mean(y)))
        base += 0.30 * dow_map.get(int(date.dayofweek), float(np.mean(y)))
        base += 0.20 * month_map.get(int(date.month), float(np.mean(y)))
        pred = 0.82 * base + 0.18 * trend
        out.append(float(np.clip(pred, 0.75, 1.02)))
    return np.asarray(out, dtype=float)


# ----------------------------
# Blending utilities
# ----------------------------

def _simplex_weights(step: float = 0.1):
    grid = np.arange(0.0, 1.0 + 1e-9, step)
    for a in grid:
        for b in grid:
            if a + b <= 1.0 + 1e-9:
                c = 1.0 - a - b
                yield (float(a), float(b), float(c))


def tune_revenue_blend(train: pd.DataFrame) -> tuple[dict[str, float], tuple[float, float], float]:
    cut = len(train) - HOLDOUT_DAYS
    tr = train.iloc[:cut].copy().reset_index(drop=True)
    va = train.iloc[cut:].copy().reset_index(drop=True)

    full_models = fit_revenue_models(tr)
    recent_models = fit_revenue_models(tr[tr["Date"] >= RECENT_START].copy().reset_index(drop=True))
    season_pred = revenue_seasonal_baseline(tr, va["Date"])
    full_pred = predict_revenue(full_models, tr, va["Date"])
    recent_pred = predict_revenue(recent_models, tr[tr["Date"] >= RECENT_START].copy().reset_index(drop=True), va["Date"])

    best = {"full": 0.0, "recent": 0.0, "seasonal": 1.0}
    best_calibration = (1.0, 0.0)
    best_mae = float("inf")
    for w_full, w_recent, w_seasonal in _simplex_weights(0.1):
        pred = w_full * full_pred + w_recent * recent_pred + w_seasonal * season_pred
        scale, bias = fit_affine_calibration(pred, va["Revenue"].to_numpy(dtype=float))
        calibrated = apply_affine_calibration(pred, scale, bias)
        mae = mean_absolute_error(va["Revenue"], calibrated)
        if mae < best_mae:
            best_mae = mae
            best = {"full": w_full, "recent": w_recent, "seasonal": w_seasonal}
            best_calibration = (scale, bias)
    return best, best_calibration, float(best_mae)


def tune_ratio_blend(
    train: pd.DataFrame,
    revenue_weights: dict[str, float],
    revenue_calibration: tuple[float, float],
) -> tuple[dict[str, float], float]:
    cut = len(train) - HOLDOUT_DAYS
    tr = train.iloc[:cut].copy().reset_index(drop=True)
    va = train.iloc[cut:].copy().reset_index(drop=True)

    full_models = fit_ratio_models(tr)
    recent_models = fit_ratio_models(tr[tr["Date"] >= RECENT_START].copy().reset_index(drop=True))
    season_pred = ratio_seasonal_baseline(tr, va["Date"])

    rev_full = predict_revenue(fit_revenue_models(tr), tr, va["Date"])
    rev_recent = predict_revenue(
        fit_revenue_models(tr[tr["Date"] >= RECENT_START].copy().reset_index(drop=True)),
        tr[tr["Date"] >= RECENT_START].copy().reset_index(drop=True),
        va["Date"],
    )
    rev_season = revenue_seasonal_baseline(tr, va["Date"])
    rev_pred = revenue_weights["full"] * rev_full + revenue_weights["recent"] * rev_recent + revenue_weights["seasonal"] * rev_season
    rev_pred = apply_affine_calibration(rev_pred, revenue_calibration[0], revenue_calibration[1])

    full_pred = predict_ratio(full_models, tr, va["Date"], rev_pred)
    recent_pred = predict_ratio(
        recent_models,
        tr[tr["Date"] >= RECENT_START].copy().reset_index(drop=True),
        va["Date"],
        rev_pred,
    )

    best = {"full": 0.0, "recent": 0.0, "seasonal": 1.0}
    best_mae = float("inf")
    for w_full, w_recent, w_seasonal in _simplex_weights(0.1):
        ratio_pred = w_full * full_pred + w_recent * recent_pred + w_seasonal * season_pred
        ratio_pred = np.clip(ratio_pred, 0.78, 1.02)
        mae = mean_absolute_error(va["COGS"], rev_pred * ratio_pred)
        if mae < best_mae:
            best_mae = mae
            best = {"full": w_full, "recent": w_recent, "seasonal": w_seasonal}
    return best, float(best_mae)


# ----------------------------
# Main
# ----------------------------

def main() -> None:
    sales = pd.read_csv(DATA_DIR / "sales.csv", parse_dates=["Date"])
    sales = sales.sort_values("Date").reset_index(drop=True)

    rev_weights, rev_calibration, rev_holdout_mae = tune_revenue_blend(sales)
    ratio_weights, ratio_holdout_mae = tune_ratio_blend(sales, rev_weights, rev_calibration)

    # Fit final models on full data.
    revenue_full = fit_revenue_models(sales)
    revenue_recent = fit_revenue_models(sales[sales["Date"] >= RECENT_START].copy().reset_index(drop=True))
    ratio_full = fit_ratio_models(sales)
    ratio_recent = fit_ratio_models(sales[sales["Date"] >= RECENT_START].copy().reset_index(drop=True))

    sub = pd.read_csv(DATA_DIR / "sample_submission.csv")
    sub["Date"] = pd.to_datetime(sub["Date"])
    future_dates = sub["Date"]

    revenue_full_pred = predict_revenue(revenue_full, sales, future_dates)
    revenue_recent_pred = predict_revenue(
        revenue_recent,
        sales[sales["Date"] >= RECENT_START].copy().reset_index(drop=True),
        future_dates,
    )
    revenue_season_pred = revenue_seasonal_baseline(sales, future_dates)
    revenue_pred = (
        rev_weights["full"] * revenue_full_pred
        + rev_weights["recent"] * revenue_recent_pred
        + rev_weights["seasonal"] * revenue_season_pred
    )
    revenue_pred = apply_affine_calibration(revenue_pred, rev_calibration[0], rev_calibration[1])
    revenue_pred = np.maximum(revenue_pred, 0.0)

    ratio_season_pred = ratio_seasonal_baseline(sales, future_dates)
    ratio_full_pred = predict_ratio(ratio_full, sales, future_dates, revenue_pred)
    ratio_recent_pred = predict_ratio(
        ratio_recent,
        sales[sales["Date"] >= RECENT_START].copy().reset_index(drop=True),
        future_dates,
        revenue_pred,
    )
    ratio_pred = (
        ratio_weights["full"] * ratio_full_pred
        + ratio_weights["recent"] * ratio_recent_pred
        + ratio_weights["seasonal"] * ratio_season_pred
    )
    ratio_pred = np.clip(ratio_pred, 0.78, 1.02)

    out = sub.copy()
    out["Revenue"] = revenue_pred
    out["COGS"] = revenue_pred * ratio_pred
    out["Date"] = out["Date"].dt.strftime("%Y-%m-%d")

    out_path = DATA_DIR / "submission_v17_2.csv"
    out.to_csv(out_path, index=False)
    sha = hashlib.sha256(out_path.read_bytes()).hexdigest()

    print("Created:", out_path)
    print("Rows:", len(out))
    print("Revenue weights:", rev_weights)
    print("Revenue calibration:", {"scale": rev_calibration[0], "bias": rev_calibration[1]})
    print("Revenue holdout MAE:", rev_holdout_mae)
    print("Ratio weights:", ratio_weights)
    print("Ratio holdout MAE:", ratio_holdout_mae)
    print("SHA256:", sha)
    print("Revenue mean:", float(out["Revenue"].mean()))
    print("COGS mean:", float(out["COGS"].mean()))


if __name__ == "__main__":
    main()
