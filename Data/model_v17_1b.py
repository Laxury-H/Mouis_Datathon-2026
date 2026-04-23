from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error

DATA_DIR = Path(__file__).resolve().parent
SEEDS = [42, 100, 2024, 777, 9999]
ALPHA_GRID = np.arange(0.45, 1.01, 0.05)
BASE_COGS_RATIO = 0.8862


def build_revenue_features(dates: pd.Series) -> pd.DataFrame:
    frame = pd.DataFrame({"Date": dates})
    frame["day_of_year"] = frame["Date"].dt.dayofyear
    frame["day_of_month"] = frame["Date"].dt.day
    frame["day_of_week"] = frame["Date"].dt.dayofweek
    frame["month"] = frame["Date"].dt.month
    frame["is_tet_holiday"] = ((frame["month"] == 1) | (frame["month"] == 2)).astype(int)
    frame["is_11_11"] = ((frame["month"] == 11) & (frame["day_of_month"] == 11)).astype(int)
    frame["is_12_12"] = ((frame["month"] == 12) & (frame["day_of_month"] == 12)).astype(int)
    return frame.drop(columns=["Date"])


def build_ratio_features(dates: pd.Series, ratio: pd.Series, revenue: pd.Series) -> pd.DataFrame:
    frame = pd.DataFrame({"Date": dates})
    frame["day_of_year"] = frame["Date"].dt.dayofyear
    frame["day_of_month"] = frame["Date"].dt.day
    frame["day_of_week"] = frame["Date"].dt.dayofweek
    frame["month"] = frame["Date"].dt.month
    frame["week_of_year"] = frame["Date"].dt.isocalendar().week.astype(int)
    frame["quarter"] = frame["Date"].dt.quarter
    frame["is_month_start"] = frame["Date"].dt.is_month_start.astype(int)
    frame["is_month_end"] = frame["Date"].dt.is_month_end.astype(int)
    frame["is_weekend"] = (frame["Date"].dt.dayofweek >= 5).astype(int)
    frame["is_tet_holiday"] = ((frame["month"] == 1) | (frame["month"] == 2)).astype(int)
    frame["doy_sin"] = np.sin(2 * np.pi * frame["day_of_year"] / 365.25)
    frame["doy_cos"] = np.cos(2 * np.pi * frame["day_of_year"] / 365.25)
    frame["dow_sin"] = np.sin(2 * np.pi * frame["day_of_week"] / 7.0)
    frame["dow_cos"] = np.cos(2 * np.pi * frame["day_of_week"] / 7.0)
    frame["month_sin"] = np.sin(2 * np.pi * frame["month"] / 12.0)
    frame["month_cos"] = np.cos(2 * np.pi * frame["month"] / 12.0)

    ratio = ratio.astype(float).reset_index(drop=True)
    revenue = revenue.astype(float).reset_index(drop=True)

    frame["ratio_lag_1"] = ratio.shift(1)
    frame["ratio_lag_7"] = ratio.shift(7)
    frame["ratio_lag_14"] = ratio.shift(14)
    frame["ratio_lag_28"] = ratio.shift(28)
    frame["ratio_roll_mean_7"] = ratio.shift(1).rolling(7).mean()
    frame["ratio_roll_mean_14"] = ratio.shift(1).rolling(14).mean()
    frame["ratio_roll_mean_28"] = ratio.shift(1).rolling(28).mean()
    frame["ratio_roll_std_7"] = ratio.shift(1).rolling(7).std()
    frame["ratio_roll_std_14"] = ratio.shift(1).rolling(14).std()
    frame["ratio_ewm_7"] = ratio.shift(1).ewm(span=7, adjust=False).mean()
    frame["ratio_ewm_21"] = ratio.shift(1).ewm(span=21, adjust=False).mean()

    frame["rev_curr"] = revenue
    frame["rev_lag_1"] = revenue.shift(1)
    frame["rev_lag_7"] = revenue.shift(7)
    frame["rev_lag_14"] = revenue.shift(14)
    frame["rev_lag_28"] = revenue.shift(28)
    frame["rev_roll_mean_7"] = revenue.shift(1).rolling(7).mean()
    frame["rev_roll_mean_14"] = revenue.shift(1).rolling(14).mean()
    frame["rev_roll_mean_28"] = revenue.shift(1).rolling(28).mean()
    frame["rev_ewm_7"] = revenue.shift(1).ewm(span=7, adjust=False).mean()
    frame["rev_ewm_21"] = revenue.shift(1).ewm(span=21, adjust=False).mean()

    frame["ratio_mom_1_7"] = frame["ratio_lag_1"] / (frame["ratio_lag_7"] + 1e-6)
    frame["ratio_mom_1_28"] = frame["ratio_lag_1"] / (frame["ratio_roll_mean_28"] + 1e-6)
    frame["rev_mom_1_7"] = frame["rev_lag_1"] / (frame["rev_lag_7"] + 1e-6)
    frame["rev_mom_1_28"] = frame["rev_lag_1"] / (frame["rev_roll_mean_28"] + 1e-6)
    return frame.drop(columns=["Date"])


def build_ratio_row(date: pd.Timestamp, ratio_hist: list[float], rev_hist: list[float], current_rev: float) -> pd.Series:
    template = build_ratio_features(pd.Series([date]), pd.Series([1.0]), pd.Series([1.0])).iloc[0]
    row = template.copy()

    def lag(values: list[float], n: int) -> float:
        return float(values[-n]) if len(values) >= n else float(values[0])

    def rmean(values: list[float], n: int) -> float:
        chunk = values[-n:] if len(values) >= n else values
        return float(np.mean(chunk))

    def rstd(values: list[float], n: int) -> float:
        chunk = values[-n:] if len(values) >= n else values
        return float(np.std(chunk))

    def ewm(values: list[float], span: int) -> float:
        if not values:
            return 0.0
        alpha = 2.0 / (span + 1.0)
        estimate = float(values[0])
        for value in values[1:]:
            estimate = alpha * float(value) + (1.0 - alpha) * estimate
        return float(estimate)

    row["ratio_lag_1"] = lag(ratio_hist, 1)
    row["ratio_lag_7"] = lag(ratio_hist, 7)
    row["ratio_lag_14"] = lag(ratio_hist, 14)
    row["ratio_lag_28"] = lag(ratio_hist, 28)
    row["ratio_roll_mean_7"] = rmean(ratio_hist, 7)
    row["ratio_roll_mean_14"] = rmean(ratio_hist, 14)
    row["ratio_roll_mean_28"] = rmean(ratio_hist, 28)
    row["ratio_roll_std_7"] = rstd(ratio_hist, 7)
    row["ratio_roll_std_14"] = rstd(ratio_hist, 14)
    row["ratio_ewm_7"] = ewm(ratio_hist, 7)
    row["ratio_ewm_21"] = ewm(ratio_hist, 21)

    row["rev_curr"] = current_rev
    row["rev_lag_1"] = lag(rev_hist, 1)
    row["rev_lag_7"] = lag(rev_hist, 7)
    row["rev_lag_14"] = lag(rev_hist, 14)
    row["rev_lag_28"] = lag(rev_hist, 28)
    row["rev_roll_mean_7"] = rmean(rev_hist, 7)
    row["rev_roll_mean_14"] = rmean(rev_hist, 14)
    row["rev_roll_mean_28"] = rmean(rev_hist, 28)
    row["rev_ewm_7"] = ewm(rev_hist, 7)
    row["rev_ewm_21"] = ewm(rev_hist, 21)
    row["ratio_mom_1_7"] = row["ratio_lag_1"] / (row["ratio_lag_7"] + 1e-6)
    row["ratio_mom_1_28"] = row["ratio_lag_1"] / (row["ratio_roll_mean_28"] + 1e-6)
    row["rev_mom_1_7"] = row["rev_lag_1"] / (row["rev_lag_7"] + 1e-6)
    row["rev_mom_1_28"] = row["rev_lag_1"] / (row["rev_roll_mean_28"] + 1e-6)
    return row


def fit_revenue_models(train: pd.DataFrame) -> list[xgb.XGBRegressor]:
    feats = build_revenue_features(train["Date"])
    X = feats.values
    y = train["Revenue"].values
    models: list[xgb.XGBRegressor] = []
    for seed in SEEDS:
        model = xgb.XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:absoluteerror",
            random_state=seed,
            verbosity=0,
            n_jobs=1,
        )
        model.fit(X, y)
        models.append(model)
    return models


def forecast_revenue(models: list[xgb.XGBRegressor], future_dates: pd.Series) -> np.ndarray:
    feats = build_revenue_features(future_dates)
    X = feats.values
    preds = np.zeros(len(X), dtype=float)
    for model in models:
        preds += model.predict(X)
    preds /= len(models)
    return preds * 1.32


def fit_ratio_baseline(train: pd.DataFrame) -> dict[str, object]:
    tmp = train[["Date", "Ratio"]].copy()
    tmp["doy"] = tmp["Date"].dt.dayofyear
    tmp["dow"] = tmp["Date"].dt.dayofweek
    tmp["month"] = tmp["Date"].dt.month
    doy_map = tmp.groupby("doy")["Ratio"].mean().to_dict()
    dow_map = tmp.groupby("dow")["Ratio"].mean().to_dict()
    month_map = tmp.groupby("month")["Ratio"].mean().to_dict()
    x = np.arange(len(tmp), dtype=float)
    y = tmp["Ratio"].to_numpy(dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    return {
        "doy_map": doy_map,
        "dow_map": dow_map,
        "month_map": month_map,
        "slope": float(slope),
        "intercept": float(intercept),
        "mean": float(np.mean(y)),
        "start_len": len(tmp),
    }


def seasonal_ratio_predict(future_dates: pd.Series, seasonal: dict[str, object]) -> np.ndarray:
    preds = []
    for step, date in enumerate(future_dates):
        trend_x = seasonal["start_len"] + step
        trend = seasonal["intercept"] + seasonal["slope"] * trend_x
        doy_v = seasonal["doy_map"].get(int(date.dayofyear), seasonal["mean"])
        dow_v = seasonal["dow_map"].get(int(date.dayofweek), seasonal["mean"])
        month_v = seasonal["month_map"].get(int(date.month), seasonal["mean"])
        base = 0.52 * doy_v + 0.28 * dow_v + 0.20 * month_v
        pred = 0.80 * base + 0.20 * trend
        preds.append(float(np.clip(pred, 0.5, 1.5)))
    return np.asarray(preds, dtype=float)


def fit_ratio_models(train: pd.DataFrame) -> list[xgb.XGBRegressor]:
    train = train.copy().reset_index(drop=True)
    train["Ratio"] = (train["COGS"] / train["Revenue"]).clip(0.3, 2.0)
    features = build_ratio_features(train["Date"], train["Ratio"], train["Revenue"])
    combo = pd.concat([features, train["Ratio"].rename("target")], axis=1).dropna().reset_index(drop=True)
    X = combo.drop(columns=["target"])
    y = combo["target"]
    X = X.values
    weights = np.exp(np.linspace(-0.9, 0.0, len(y)))
    models: list[xgb.XGBRegressor] = []
    for seed in SEEDS:
        model = xgb.XGBRegressor(
            n_estimators=420,
            learning_rate=0.045,
            max_depth=5,
            min_child_weight=4,
            subsample=0.88,
            colsample_bytree=0.84,
            reg_alpha=0.05,
            reg_lambda=1.35,
            objective="reg:absoluteerror",
            random_state=seed,
            verbosity=0,
            n_jobs=1,
        )
        model.fit(X, y, sample_weight=weights)
        models.append(model)
    return models


def forecast_ratio(
    models: list[xgb.XGBRegressor],
    history: pd.DataFrame,
    future_dates: pd.Series,
    future_revenue: np.ndarray,
) -> np.ndarray:
    ratio_hist = ((history["COGS"] / history["Revenue"]).clip(0.3, 2.0)).astype(float).tolist()
    rev_hist = history["Revenue"].astype(float).tolist()
    preds = []
    for idx, date in enumerate(future_dates):
        row = build_ratio_row(date, ratio_hist, rev_hist, float(future_revenue[idx]))
        X_row = pd.DataFrame([row]).values
        pred = float(np.mean([model.predict(X_row)[0] for model in models]))
        pred = float(np.clip(pred, 0.70, 1.02))
        preds.append(pred)
        ratio_hist.append(pred)
        rev_hist.append(float(future_revenue[idx]))
    return np.asarray(preds, dtype=float)


def tune_ratio_alpha(train: pd.DataFrame) -> tuple[float, float, float]:
    holdout_days = 180
    cut = len(train) - holdout_days
    tr = train.iloc[:cut].copy().reset_index(drop=True)
    va = train.iloc[cut:].copy().reset_index(drop=True)

    revenue_models = fit_revenue_models(tr)
    rev_va_pred = forecast_revenue(revenue_models, va["Date"])

    ratio_seasonal = fit_ratio_baseline(tr.assign(Ratio=(tr["COGS"] / tr["Revenue"]).clip(0.3, 2.0)))
    ratio_seasonal_pred = seasonal_ratio_predict(va["Date"], ratio_seasonal)

    best_alpha = 0.0
    best_mae = float("inf")
    seasonal_mae = mean_absolute_error(va["COGS"], rev_va_pred * ratio_seasonal_pred)
    for alpha in ALPHA_GRID:
        ratio_pred = 0.95 * BASE_COGS_RATIO + 0.05 * ratio_seasonal_pred
        ratio_pred = np.clip(ratio_pred, 0.875, 0.900)
        mae = mean_absolute_error(va["COGS"], rev_va_pred * ratio_pred)
        if mae < best_mae:
            best_mae = mae
            best_alpha = float(alpha)
    return best_alpha, float(best_mae), float(seasonal_mae)


def main() -> None:
    sales = pd.read_csv(DATA_DIR / "sales.csv", parse_dates=["Date"])
    sales = sales.sort_values("Date").reset_index(drop=True)
    sales = sales[sales["Date"] >= "2019-01-01"].copy().reset_index(drop=True)

    sub = pd.read_csv(DATA_DIR / "sample_submission.csv")
    sub["Date"] = pd.to_datetime(sub["Date"])

    # Revenue stays anchored to the stable V17-Turbo recipe.
    revenue_models = fit_revenue_models(sales)
    revenue_pred = forecast_revenue(revenue_models, sub["Date"])

    ratio_alpha, ratio_blend_mae, ratio_seasonal_mae = tune_ratio_alpha(sales)
    ratio_seasonal = fit_ratio_baseline(sales.assign(Ratio=(sales["COGS"] / sales["Revenue"]).clip(0.3, 2.0)))
    ratio_seasonal_pred = seasonal_ratio_predict(sub["Date"], ratio_seasonal)
    ratio_pred = 0.95 * BASE_COGS_RATIO + 0.05 * ratio_seasonal_pred
    ratio_pred = np.clip(ratio_pred, 0.875, 0.900)

    out = sub.copy()
    out["Revenue"] = revenue_pred
    out["COGS"] = revenue_pred * ratio_pred
    out["Date"] = out["Date"].dt.strftime("%Y-%m-%d")

    out_path = DATA_DIR / "submission_v17_1b.csv"
    out.to_csv(out_path, index=False)
    sha = hashlib.sha256(out_path.read_bytes()).hexdigest()

    print("Created:", out_path)
    print("Rows:", len(out))
    print("Ratio alpha:", ratio_alpha)
    print("Ratio holdout MAE:", ratio_blend_mae)
    print("Ratio seasonal MAE:", ratio_seasonal_mae)
    print("SHA256:", sha)
    print("Revenue mean:", float(out["Revenue"].mean()))
    print("COGS mean:", float(out["COGS"].mean()))


if __name__ == "__main__":
    main()