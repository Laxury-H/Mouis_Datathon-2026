from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error

SEEDS = [42, 100, 2024]
ALPHA_GRID = np.arange(0.2, 1.01, 0.1)


def date_features_for_series(dates: pd.Series) -> pd.DataFrame:
    out = pd.DataFrame(index=dates.index)
    out["day_of_year"] = dates.dt.dayofyear
    out["day_of_month"] = dates.dt.day
    out["day_of_week"] = dates.dt.dayofweek
    out["month"] = dates.dt.month
    out["week_of_year"] = dates.dt.isocalendar().week.astype(int)
    out["quarter"] = dates.dt.quarter
    out["is_month_start"] = dates.dt.is_month_start.astype(int)
    out["is_month_end"] = dates.dt.is_month_end.astype(int)
    out["is_weekend"] = (dates.dt.dayofweek >= 5).astype(int)
    out["is_tet_holiday"] = ((dates.dt.month == 1) | (dates.dt.month == 2)).astype(int)
    out["is_11_11"] = ((dates.dt.month == 11) & (dates.dt.day == 11)).astype(int)
    out["is_12_12"] = ((dates.dt.month == 12) & (dates.dt.day == 12)).astype(int)

    # Cyclical encoding for seasonality.
    out["doy_sin"] = np.sin(2 * np.pi * out["day_of_year"] / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * out["day_of_year"] / 365.25)
    out["dow_sin"] = np.sin(2 * np.pi * out["day_of_week"] / 7.0)
    out["dow_cos"] = np.cos(2 * np.pi * out["day_of_week"] / 7.0)
    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12.0)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12.0)

    return out


def date_features_for_one(date: pd.Timestamp) -> dict[str, float]:
    s = pd.Series([date])
    f = date_features_for_series(s)
    return f.iloc[0].to_dict()


def fit_seasonal_baseline(train: pd.DataFrame, target_col: str) -> dict[str, object]:
    tmp = train[["Date", target_col]].copy()
    tmp["doy"] = tmp["Date"].dt.dayofyear
    tmp["dow"] = tmp["Date"].dt.dayofweek
    tmp["month"] = tmp["Date"].dt.month

    doy_map = tmp.groupby("doy")[target_col].mean().to_dict()
    dow_map = tmp.groupby("dow")[target_col].mean().to_dict()
    month_map = tmp.groupby("month")[target_col].mean().to_dict()

    x = np.arange(len(tmp), dtype=float)
    y = tmp[target_col].to_numpy(dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    mean_level = float(np.mean(y))

    return {
        "doy_map": doy_map,
        "dow_map": dow_map,
        "month_map": month_map,
        "slope": float(slope),
        "intercept": float(intercept),
        "mean": mean_level,
        "start_len": len(tmp),
    }


def seasonal_predict(
    future_dates: pd.Series,
    seasonal_model: dict[str, object],
) -> np.ndarray:
    preds = []
    for step, date in enumerate(future_dates):
        trend_x = seasonal_model["start_len"] + step
        trend = seasonal_model["intercept"] + seasonal_model["slope"] * trend_x

        doy = int(date.dayofyear)
        dow = int(date.dayofweek)
        month = int(date.month)

        doy_v = seasonal_model["doy_map"].get(doy, seasonal_model["mean"])
        dow_v = seasonal_model["dow_map"].get(dow, seasonal_model["mean"])
        month_v = seasonal_model["month_map"].get(month, seasonal_model["mean"])

        pred = 0.50 * doy_v + 0.30 * dow_v + 0.20 * month_v
        pred = 0.75 * pred + 0.25 * trend
        preds.append(max(float(pred), 0.0))

    return np.asarray(preds, dtype=float)


def build_training_matrix(train: pd.DataFrame, target_col: str) -> tuple[pd.DataFrame, pd.Series]:
    df = train[["Date", target_col]].copy().sort_values("Date").reset_index(drop=True)

    feats = date_features_for_series(df["Date"])
    vals = df[target_col]

    feats["lag_1"] = vals.shift(1)
    feats["lag_7"] = vals.shift(7)
    feats["lag_14"] = vals.shift(14)
    feats["lag_28"] = vals.shift(28)

    feats["roll_mean_7"] = vals.shift(1).rolling(7).mean()
    feats["roll_mean_14"] = vals.shift(1).rolling(14).mean()
    feats["roll_mean_28"] = vals.shift(1).rolling(28).mean()
    feats["roll_std_7"] = vals.shift(1).rolling(7).std()
    feats["roll_std_14"] = vals.shift(1).rolling(14).std()
    feats["ewm_7"] = vals.shift(1).ewm(span=7, adjust=False).mean()
    feats["ewm_21"] = vals.shift(1).ewm(span=21, adjust=False).mean()

    combo = pd.concat([feats, vals.rename("target")], axis=1).dropna().reset_index(drop=True)
    X = combo.drop(columns=["target"])
    y = combo["target"]
    return X, y


def train_xgb_ensemble(X: pd.DataFrame, y: pd.Series) -> list[xgb.XGBRegressor]:
    models: list[xgb.XGBRegressor] = []
    for seed in SEEDS:
        model = xgb.XGBRegressor(
            n_estimators=420,
            learning_rate=0.045,
            max_depth=6,
            min_child_weight=4,
            subsample=0.86,
            colsample_bytree=0.86,
            reg_alpha=0.08,
            reg_lambda=1.6,
            objective="reg:absoluteerror",
            random_state=seed,
            n_jobs=1,
            verbosity=0,
        )
        model.fit(X, y)
        models.append(model)
    return models


def _lags_from_history(history: list[float], lag: int) -> float:
    if len(history) >= lag:
        return float(history[-lag])
    return float(history[0])


def _roll_mean(history: list[float], window: int) -> float:
    win = history[-window:] if len(history) >= window else history
    return float(np.mean(win))


def _roll_std(history: list[float], window: int) -> float:
    win = history[-window:] if len(history) >= window else history
    return float(np.std(win))


def recursive_predict(
    models: list[xgb.XGBRegressor],
    history: pd.Series,
    future_dates: pd.Series,
) -> np.ndarray:
    hist = [float(v) for v in history.to_list()]
    preds: list[float] = []

    for d in future_dates:
        row = date_features_for_one(pd.Timestamp(d))

        row["lag_1"] = _lags_from_history(hist, 1)
        row["lag_7"] = _lags_from_history(hist, 7)
        row["lag_14"] = _lags_from_history(hist, 14)
        row["lag_28"] = _lags_from_history(hist, 28)

        row["roll_mean_7"] = _roll_mean(hist, 7)
        row["roll_mean_14"] = _roll_mean(hist, 14)
        row["roll_mean_28"] = _roll_mean(hist, 28)
        row["roll_std_7"] = _roll_std(hist, 7)
        row["roll_std_14"] = _roll_std(hist, 14)
        row["ewm_7"] = _roll_mean(hist[-7:] if len(hist) >= 7 else hist, len(hist[-7:]) if len(hist) >= 7 else len(hist))
        row["ewm_21"] = _roll_mean(hist[-21:] if len(hist) >= 21 else hist, len(hist[-21:]) if len(hist) >= 21 else len(hist))

        X_row = pd.DataFrame([row])
        pred = float(np.mean([m.predict(X_row)[0] for m in models]))
        pred = max(pred, 0.0)

        preds.append(pred)
        hist.append(pred)

    return np.asarray(preds, dtype=float)


def tune_alpha(train: pd.DataFrame, target_col: str) -> tuple[float, float, float]:
    # Tune blend weight on the most recent holdout block.
    holdout_days = 120
    cut = len(train) - holdout_days
    if cut <= 200:
        return 0.75, np.nan, np.nan

    tr = train.iloc[:cut].copy()
    va = train.iloc[cut:].copy()

    seasonal = fit_seasonal_baseline(tr, target_col)
    X_tr, y_tr = build_training_matrix(tr, target_col)
    models = train_xgb_ensemble(X_tr, y_tr)

    pred_model = recursive_predict(models, tr[target_col], va["Date"])
    pred_seasonal = seasonal_predict(va["Date"], seasonal)

    best_alpha = 0.75
    best_mae = float("inf")
    baseline_mae = mean_absolute_error(va[target_col], pred_seasonal)

    for alpha in ALPHA_GRID:
        pred = alpha * pred_model + (1.0 - alpha) * pred_seasonal
        mae = mean_absolute_error(va[target_col], pred)
        if mae < best_mae:
            best_mae = mae
            best_alpha = float(alpha)

    return best_alpha, float(best_mae), float(baseline_mae)


def predict_target(train: pd.DataFrame, future_dates: pd.Series, target_col: str) -> tuple[np.ndarray, dict[str, float]]:
    alpha, blend_mae, seasonal_mae = tune_alpha(train, target_col)

    seasonal = fit_seasonal_baseline(train, target_col)
    X_train, y_train = build_training_matrix(train, target_col)
    models = train_xgb_ensemble(X_train, y_train)

    pred_model = recursive_predict(models, train[target_col], future_dates)
    pred_seasonal = seasonal_predict(future_dates, seasonal)

    pred = alpha * pred_model + (1.0 - alpha) * pred_seasonal
    pred = np.maximum(pred, 0.0)

    info = {
        "alpha": alpha,
        "blend_mae": blend_mae,
        "seasonal_mae": seasonal_mae,
    }
    return pred, info


def main() -> None:
    root = Path(__file__).resolve().parent
    data_dir = root

    sales = pd.read_csv(data_dir / "sales.csv", parse_dates=["Date"])  # type: ignore[arg-type]
    sales = sales.sort_values("Date").reset_index(drop=True)
    sales = sales[sales["Date"] >= "2019-01-01"].copy().reset_index(drop=True)

    sub = pd.read_csv(data_dir / "sample_submission.csv")
    sub["Date"] = pd.to_datetime(sub["Date"])

    rev_pred, rev_info = predict_target(sales, sub["Date"], "Revenue")
    cogs_pred, cogs_info = predict_target(sales, sub["Date"], "COGS")

    out = sub.copy()
    out["Revenue"] = rev_pred
    out["COGS"] = cogs_pred
    out["Date"] = out["Date"].dt.strftime("%Y-%m-%d")

    out_path = data_dir / "submission_v17_1.csv"
    out.to_csv(out_path, index=False)

    file_hash = hashlib.sha256(out_path.read_bytes()).hexdigest()

    print("Created:", out_path)
    print("Rows:", len(out))
    print("Revenue alpha:", rev_info["alpha"], "blend_mae:", rev_info["blend_mae"], "seasonal_mae:", rev_info["seasonal_mae"])
    print("COGS alpha:", cogs_info["alpha"], "blend_mae:", cogs_info["blend_mae"], "seasonal_mae:", cogs_info["seasonal_mae"])
    print("SHA256:", file_hash)
    print("Revenue mean:", float(out["Revenue"].mean()))
    print("COGS mean:", float(out["COGS"].mean()))


if __name__ == "__main__":
    main()
