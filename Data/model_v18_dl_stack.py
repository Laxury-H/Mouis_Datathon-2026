from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit

DATA_DIR = Path(__file__).resolve().parent
BEST_BASE_NAME = "submission_v17_5_ratio_doy_g26.csv"
HOLDOUT_DAYS = 180
SEED = 2026
OOF_SPLITS = 5
ANCHOR_ALPHAS = (0.10, 0.12, 0.15, 0.18, 0.22, 0.25, 0.30)

LAGS = (1, 2, 3, 7, 14, 28)
ROLL_WINDOWS = (3, 7, 14)
EMA_SPANS = (3, 7)

PROMO_PRIORS: dict[str, Any] = {
    "doy_active": {},
    "doy_discount": {},
    "doy_stackable": {},
    "doy_channel_diversity": {},
    "global_active": 0.0,
    "global_discount": 0.0,
    "global_stackable": 0.0,
    "global_channel_diversity": 0.0,
}

CUSTOMER_GROWTH: dict[pd.Timestamp, float] = {}
REV_DIFF_OFFSET: float = 0.0
RATIO_LOG_OFFSET: float = 0.0


def resolve_best_base_file() -> Path:
    candidates = [
        DATA_DIR / BEST_BASE_NAME,
        DATA_DIR.parent / "Results" / "submissions" / "v17" / BEST_BASE_NAME,
        DATA_DIR.parent / "Results" / "history" / "submissions" / "v17" / BEST_BASE_NAME,
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Cannot locate base submission file: {BEST_BASE_NAME}")


def init_promo_priors(promotions_path: Path) -> None:
    global PROMO_PRIORS

    if not promotions_path.exists():
        return

    promo = pd.read_csv(promotions_path, engine="python", on_bad_lines="skip")
    required = {"start_date", "end_date", "discount_value", "promo_type", "stackable_flag", "promo_channel"}
    if not required.issubset(promo.columns):
        return

    promo["start_date"] = pd.to_datetime(promo["start_date"], errors="coerce")
    promo["end_date"] = pd.to_datetime(promo["end_date"], errors="coerce")
    promo = promo.dropna(subset=["start_date", "end_date"]).copy()
    if promo.empty:
        return

    rows: list[dict[str, float | str | pd.Timestamp]] = []
    for _, r in promo.iterrows():
        start = pd.Timestamp(r["start_date"])
        end = pd.Timestamp(r["end_date"])
        if end < start:
            continue

        discount_value = float(r.get("discount_value", 0.0) or 0.0)
        promo_type = str(r.get("promo_type", "")).strip().lower()
        discount_pct = discount_value if promo_type == "percentage" else 0.0
        channel = str(r.get("promo_channel", "unknown")).strip().lower()
        stackable = float(r.get("stackable_flag", 0.0) or 0.0)

        for d in pd.date_range(start=start, end=end, freq="D"):
            rows.append(
                {
                    "date": d,
                    "active": 1.0,
                    "discount_pct": discount_pct,
                    "stackable": stackable,
                    "channel": channel,
                }
            )

    if not rows:
        return

    daily = pd.DataFrame(rows)
    agg = daily.groupby("date").agg(
        active=("active", "sum"),
        discount_pct=("discount_pct", "sum"),
        stackable=("stackable", "mean"),
        channel_diversity=("channel", "nunique"),
    )
    agg = agg.reset_index()
    agg["doy"] = agg["date"].dt.dayofyear

    doy_agg = agg.groupby("doy").agg(
        active=("active", "mean"),
        discount_pct=("discount_pct", "mean"),
        stackable=("stackable", "mean"),
        channel_diversity=("channel_diversity", "mean"),
    )

    PROMO_PRIORS = {
        "doy_active": {int(k): float(v) for k, v in doy_agg["active"].to_dict().items()},
        "doy_discount": {int(k): float(v) for k, v in doy_agg["discount_pct"].to_dict().items()},
        "doy_stackable": {int(k): float(v) for k, v in doy_agg["stackable"].to_dict().items()},
        "doy_channel_diversity": {int(k): float(v) for k, v in doy_agg["channel_diversity"].to_dict().items()},
        "global_active": float(agg["active"].mean()),
        "global_discount": float(agg["discount_pct"].mean()),
        "global_stackable": float(agg["stackable"].mean()),
        "global_channel_diversity": float(agg["channel_diversity"].mean()),
    }


def init_customer_features(customers_path: Path) -> None:
    global CUSTOMER_GROWTH

    if not customers_path.exists():
        return

    cust = pd.read_csv(customers_path)
    if "signup_date" not in cust.columns:
        return

    cust["signup_date"] = pd.to_datetime(cust["signup_date"], errors="coerce").dt.normalize()
    cust = cust.dropna(subset=["signup_date"])
    if cust.empty:
        return

    daily_new = cust.groupby("signup_date").size().sort_index()
    rolling_new = daily_new.rolling(7, min_periods=1).mean()
    CUSTOMER_GROWTH = {pd.Timestamp(k): float(v) for k, v in rolling_new.to_dict().items()}


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


def _safe_expm1(x: float, low: float = -8.0, high: float = 18.0) -> float:
    return float(np.expm1(np.clip(x, low, high)))


def _sanitize_matrix(mat: np.ndarray, floor: float = 0.0, cap: float = 8.0e7) -> np.ndarray:
    clean = np.nan_to_num(mat, nan=0.0, posinf=cap, neginf=floor)
    return np.clip(clean, floor, cap)


def calendar_features(date: pd.Timestamp) -> dict[str, float]:
    day_of_month = int(date.day)
    day_of_week = int(date.dayofweek)
    month = int(date.month)
    doy = int(date.dayofyear)

    is_weekend = 1.0 if day_of_week >= 5 else 0.0
    is_mega_1111 = 1.0 if (month == 11 and day_of_month == 11) else 0.0
    is_mega_1212 = 1.0 if (month == 12 and day_of_month == 12) else 0.0
    is_payday = 1.0 if (day_of_month >= 25 or day_of_month <= 3) else 0.0

    promo_active = PROMO_PRIORS["doy_active"].get(doy, PROMO_PRIORS["global_active"])
    promo_discount = PROMO_PRIORS["doy_discount"].get(doy, PROMO_PRIORS["global_discount"])
    promo_stackable = PROMO_PRIORS["doy_stackable"].get(doy, PROMO_PRIORS["global_stackable"])
    promo_channel_diversity = PROMO_PRIORS["doy_channel_diversity"].get(doy, PROMO_PRIORS["global_channel_diversity"])
    lookback_date = pd.Timestamp(date).normalize() - pd.Timedelta(days=7)
    recent_user_growth = CUSTOMER_GROWTH.get(lookback_date, 0.0)

    return {
        "day_of_year": float(doy),
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
        "doy_sin": float(np.sin(2.0 * np.pi * doy / 365.25)),
        "doy_cos": float(np.cos(2.0 * np.pi * doy / 365.25)),
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
        "promo_active_prior": float(promo_active),
        "promo_discount_prior": float(promo_discount),
        "promo_stackable_prior": float(promo_stackable),
        "promo_channel_div_prior": float(promo_channel_diversity),
        "promo_payday_interact": float(promo_active) * is_payday,
        "promo_weekend_interact": float(promo_active) * is_weekend,
        "promo_mega1111_interact": float(promo_discount) * is_mega_1111,
        "promo_mega1212_interact": float(promo_discount) * is_mega_1212,
        "recent_user_growth": float(recent_user_growth),
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
    feats["rev_mom_1_14"] = feats["rev_lag_1"] / (feats["rev_roll_mean_14"] + 1e-6)
    feats["rev_diff_1_7"] = feats["rev_lag_1"] - feats["rev_lag_7"]

    cal = [calendar_features(d) for d in sales["Date"]]
    feats = pd.concat([feats, pd.DataFrame(cal, index=sales.index)], axis=1)

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
    feats["rev_mom_1_14"] = feats["rev_lag_1"] / (feats["rev_roll_mean_14"] + 1e-6)
    feats["rev_diff_1_7"] = feats["rev_lag_1"] - feats["rev_lag_7"]

    cal = [calendar_features(d) for d in sales["Date"]]
    feats = pd.concat([feats, pd.DataFrame(cal, index=sales.index)], axis=1)

    target = rev.diff() + REV_DIFF_OFFSET
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
    feats["ratio_mom_1_7"] = feats["ratio_lag_1"] / (feats["ratio_lag_7"] + 1e-6)
    feats["ratio_mom_1_14"] = feats["ratio_lag_1"] / (feats["ratio_roll_mean_14"] + 1e-6)

    cal = [calendar_features(d) for d in sales["Date"]]
    feats = pd.concat([feats, pd.DataFrame(cal, index=sales.index)], axis=1)

    combo = pd.concat([feats, (np.log(ratio) + RATIO_LOG_OFFSET).rename("target")], axis=1).dropna().reset_index(drop=True)
    feature_cols = [c for c in combo.columns if c != "target"]
    return combo[feature_cols], combo["target"].to_numpy(dtype=float), feature_cols


def build_models() -> dict[str, object]:
    return {
        "xgb": xgb.XGBRegressor(
            n_estimators=850,
            learning_rate=0.025,
            max_depth=6,
            min_child_weight=3,
            subsample=0.86,
            colsample_bytree=0.85,
            reg_alpha=0.08,
            reg_lambda=1.35,
            objective="reg:tweedie",
            tweedie_variance_power=1.35,
            random_state=SEED,
            n_jobs=1,
            verbosity=0,
        ),
        "lgb": lgb.LGBMRegressor(
            n_estimators=1000,
            learning_rate=0.025,
            num_leaves=63,
            min_child_samples=20,
            subsample=0.88,
            colsample_bytree=0.86,
            reg_alpha=0.05,
            reg_lambda=1.20,
            objective="tweedie",
            tweedie_variance_power=1.35,
            random_state=SEED,
            n_jobs=1,
            verbose=-1,
        ),
        "cat": CatBoostRegressor(
            iterations=1000,
            learning_rate=0.025,
            depth=7,
            loss_function="Tweedie:variance_power=1.35",
            random_seed=SEED,
            verbose=False,
        ),
    }


def compute_sample_weights(dates: pd.Series) -> np.ndarray:
    weights = np.ones(len(dates), dtype=float)
    for i, d in enumerate(pd.to_datetime(dates)):
        if (d.month == 11 and d.day == 11) or (d.month == 12 and d.day == 12):
            weights[i] = 3.0
        elif d.day >= 25 or d.day <= 3:
            weights[i] = 1.5
    return weights


def fit_model_family(X: pd.DataFrame, y: np.ndarray, dates: pd.Series | None = None) -> dict[str, object]:
    models = build_models()
    weights = None
    if dates is not None:
        aligned_dates = pd.Series(dates).reset_index(drop=True)
        if len(aligned_dates) != len(y):
            aligned_dates = aligned_dates.iloc[-len(y):].reset_index(drop=True)
        weights = compute_sample_weights(aligned_dates)
    for model in models.values():
        if weights is not None:
            model.fit(X, y, sample_weight=weights)
        else:
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
    row["rev_mom_1_14"] = row["rev_lag_1"] / (row["rev_roll_mean_14"] + 1e-6)
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
    row["ratio_mom_1_7"] = row["ratio_lag_1"] / (row["ratio_lag_7"] + 1e-6)
    row["ratio_mom_1_14"] = row["ratio_lag_1"] / (row["ratio_roll_mean_14"] + 1e-6)
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
            y = max(_safe_expm1(log_pred), 0.0)
            preds.append(y)
            local_hist.append(y)
        all_model_preds.append(np.asarray(preds, dtype=float))
    return _sanitize_matrix(np.column_stack(all_model_preds))


def recursive_predict_revenue_diff(models: dict[str, object], history: pd.DataFrame, dates: pd.Series, feature_cols: list[str]) -> np.ndarray:
    hist = history["Revenue"].astype(float).tolist()
    all_model_preds: list[np.ndarray] = []

    for model in models.values():
        local_hist = list(hist)
        preds = []
        for d in dates:
            x = row_revenue_features(pd.Timestamp(d), local_hist, feature_cols)
            diff_pred = float(model.predict(x)[0])
            diff_pred -= REV_DIFF_OFFSET
            y = max(local_hist[-1] + diff_pred, 0.0)
            preds.append(y)
            local_hist.append(y)
        all_model_preds.append(np.asarray(preds, dtype=float))
    return _sanitize_matrix(np.column_stack(all_model_preds))


def recursive_predict_ratio(models: dict[str, object], history: pd.DataFrame, dates: pd.Series, feature_cols: list[str], revenue_pred: np.ndarray) -> np.ndarray:
    ratio_hist = ((history["COGS"] / history["Revenue"]).clip(0.72, 1.1)).astype(float).tolist()
    rev_hist = history["Revenue"].astype(float).tolist()
    all_model_preds: list[np.ndarray] = []

    for model in models.values():
        local_ratio = list(ratio_hist)
        local_rev = list(rev_hist)
        preds = []
        for i, d in enumerate(dates):
            x = row_ratio_features(pd.Timestamp(d), local_ratio, local_rev, feature_cols)
            log_ratio = float(model.predict(x)[0])
            log_ratio -= RATIO_LOG_OFFSET
            ratio = float(np.exp(log_ratio))
            ratio = float(np.clip(ratio, 0.82, 0.98))
            preds.append(ratio)
            local_ratio.append(ratio)
            local_rev.append(float(revenue_pred[i]))
        all_model_preds.append(np.asarray(preds, dtype=float))
    return _sanitize_matrix(np.column_stack(all_model_preds), floor=0.6, cap=1.2)


def fit_stacker(pred_matrix: np.ndarray, y_true: np.ndarray) -> LinearRegression:
    pred_matrix = _sanitize_matrix(pred_matrix)
    meta = LinearRegression(positive=True)
    meta.fit(pred_matrix, y_true)
    return meta


def build_oof_revenue_matrix(
    train_df: pd.DataFrame,
    frame_builder: Any,
    recursive_predictor: Any,
) -> tuple[np.ndarray, np.ndarray, pd.Series]:
    tscv = TimeSeriesSplit(n_splits=OOF_SPLITS)
    mats: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    dates: list[pd.Series] = []

    for tr_idx, va_idx in tscv.split(train_df):
        fold_tr = train_df.iloc[tr_idx].copy().reset_index(drop=True)
        fold_va = train_df.iloc[va_idx].copy().reset_index(drop=True)

        X_fold, y_fold, f_fold = frame_builder(fold_tr)
        base_models = fit_model_family(X_fold, y_fold, fold_tr["Date"])
        pred_matrix = recursive_predictor(base_models, fold_tr, fold_va["Date"], f_fold)

        mats.append(pred_matrix)
        ys.append(fold_va["Revenue"].to_numpy(dtype=float))
        dates.append(fold_va["Date"])

    return np.vstack(mats), np.concatenate(ys), pd.concat(dates, ignore_index=True)


def build_oof_ratio_matrix(train_df: pd.DataFrame, revenue_oof: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    tscv = TimeSeriesSplit(n_splits=OOF_SPLITS)
    mats: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    cursor = 0

    for tr_idx, va_idx in tscv.split(train_df):
        fold_tr = train_df.iloc[tr_idx].copy().reset_index(drop=True)
        fold_va = train_df.iloc[va_idx].copy().reset_index(drop=True)

        X_fold, y_fold, f_fold = build_ratio_log_frame(fold_tr)
        base_models = fit_model_family(X_fold, y_fold, fold_tr["Date"])

        fold_len = len(fold_va)
        fold_rev = revenue_oof[cursor : cursor + fold_len]
        cursor += fold_len

        pred_matrix = recursive_predict_ratio(base_models, fold_tr, fold_va["Date"], f_fold, fold_rev)
        ratio_true = (fold_va["COGS"] / fold_va["Revenue"]).to_numpy(dtype=float)

        mats.append(pred_matrix)
        ys.append(ratio_true)

    return np.vstack(mats), np.concatenate(ys)


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
    global REV_DIFF_OFFSET, RATIO_LOG_OFFSET

    np.random.seed(SEED)
    init_promo_priors(DATA_DIR / "promotions.csv")
    init_customer_features(DATA_DIR / "customers.csv")
    best_base_file = resolve_best_base_file()

    sales = pd.read_csv(DATA_DIR / "sales.csv", parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)
    rev_diff_min = float(sales["Revenue"].diff().min())
    REV_DIFF_OFFSET = max(1.0, -rev_diff_min + 1.0)
    ratio_log_min = float(np.log((sales["COGS"] / sales["Revenue"]).clip(0.72, 1.1)).min())
    RATIO_LOG_OFFSET = max(1.0, -ratio_log_min + 1.0)
    sample = pd.read_csv(DATA_DIR / "sample_submission.csv")
    sample["Date"] = pd.to_datetime(sample["Date"])

    cut = len(sales) - HOLDOUT_DAYS
    tr = sales.iloc[:cut].copy().reset_index(drop=True)
    va = sales.iloc[cut:].copy().reset_index(drop=True)

    lvl_oof_matrix, lvl_oof_y, lvl_oof_dates = build_oof_revenue_matrix(tr, build_revenue_level_frame, recursive_predict_revenue_level)
    lvl_meta = fit_stacker(lvl_oof_matrix, lvl_oof_y)
    rev_lvl_oof = np.maximum(lvl_meta.predict(lvl_oof_matrix), 0.0)

    diff_oof_matrix, diff_oof_y, _ = build_oof_revenue_matrix(tr, build_revenue_diff_frame, recursive_predict_revenue_diff)
    diff_meta = fit_stacker(diff_oof_matrix, diff_oof_y)
    rev_diff_oof = np.maximum(diff_meta.predict(diff_oof_matrix), 0.0)

    best_beta = 0.0
    best_oof_rev_mae = float("inf")
    rev_oof_blend = rev_lvl_oof.copy()
    for beta in np.arange(0.0, 1.01, 0.05):
        blend = (1.0 - beta) * rev_lvl_oof + beta * rev_diff_oof
        mae = mean_absolute_error(lvl_oof_y, blend)
        if mae < best_oof_rev_mae:
            best_oof_rev_mae = float(mae)
            best_beta = float(beta)
            rev_oof_blend = blend

    ratio_oof_matrix, ratio_oof_true = build_oof_ratio_matrix(tr, rev_oof_blend)
    ratio_meta = fit_stacker(ratio_oof_matrix, ratio_oof_true)

    X_lvl, y_lvl, f_lvl = build_revenue_level_frame(tr)
    lvl_models = fit_model_family(X_lvl, y_lvl, tr["Date"])
    lvl_val_matrix = recursive_predict_revenue_level(lvl_models, tr, va["Date"], f_lvl)
    rev_lvl_val = np.maximum(lvl_meta.predict(lvl_val_matrix), 0.0)

    X_diff, y_diff, f_diff = build_revenue_diff_frame(tr)
    diff_models = fit_model_family(X_diff, y_diff, tr["Date"])
    diff_val_matrix = recursive_predict_revenue_diff(diff_models, tr, va["Date"], f_diff)
    rev_diff_val = np.maximum(diff_meta.predict(diff_val_matrix), 0.0)
    best_rev_val = (1.0 - best_beta) * rev_lvl_val + best_beta * rev_diff_val

    dyn_model, dyn_gamma = tune_and_fit_dynamic_multiplier(
        lvl_oof_dates.reset_index(drop=True),
        rev_oof_blend,
        lvl_oof_y,
    )
    dyn_feat_val = multiplier_features(va["Date"], best_rev_val)
    dyn_mul_val = dyn_model.predict(dyn_feat_val)
    dyn_mul_val = np.clip(1.0 + dyn_gamma * (dyn_mul_val - 1.0), 0.88, 1.15)
    rev_val = best_rev_val * dyn_mul_val

    X_ratio, y_ratio, f_ratio = build_ratio_log_frame(tr)
    ratio_models = fit_model_family(X_ratio, y_ratio, tr["Date"])
    ratio_val_matrix = recursive_predict_ratio(ratio_models, tr, va["Date"], f_ratio, rev_val)
    ratio_val = np.clip(ratio_meta.predict(ratio_val_matrix), 0.82, 0.98)

    cogs_val = rev_val * ratio_val
    print("holdout_revenue_mae", float(mean_absolute_error(va["Revenue"], rev_val)))
    print("holdout_cogs_mae", float(mean_absolute_error(va["COGS"], cogs_val)))
    print("oof_revenue_mae", best_oof_rev_mae)
    print("best_beta_diff_blend", best_beta)
    print("dynamic_multiplier_gamma", dyn_gamma)

    X_lvl_full, y_lvl_full, f_lvl_full = build_revenue_level_frame(sales)
    lvl_models_full = fit_model_family(X_lvl_full, y_lvl_full, sales["Date"])

    X_diff_full, y_diff_full, f_diff_full = build_revenue_diff_frame(sales)
    diff_models_full = fit_model_family(X_diff_full, y_diff_full, sales["Date"])

    X_ratio_full, y_ratio_full, f_ratio_full = build_ratio_log_frame(sales)
    ratio_models_full = fit_model_family(X_ratio_full, y_ratio_full, sales["Date"])

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

    ref = pd.read_csv(best_base_file)
    ref_ratio = ref["COGS"] / ref["Revenue"]

    outputs: dict[str, pd.DataFrame] = {
        "submission_v18_dl_stack_raw.csv": out_raw,
    }

    raw_ratio = out_raw["COGS"] / out_raw["Revenue"]
    for alpha in ANCHOR_ALPHAS:
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
