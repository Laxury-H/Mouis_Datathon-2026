from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DATA_DIR = Path(__file__).resolve().parent
HOLDOUT_DAYS = 180
BASE_FILE = DATA_DIR / "submission_v17_5_ratio_doy_g26.csv"
SEED = 2026

LAGS = (1, 2, 3, 7, 14, 28, 56)
ROLL_WINDOWS = (7, 14, 28)


def add_calendar_features(df: pd.DataFrame, dates: pd.Series) -> pd.DataFrame:
    out = df.copy()
    out["day_of_year"] = dates.dt.dayofyear
    out["day_of_week"] = dates.dt.dayofweek
    out["day_of_month"] = dates.dt.day
    out["month"] = dates.dt.month
    out["is_weekend"] = (dates.dt.dayofweek >= 5).astype(int)
    out["is_month_start"] = dates.dt.is_month_start.astype(int)
    out["is_month_end"] = dates.dt.is_month_end.astype(int)
    out["doy_sin"] = np.sin(2 * np.pi * out["day_of_year"] / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * out["day_of_year"] / 365.25)
    out["dow_sin"] = np.sin(2 * np.pi * out["day_of_week"] / 7.0)
    out["dow_cos"] = np.cos(2 * np.pi * out["day_of_week"] / 7.0)
    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12.0)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12.0)
    return out


def revenue_training_frame(sales: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    rev = sales["Revenue"].astype(float)
    feats = pd.DataFrame(index=sales.index)
    for lag in LAGS:
        feats[f"rev_lag_{lag}"] = rev.shift(lag)
    for w in ROLL_WINDOWS:
        shifted = rev.shift(1)
        feats[f"rev_roll_mean_{w}"] = shifted.rolling(w).mean()
        feats[f"rev_roll_std_{w}"] = shifted.rolling(w).std()
    feats = add_calendar_features(feats, sales["Date"])
    combo = pd.concat([feats, rev.rename("target")], axis=1).dropna().reset_index(drop=True)
    return combo.drop(columns=["target"]), combo["target"]


def ratio_training_frame(sales: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    rev = sales["Revenue"].astype(float)
    ratio = (sales["COGS"] / sales["Revenue"]).clip(0.72, 1.1).astype(float)
    target = np.log(ratio)
    feats = pd.DataFrame(index=sales.index)
    for lag in LAGS:
        feats[f"ratio_lag_{lag}"] = ratio.shift(lag)
    for w in ROLL_WINDOWS:
        shifted = ratio.shift(1)
        feats[f"ratio_roll_mean_{w}"] = shifted.rolling(w).mean()
        feats[f"ratio_roll_std_{w}"] = shifted.rolling(w).std()
    for lag in (1, 7, 14, 28):
        feats[f"rev_lag_{lag}"] = rev.shift(lag)
    feats["rev_roll_mean_14"] = rev.shift(1).rolling(14).mean()
    feats["rev_roll_mean_28"] = rev.shift(1).rolling(28).mean()
    feats = add_calendar_features(feats, sales["Date"])
    combo = pd.concat([feats, target.rename("target")], axis=1).dropna().reset_index(drop=True)
    return combo.drop(columns=["target"]), combo["target"]


def build_mlp() -> Pipeline:
    mlp = MLPRegressor(
        hidden_layer_sizes=(256, 128, 64),
        activation="relu",
        solver="adam",
        learning_rate_init=0.001,
        max_iter=3000,
        alpha=1e-4,
        early_stopping=True,
        n_iter_no_change=40,
        validation_fraction=0.12,
        random_state=SEED,
    )
    return Pipeline([("scaler", StandardScaler()), ("mlp", mlp)])


def _lag(values: list[float], lag: int) -> float:
    if len(values) >= lag:
        return float(values[-lag])
    return float(values[0])


def _rmean(values: list[float], w: int) -> float:
    chunk = values[-w:] if len(values) >= w else values
    return float(np.mean(chunk))


def _rstd(values: list[float], w: int) -> float:
    chunk = values[-w:] if len(values) >= w else values
    return float(np.std(chunk))


def revenue_row(date: pd.Timestamp, rev_hist: list[float]) -> pd.DataFrame:
    row = {}
    for lag in LAGS:
        row[f"rev_lag_{lag}"] = _lag(rev_hist, lag)
    for w in ROLL_WINDOWS:
        row[f"rev_roll_mean_{w}"] = _rmean(rev_hist, w)
        row[f"rev_roll_std_{w}"] = _rstd(rev_hist, w)
    row_df = pd.DataFrame([row])
    return add_calendar_features(row_df, pd.Series([date]))


def ratio_row(date: pd.Timestamp, ratio_hist: list[float], rev_hist: list[float]) -> pd.DataFrame:
    row = {}
    for lag in LAGS:
        row[f"ratio_lag_{lag}"] = _lag(ratio_hist, lag)
    for w in ROLL_WINDOWS:
        row[f"ratio_roll_mean_{w}"] = _rmean(ratio_hist, w)
        row[f"ratio_roll_std_{w}"] = _rstd(ratio_hist, w)
    for lag in (1, 7, 14, 28):
        row[f"rev_lag_{lag}"] = _lag(rev_hist, lag)
    row["rev_roll_mean_14"] = _rmean(rev_hist, 14)
    row["rev_roll_mean_28"] = _rmean(rev_hist, 28)
    row_df = pd.DataFrame([row])
    return add_calendar_features(row_df, pd.Series([date]))


def recursive_forecast(models: tuple[Pipeline, Pipeline], history: pd.DataFrame, future_dates: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    rev_model, ratio_model = models
    rev_hist = history["Revenue"].astype(float).tolist()
    ratio_hist = (history["COGS"] / history["Revenue"]).clip(0.72, 1.1).astype(float).tolist()

    rev_out = []
    ratio_out = []
    for d in future_dates:
        rev_x = revenue_row(pd.Timestamp(d), rev_hist)
        rev_pred = float(rev_model.predict(rev_x)[0])
        rev_pred = max(rev_pred, 0.0)
        rev_out.append(rev_pred)
        rev_hist.append(rev_pred)

        ratio_x = ratio_row(pd.Timestamp(d), ratio_hist, rev_hist)
        log_ratio = float(ratio_model.predict(ratio_x)[0])
        ratio_pred = float(np.exp(log_ratio))
        ratio_pred = float(np.clip(ratio_pred, 0.82, 0.98))
        ratio_out.append(ratio_pred)
        ratio_hist.append(ratio_pred)

    return np.asarray(rev_out, dtype=float), np.asarray(ratio_out, dtype=float)


def fit_models(train_df: pd.DataFrame) -> tuple[Pipeline, Pipeline]:
    rev_x, rev_y = revenue_training_frame(train_df)
    ratio_x, ratio_y = ratio_training_frame(train_df)

    rev_model = build_mlp()
    ratio_model = build_mlp()
    rev_model.fit(rev_x, rev_y)
    ratio_model.fit(ratio_x, ratio_y)
    return rev_model, ratio_model


def evaluate_holdout(sales: pd.DataFrame) -> dict[str, float]:
    cut = len(sales) - HOLDOUT_DAYS
    tr = sales.iloc[:cut].copy().reset_index(drop=True)
    va = sales.iloc[cut:].copy().reset_index(drop=True)

    models = fit_models(tr)
    pred_rev, pred_ratio = recursive_forecast(models, tr, va["Date"])
    pred_cogs = pred_rev * pred_ratio

    return {
        "revenue_mae": float(mean_absolute_error(va["Revenue"], pred_rev)),
        "cogs_mae": float(mean_absolute_error(va["COGS"], pred_cogs)),
    }


def build_outputs(sales: pd.DataFrame, sample_sub: pd.DataFrame) -> dict[str, pd.DataFrame]:
    models = fit_models(sales)
    pred_rev, pred_ratio = recursive_forecast(models, sales, sample_sub["Date"])

    raw = sample_sub.copy()
    raw["Revenue"] = pred_rev
    raw["COGS"] = pred_rev * pred_ratio

    base = pd.read_csv(BASE_FILE)
    base_ratio = (base["COGS"] / base["Revenue"]).astype(float)
    raw_ratio = (raw["COGS"] / raw["Revenue"]).astype(float)

    outs: dict[str, pd.DataFrame] = {}
    outs["submission_v17_dl_raw.csv"] = raw

    for a in (0.05, 0.10, 0.15, 0.20):
        out = raw.copy()
        out["Revenue"] = (1.0 - a) * base["Revenue"] + a * raw["Revenue"]
        ratio = (1.0 - a) * base_ratio + a * raw_ratio
        ratio = ratio.clip(0.82, 0.98)
        out["COGS"] = out["Revenue"] * ratio
        outs[f"submission_v17_dl_anchor_a{int(a * 100):02d}.csv"] = out

    return outs


def summarize_drift(file_path: Path, ref: pd.DataFrame) -> dict[str, float]:
    df = pd.read_csv(file_path)
    ratio = df["COGS"] / df["Revenue"]
    ref_ratio = ref["COGS"] / ref["Revenue"]
    return {
        "rev_mean": float(df["Revenue"].mean()),
        "cogs_mean": float(df["COGS"].mean()),
        "ratio_mean": float(ratio.mean()),
        "rev_mae_vs_ref": float(np.mean(np.abs(df["Revenue"] - ref["Revenue"]))),
        "cogs_mae_vs_ref": float(np.mean(np.abs(df["COGS"] - ref["COGS"]))),
        "ratio_mae_vs_ref": float(np.mean(np.abs(ratio - ref_ratio))),
    }


def main() -> None:
    np.random.seed(SEED)

    sales = pd.read_csv(DATA_DIR / "sales.csv", parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)
    sample_sub = pd.read_csv(DATA_DIR / "sample_submission.csv")
    sample_sub["Date"] = pd.to_datetime(sample_sub["Date"])

    holdout = evaluate_holdout(sales)
    print("DL holdout revenue MAE:", holdout["revenue_mae"])
    print("DL holdout cogs MAE:", holdout["cogs_mae"])

    outputs = build_outputs(sales, sample_sub)
    ref = pd.read_csv(BASE_FILE)

    for name, df in outputs.items():
        out = df.copy()
        out["Date"] = pd.to_datetime(out["Date"]).dt.strftime("%Y-%m-%d")
        path = DATA_DIR / name
        out.to_csv(path, index=False)
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        drift = summarize_drift(path, ref)
        print("\ncreated", path.name)
        print("sha256", sha)
        print("rev_mean", drift["rev_mean"], "cogs_mean", drift["cogs_mean"], "ratio_mean", drift["ratio_mean"])
        print("MAE_vs_g26 rev", drift["rev_mae_vs_ref"], "cogs", drift["cogs_mae_vs_ref"], "ratio", drift["ratio_mae_vs_ref"])


if __name__ == "__main__":
    main()
