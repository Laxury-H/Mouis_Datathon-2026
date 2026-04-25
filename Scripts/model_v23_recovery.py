from __future__ import annotations
import hashlib
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error
import warnings

warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
DATA_DIR = Path("Data")
OUTPUT_DIR = Path("Results/submissions")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 2026
N_SEEDS = 10 
TARGET_MEAN = 4450000.0
BEST_BASE_NAME = "submission_v17_5_ratio_doy_g26.csv"

# --- FEATURES ---
LAGS = (1, 2, 3, 7, 14, 28)
ROLL_WINDOWS = (3, 7, 14, 28)

def calendar_features(date: pd.Timestamp) -> dict[str, float]:
    doy = int(date.dayofyear)
    dom = int(date.day)
    dow = int(date.dayofweek)
    month = int(date.month)
    return {
        "doy": float(doy), "dom": float(dom), "dow": float(dow), "month": float(month),
        "is_weekend": 1.0 if dow >= 5 else 0.0,
        "is_payday": 1.0 if (dom >= 25 or dom <= 3) else 0.0,
        "is_mega": 1.0 if (month in [11, 12] and dom in [11, 12]) else 0.0,
        "doy_sin": float(np.sin(2.0 * np.pi * doy / 365.25)),
        "doy_cos": float(np.cos(2.0 * np.pi * doy / 365.25)),
    }

def build_training_data(sales: pd.DataFrame):
    rev = sales["Revenue"].astype(float)
    feats = pd.DataFrame(index=sales.index)
    for lag in LAGS: feats[f"lag_{lag}"] = rev.shift(lag)
    for w in ROLL_WINDOWS:
        feats[f"mean_{w}"] = rev.shift(1).rolling(w).mean()
        feats[f"std_{w}"] = rev.shift(1).rolling(w).std()
    
    cal = [calendar_features(d) for d in sales["Date"]]
    feats = pd.concat([feats, pd.DataFrame(cal, index=sales.index)], axis=1)
    df = pd.concat([feats, rev.rename("target")], axis=1).dropna().reset_index(drop=True)
    return df.drop(columns="target"), df["target"], df.columns.drop("target").tolist()

def get_models(seed: int):
    return {
        "xgb": xgb.XGBRegressor(n_estimators=1500, learning_rate=0.02, max_depth=7, objective="reg:tweedie", random_state=seed, n_jobs=-1),
        "lgb": lgb.LGBMRegressor(n_estimators=1500, learning_rate=0.02, num_leaves=63, objective="tweedie", random_state=seed, n_jobs=-1, verbose=-1),
        "cat": CatBoostRegressor(iterations=1500, learning_rate=0.02, depth=7, loss_function="Tweedie:variance_power=1.35", random_seed=seed, verbose=False)
    }

def recursive_predict(models_list, history, dates, feature_cols):
    h = history.astype(float).tolist()
    preds = []
    for d in dates:
        row = {}
        for lag in LAGS: row[f"lag_{lag}"] = h[-lag]
        for w in ROLL_WINDOWS:
            win = h[-w:]
            row[f"mean_{w}"] = np.mean(win)
            row[f"std_{w}"] = np.std(win)
        row.update(calendar_features(pd.Timestamp(d)))
        
        X_row = pd.DataFrame([row])[feature_cols]
        step_p = []
        for m in models_list: step_p.append(m.predict(X_row)[0])
        y = max(np.mean(step_p), 0.0)
        preds.append(y)
        h.append(y)
    return np.array(preds)

def main():
    print("🚀 [V23 RECOVERY] Rèn lại bản Stack chuẩn...")
    sales = pd.read_csv(DATA_DIR / "sales.csv", parse_dates=["Date"]).sort_values("Date")
    X, y, f_cols = build_training_data(sales)
    
    all_trained = []
    for i in range(N_SEEDS):
        models = get_models(SEED + i)
        for m in models.values():
            m.fit(X, y)
            all_trained.append(m)
        print(f"  ✅ Seed {i+1} done")
        
    sample = pd.read_csv(DATA_DIR / "sample_submission.csv")
    preds = recursive_predict(all_trained, sales["Revenue"], pd.to_datetime(sample["Date"]), f_cols)
    
    # Scale
    scale = TARGET_MEAN / preds.mean()
    print(f"⚖️ Scale factor: {scale:.4f}")
    final_rev = preds * scale
    
    out = sample.copy()
    out["Revenue"] = final_rev
    out["COGS"] = final_rev * 0.8862
    
    out.to_csv(OUTPUT_DIR / "submission_v23_recovery_pure.csv", index=False)
    
    # Anchor a25
    ref_path = Path("Results/history/submissions/v17") / BEST_BASE_NAME
    if ref_path.exists():
        ref = pd.read_csv(ref_path)
        anchored = out.copy()
        anchored["Revenue"] = 0.75 * ref["Revenue"] + 0.25 * out["Revenue"]
        anchored["COGS"] = anchored["Revenue"] * (0.75*(ref["COGS"]/ref["Revenue"]) + 0.25*0.8862)
        anchored.to_csv(OUTPUT_DIR / "submission_v23_recovery_anchor_a25.csv", index=False)
        print("✅ Đã tạo bản Anchor a25.")

if __name__ == "__main__": main()
