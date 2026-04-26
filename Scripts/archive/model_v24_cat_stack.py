from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
import warnings

warnings.filterwarnings('ignore')

DATA_DIR = Path("Data")
OUTPUT_DIR = Path("Results/submissions")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 2026
N_SEEDS = 5 # Giữ ở mức 5 seed mỗi category để không mất quá nhiều thời gian
CATEGORIES = ['Casual', 'GenZ', 'Outdoor', 'Streetwear']
TARGET_MEAN = 4450000.0

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

def build_cat_data(df: pd.DataFrame, cat: str):
    cat_df = df[df["category"] == cat].copy().sort_values("order_date").reset_index(drop=True)
    rev = cat_df["cat_revenue"].astype(float)
    
    feats = pd.DataFrame(index=cat_df.index)
    for lag in LAGS: feats[f"lag_{lag}"] = rev.shift(lag)
    for w in ROLL_WINDOWS:
        feats[f"mean_{w}"] = rev.shift(1).rolling(w).mean()
        feats[f"std_{w}"] = rev.shift(1).rolling(w).std()
        
    cal = [calendar_features(d) for d in pd.to_datetime(cat_df["order_date"])]
    feats = pd.concat([feats, pd.DataFrame(cal, index=cat_df.index)], axis=1)
    
    combo = pd.concat([feats, rev.rename("target")], axis=1).dropna().reset_index(drop=True)
    return combo.drop(columns="target"), combo["target"], combo.columns.drop("target").tolist(), rev

def get_models(seed: int):
    return [
        xgb.XGBRegressor(n_estimators=1000, learning_rate=0.03, max_depth=6, objective="reg:tweedie", random_state=seed, n_jobs=-1, verbosity=0),
        lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.03, num_leaves=31, objective="tweedie", random_state=seed, n_jobs=-1, verbose=-1),
        CatBoostRegressor(iterations=1000, learning_rate=0.03, depth=6, loss_function="Tweedie:variance_power=1.35", random_seed=seed, verbose=False, thread_count=-1)
    ]

def recursive_predict_cat(models_list, history, dates, feature_cols):
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
        step_p = [m.predict(X_row)[0] for m in models_list]
        y = max(np.mean(step_p), 0.0)
        preds.append(y)
        h.append(y)
    return np.array(preds)

def main():
    print("🚀 [V24 CATEGORY STACK] Khởi động chiến dịch xé nhỏ dữ liệu...")
    df = pd.read_csv(DATA_DIR / "category_daily_rfm.csv")
    sample = pd.read_csv(DATA_DIR / "sample_submission.csv")
    dates = pd.to_datetime(sample["Date"])
    
    total_preds = np.zeros(len(dates))
    
    for cat in CATEGORIES:
        print(f"\\n🎯 Đang xử lý Category: {cat}")
        X, y, f_cols, full_history = build_cat_data(df, cat)
        
        cat_models = []
        for i in range(N_SEEDS):
            models = get_models(SEED + i)
            for m in models:
                m.fit(X, y)
                cat_models.append(m)
            print(f"  ✅ Seed {i+1} done")
            
        print(f"  🔮 Dự báo đệ quy cho {cat}...")
        cat_preds = recursive_predict_cat(cat_models, full_history, dates, f_cols)
        total_preds += cat_preds
        print(f"  👉 {cat} Mean: {cat_preds.mean():,.0f}")
        
    print(f"\\n📊 Tổng hợp: Raw Mean = {total_preds.mean():,.0f}")
    
    # Scale & Save
    scale = TARGET_MEAN / total_preds.mean()
    print(f"⚖️ Scale factor: {scale:.4f}")
    final_rev = total_preds * scale
    
    out = sample.copy()
    out["Revenue"] = final_rev
    out["COGS"] = final_rev * 0.8862
    
    out_path = OUTPUT_DIR / "submission_v24_cat_stack_raw.csv"
    out.to_csv(out_path, index=False)
    print(f"✅ Đã lưu: {out_path.name}")
    
    # Anchor with best V18 (V18 a22 is the current king)
    ref_path = OUTPUT_DIR / "v18" / "submission_v18_dl_stack_anchor_a22.csv"
    if ref_path.exists():
        ref = pd.read_csv(ref_path)
        anchored = out.copy()
        alpha = 0.50 # 50/50 blend for V24
        anchored["Revenue"] = (1-alpha) * ref["Revenue"] + alpha * out["Revenue"]
        anchored["COGS"] = (1-alpha) * ref["COGS"] + alpha * out["COGS"]
        anchored_path = OUTPUT_DIR / "submission_v24_cat_stack_anchor_50.csv"
        anchored.to_csv(anchored_path, index=False)
        print(f"✅ Đã lưu bản Anchor 50%: {anchored_path.name}")

if __name__ == "__main__":
    main()
