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
N_SEEDS = 10
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

def build_component_data(df: pd.DataFrame, target_col: str):
    target = df[target_col].astype(float)
    feats = pd.DataFrame(index=df.index)
    
    for lag in LAGS: feats[f"lag_{lag}"] = target.shift(lag)
    for w in ROLL_WINDOWS:
        feats[f"mean_{w}"] = target.shift(1).rolling(w).mean()
        feats[f"std_{w}"] = target.shift(1).rolling(w).std()
        
    cal = [calendar_features(d) for d in pd.to_datetime(df["date"])]
    feats = pd.concat([feats, pd.DataFrame(cal, index=df.index)], axis=1)
    
    combo = pd.concat([feats, target.rename("target")], axis=1).dropna().reset_index(drop=True)
    return combo.drop(columns="target"), combo["target"], combo.columns.drop("target").tolist(), target

def get_models(seed: int, is_aov: bool = False):
    # AOV ít biến động hơn, cần cây nông hơn
    depth = 5 if is_aov else 7
    return [
        xgb.XGBRegressor(n_estimators=1000, learning_rate=0.02, max_depth=depth, objective="reg:tweedie", random_state=seed, n_jobs=-1, verbosity=0),
        lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.02, num_leaves=31 if is_aov else 63, objective="tweedie", random_state=seed, n_jobs=-1, verbose=-1),
        CatBoostRegressor(iterations=1000, learning_rate=0.02, depth=depth, loss_function="Tweedie:variance_power=1.35", random_seed=seed, verbose=False, thread_count=-1)
    ]

def recursive_predict_comp(models_list, history, dates, feature_cols):
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
    print("🚀 [V25 COMPONENT STACK] Bóc tách Orders và AOV...")
    df = pd.read_csv(DATA_DIR / "daily_buyers_aov.csv")
    df = df.sort_values("date").reset_index(drop=True)
    
    # Impute missing AOV with global mean just in case
    df["AOV"] = df["AOV"].fillna(df["AOV"].mean())
    
    sample = pd.read_csv(DATA_DIR / "sample_submission.csv")
    dates = pd.to_datetime(sample["Date"])
    
    components = {"total_orders": False, "AOV": True}
    final_preds = {}
    
    for comp, is_aov in components.items():
        print(f"\\n🎯 Đang xử lý: {comp}")
        X, y, f_cols, full_history = build_component_data(df, comp)
        
        comp_models = []
        for i in range(N_SEEDS):
            models = get_models(SEED + i, is_aov)
            for m in models:
                m.fit(X, y)
                comp_models.append(m)
            print(f"  ✅ Seed {i+1} done")
            
        print(f"  🔮 Dự báo đệ quy cho {comp}...")
        final_preds[comp] = recursive_predict_comp(comp_models, full_history, dates, f_cols)
        print(f"  👉 {comp} Mean Forecast: {final_preds[comp].mean():,.2f}")
        
    print("\\n📊 Kết hợp Revenue = Orders * AOV...")
    raw_revenue = final_preds["total_orders"] * final_preds["AOV"]
    print(f"Raw Revenue Mean = {raw_revenue.mean():,.0f}")
    
    # Scale to Target Mean
    scale = TARGET_MEAN / raw_revenue.mean()
    print(f"⚖️ Scale factor: {scale:.4f}")
    final_rev = raw_revenue * scale
    
    out = sample.copy()
    out["Revenue"] = final_rev
    out["COGS"] = final_rev * 0.8862
    
    out_path = OUTPUT_DIR / "submission_v25_comp_stack_raw.csv"
    out.to_csv(out_path, index=False)
    print(f"✅ Đã lưu Raw: {out_path.name}")
    
    # Anchor with V18 a22 (The King)
    ref_path = OUTPUT_DIR / "history" / "submissions" / "v18" / "submission_v18_dl_stack_anchor_a22.csv"
    if ref_path.exists():
        ref = pd.read_csv(ref_path)
        anchored = out.copy()
        alpha = 0.50 # 50/50 blend
        anchored["Revenue"] = (1-alpha) * ref["Revenue"] + alpha * out["Revenue"]
        anchored["COGS"] = (1-alpha) * ref["COGS"] + alpha * out["COGS"]
        anchored_path = OUTPUT_DIR / "submission_v25_comp_stack_anchor_50.csv"
        anchored.to_csv(anchored_path, index=False)
        print(f"✅ Đã lưu Anchor 50%: {anchored_path.name}")
    else:
        # Fallback anchor path
        ref_path = Path("Results/history/submissions/v18/submission_v18_dl_stack_anchor_a22.csv")
        if ref_path.exists():
            ref = pd.read_csv(ref_path)
            anchored = out.copy()
            alpha = 0.50
            anchored["Revenue"] = (1-alpha) * ref["Revenue"] + alpha * out["Revenue"]
            anchored["COGS"] = (1-alpha) * ref["COGS"] + alpha * out["COGS"]
            anchored_path = OUTPUT_DIR / "submission_v25_comp_stack_anchor_50.csv"
            anchored.to_csv(anchored_path, index=False)
            print(f"✅ Đã lưu Anchor 50%: {anchored_path.name}")

if __name__ == "__main__":
    main()
