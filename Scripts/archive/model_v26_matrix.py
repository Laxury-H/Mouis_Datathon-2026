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
N_SEEDS = 5 # Giảm xuống 5 để chạy nhanh hơn vì có tới 8 components
TARGET_MEAN = 4450000.0

LAGS = (1, 2, 3, 7, 14, 28)
ROLL_WINDOWS = (3, 7, 14, 28)
CATEGORIES = ["Casual", "GenZ", "Outdoor", "Streetwear"]

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

def prep_data():
    print("🔄 Chuẩn bị ma trận dữ liệu (Matrix Prep)...")
    cat_df = pd.read_csv(DATA_DIR / "category_daily_rfm.csv")
    cat_df["order_date"] = pd.to_datetime(cat_df["order_date"])
    
    # Tính AOV cho từng dòng
    cat_df["AOV"] = np.where(cat_df["unique_buyers"] > 0, 
                             cat_df["cat_revenue"] / cat_df["unique_buyers"], 
                             np.nan)
                             
    # Tạo bảng ngày chuẩn
    min_date = cat_df["order_date"].min()
    max_date = cat_df["order_date"].max()
    base_df = pd.DataFrame({"date": pd.date_range(min_date, max_date)})
    
    components = {}
    for c in CATEGORIES:
        sub = cat_df[cat_df["category"] == c][["order_date", "unique_buyers", "AOV"]].copy()
        sub = sub.rename(columns={"order_date": "date", "unique_buyers": f"{c}_buyers", "AOV": f"{c}_AOV"})
        
        base_df = base_df.merge(sub, on="date", how="left")
        
        # Xử lý NaN
        base_df[f"{c}_buyers"] = base_df[f"{c}_buyers"].fillna(0)
        mean_aov = base_df[f"{c}_AOV"].mean()
        base_df[f"{c}_AOV"] = base_df[f"{c}_AOV"].fillna(mean_aov)
        
        components[f"{c}_buyers"] = False # is_aov = False
        components[f"{c}_AOV"] = True     # is_aov = True
        
    return base_df, components

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
    depth = 4 if is_aov else 6 # Hạ depth một chút vì dữ liệu bị chia nhỏ, dễ overfit
    return [
        xgb.XGBRegressor(n_estimators=700, learning_rate=0.02, max_depth=depth, objective="reg:tweedie", random_state=seed, n_jobs=-1, verbosity=0),
        lgb.LGBMRegressor(n_estimators=700, learning_rate=0.02, num_leaves=15 if is_aov else 31, objective="tweedie", random_state=seed, n_jobs=-1, verbose=-1),
        CatBoostRegressor(iterations=700, learning_rate=0.02, depth=depth, loss_function="Tweedie:variance_power=1.35", random_seed=seed, verbose=False, thread_count=-1)
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
    print("🚀 [V26 MATRIX STACK] Chạy Ma trận bóc tách Kép (Category x Components)...")
    df, components_dict = prep_data()
    
    sample = pd.read_csv(DATA_DIR / "sample_submission.csv")
    dates = pd.to_datetime(sample["Date"])
    
    final_preds = {}
    
    # Huấn luyện & Dự báo cho 8 thành phần
    for comp, is_aov in components_dict.items():
        print(f"\\n🎯 Đang xử lý: {comp}")
        X, y, f_cols, full_history = build_component_data(df, comp)
        
        comp_models = []
        for i in range(N_SEEDS):
            models = get_models(SEED + i, is_aov)
            for m in models:
                m.fit(X, y)
                comp_models.append(m)
            
        print(f"  🔮 Dự báo đệ quy cho {comp}...")
        final_preds[comp] = recursive_predict_comp(comp_models, full_history, dates, f_cols)
        print(f"  👉 {comp} Forecast Mean: {final_preds[comp].mean():,.2f}")
        
    print("\\n📊 Kết hợp Ma Trận Revenue...")
    total_raw_revenue = np.zeros(len(dates))
    for c in CATEGORIES:
        cat_rev = final_preds[f"{c}_buyers"] * final_preds[f"{c}_AOV"]
        total_raw_revenue += cat_rev
        print(f"  - {c} Revenue Mean: {cat_rev.mean():,.0f}")
        
    print(f"\\nTotal Raw Revenue Mean = {total_raw_revenue.mean():,.0f}")
    
    # Scale to Target Mean
    scale = TARGET_MEAN / total_raw_revenue.mean()
    print(f"⚖️ Scale factor: {scale:.4f}")
    final_rev = total_raw_revenue * scale
    
    out = sample.copy()
    out["Revenue"] = final_rev
    out["COGS"] = final_rev * 0.8862
    
    out_path = OUTPUT_DIR / "submission_v26_matrix_raw.csv"
    out.to_csv(out_path, index=False)
    print(f"✅ Đã lưu Raw: {out_path.name}")
    
    # Anchor with V18 a22 (The King)
    # Vì V25 a30 là tốt nhất nhì (674k), ta thử mix V26 với V18 theo tỷ lệ a30 (30% V26 + 70% V18)
    ref_path = OUTPUT_DIR / "history" / "submissions" / "v18" / "submission_v18_dl_stack_anchor_a22.csv"
    if not ref_path.exists():
        ref_path = Path("Results/history/submissions/v18/submission_v18_dl_stack_anchor_a22.csv")
        
    if ref_path.exists():
        ref = pd.read_csv(ref_path)
        anchored = out.copy()
        alpha = 0.30 # 30/70 blend
        anchored["Revenue"] = (1-alpha) * ref["Revenue"] + alpha * out["Revenue"]
        anchored["COGS"] = (1-alpha) * ref["COGS"] + alpha * out["COGS"]
        anchored_path = OUTPUT_DIR / "submission_v26_matrix_anchor_30.csv"
        anchored.to_csv(anchored_path, index=False)
        print(f"✅ Đã lưu Anchor 30%: {anchored_path.name}")

if __name__ == "__main__":
    main()
