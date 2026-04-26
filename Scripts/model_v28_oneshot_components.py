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

def create_oneshot_features(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """
    V28 One-Shot Core: Chỉ dùng thời gian, KHÔNG dùng Lags.
    Triệt tiêu hoàn toàn rủi ro sai số tích lũy của mô hình đệ quy.
    """
    dates = pd.to_datetime(df[date_col])
    out = pd.DataFrame(index=df.index)
    
    out["doy"] = dates.dt.dayofyear
    out["dom"] = dates.dt.day
    out["dow"] = dates.dt.dayofweek
    out["month"] = dates.dt.month
    
    out["is_weekend"] = (dates.dt.dayofweek >= 5).astype(int)
    out["is_payday"] = ((dates.dt.day >= 25) | (dates.dt.day <= 3)).astype(int)
    out["is_mega"] = ((dates.dt.month.isin([11, 12])) & (dates.dt.day.isin([11, 12]))).astype(int)
    out["is_tet"] = (dates.dt.month.isin([1, 2])).astype(int)
    
    out["doy_sin"] = np.sin(2.0 * np.pi * out["doy"] / 365.25)
    out["doy_cos"] = np.cos(2.0 * np.pi * out["doy"] / 365.25)
    
    return out

def get_models(seed: int, is_aov: bool = False):
    depth = 5 if is_aov else 6
    return [
        xgb.XGBRegressor(n_estimators=1000, learning_rate=0.015, max_depth=depth, objective="reg:absoluteerror", random_state=seed, n_jobs=-1, verbosity=0),
        lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.015, num_leaves=31 if is_aov else 63, objective="mae", random_state=seed, n_jobs=-1, verbose=-1),
        CatBoostRegressor(iterations=1000, learning_rate=0.015, depth=depth, loss_function="MAE", random_seed=seed, verbose=False, thread_count=-1)
    ]

def main():
    print("🚀 [V28 ONE-SHOT COMPONENTS] Khởi động: Bóc tách Orders x AOV + Phi Đệ Quy...")
    df = pd.read_csv(DATA_DIR / "daily_buyers_aov.csv")
    df = df.sort_values("date").reset_index(drop=True)
    df["AOV"] = df["AOV"].fillna(df["AOV"].mean())
    
    sample = pd.read_csv(DATA_DIR / "sample_submission.csv")
    
    X_train = create_oneshot_features(df, "date")
    X_test = create_oneshot_features(sample, "Date")
    
    components = {"total_orders": False, "AOV": True}
    final_preds = {}
    
    for comp, is_aov in components.items():
        print(f"\\n🎯 Đang xử lý Component: {comp}")
        y_train = df[comp].astype(float)
        
        comp_preds = []
        for i in range(N_SEEDS):
            models = get_models(SEED + i, is_aov)
            for m in models:
                m.fit(X_train, y_train)
                pred = m.predict(X_test)
                # Đảm bảo không có giá trị âm
                comp_preds.append(np.maximum(pred, 0))
            print(f"  ✅ Seed {i+1} done")
            
        final_preds[comp] = np.mean(comp_preds, axis=0)
        print(f"  👉 {comp} Forecast Mean: {final_preds[comp].mean():,.2f}")
        
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
    
    out_path = OUTPUT_DIR / "submission_v28_oneshot_comp_raw.csv"
    out.to_csv(out_path, index=False)
    print(f"✅ Đã lưu Raw: {out_path.name}")
    
    # Anchor with V18 a22 (The King)
    ref_path = OUTPUT_DIR / "history" / "submissions" / "v18" / "submission_v18_dl_stack_anchor_a22.csv"
    if not ref_path.exists():
        ref_path = Path("Results/history/submissions/v18/submission_v18_dl_stack_anchor_a22.csv")
        
    if ref_path.exists():
        ref = pd.read_csv(ref_path)
        anchored = out.copy()
        alpha = 0.30 # Mix 30% V28 / 70% V18
        anchored["Revenue"] = (1-alpha) * ref["Revenue"] + alpha * out["Revenue"]
        anchored["COGS"] = (1-alpha) * ref["COGS"] + alpha * out["COGS"]
        anchored_path = OUTPUT_DIR / "submission_v28_oneshot_comp_anchor_30.csv"
        anchored.to_csv(anchored_path, index=False)
        print(f"✅ Đã lưu Anchor 30%: {anchored_path.name}")
        
        # Lưu thêm bản 20% và 40%
        for a in [0.20, 0.40]:
            anchored["Revenue"] = (1-a) * ref["Revenue"] + a * out["Revenue"]
            anchored["COGS"] = (1-a) * ref["COGS"] + a * out["COGS"]
            p = OUTPUT_DIR / f"submission_v28_oneshot_comp_anchor_{int(a*100)}.csv"
            anchored.to_csv(p, index=False)

if __name__ == "__main__":
    main()
