import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import KFold
from pathlib import Path
import os
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path("Data")
RESULTS_DIR = Path("Results/submissions")
V18_DIR = Path("Results/history/submissions/v18")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

N_SEEDS = 30
N_SPLITS = 5
N_ESTIMATORS = 2500
LEARNING_RATE = 0.015
MAX_DEPTH = 6
TARGET_MEAN = 4450000

def create_calendar_features(dates):
    df = pd.DataFrame({"date": dates})
    df["date"] = pd.to_datetime(df["date"])
    
    df["dayofweek"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["dayofyear"] = df["date"].dt.dayofyear
    df["weekofyear"] = df["date"].dt.isocalendar().week.astype(int)
    
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    df["is_month_start"] = df["date"].dt.is_month_start.astype(int)
    df["is_month_end"] = df["date"].dt.is_month_end.astype(int)
    
    df["doy_sin"] = np.sin(2 * np.pi * df["dayofyear"] / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * df["dayofyear"] / 365.25)
    df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7.0)
    df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7.0)
    
    df["is_tet"] = ((df["month"] == 1) | (df["month"] == 2)).astype(int)
    df["is_11_11"] = ((df["month"] == 11) & (df["day"] == 11)).astype(int)
    df["is_12_12"] = ((df["month"] == 12) & (df["day"] == 12)).astype(int)
    
    m = df["month"]
    d = df["day"]
    y = df["date"].dt.year
    
    # 6 VŨ KHÍ BÍ MẬT TỪ PROMOTIONS.CSV
    df["is_spring_sale"] = (((m == 3) & (d >= 18)) | ((m == 4) & (d <= 17))).astype(int)
    df["is_midyear_sale"] = (((m == 6) & (d >= 23)) | ((m == 7) & (d <= 22))).astype(int)
    df["is_fall_launch"] = (((m == 8) & (d >= 30)) | (m == 9) | ((m == 10) & (d <= 2))).astype(int)
    df["is_yearend_sale"] = (((m == 11) & (d >= 18)) | (m == 12)).astype(int)
    
    is_odd_year = (y % 2 != 0)
    df["is_urban_blowout"] = (is_odd_year & (((m == 7) & (d >= 30)) | (m == 8) | ((m == 9) & (d <= 2)))).astype(int)
    df["is_rural_special"] = (is_odd_year & (((m == 1) & (d >= 30)) | (m == 2) | ((m == 3) & (d == 1)))).astype(int)
    
    return df.drop(columns=["date"])

def train_and_predict(train_df, test_df, target_col, features, objective):
    X = train_df[features]
    y = train_df[target_col]
    X_test = test_df[features]
    
    preds = np.zeros(len(X_test))
    
    for seed in range(N_SEEDS):
        kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
        seed_preds = np.zeros(len(X_test))
        
        for tr_idx, va_idx in kf.split(X, y):
            X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
            X_va, y_va = X.iloc[va_idx], y.iloc[va_idx]
            
            model = xgb.XGBRegressor(
                n_estimators=N_ESTIMATORS,
                learning_rate=LEARNING_RATE,
                max_depth=MAX_DEPTH,
                subsample=0.8,
                colsample_bytree=0.8,
                objective=objective,
                random_state=seed * 100,
                n_jobs=-1
            )
            
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_va, y_va)],
                verbose=False
            )
            
            seed_preds += model.predict(X_test) / N_SPLITS
            
        preds += seed_preds / N_SEEDS
        print(f"  ✅ Seed {seed+1} done")
        
    return preds

def main():
    print("🚀 [V31 PROMO TITAN] Khởi động Lò phản ứng One-Shot Components...")
    
    df = pd.read_csv(DATA_DIR / "daily_buyers_aov.csv")
    df["date"] = pd.to_datetime(df["date"])
    
    cal_features_train = create_calendar_features(df["date"])
    df = pd.concat([df, cal_features_train], axis=1)
    
    test_dates = pd.date_range(start="2023-01-01", end="2023-12-31")
    test_df = pd.DataFrame({"date": test_dates})
    cal_features_test = create_calendar_features(test_df["date"])
    test_df = pd.concat([test_df, cal_features_test], axis=1)
    
    features = [
        "dayofweek", "month", "day", "dayofyear", "weekofyear",
        "is_weekend", "is_month_start", "is_month_end",
        "doy_sin", "doy_cos", "dow_sin", "dow_cos",
        "is_tet", "is_11_11", "is_12_12",
        "is_spring_sale", "is_midyear_sale", "is_fall_launch", "is_yearend_sale",
        "is_urban_blowout", "is_rural_special"
    ]
    
    print("🔮 Dự báo One-Shot cho Orders...")
    pred_orders = train_and_predict(df, test_df, "total_orders", features, "reg:tweedie")
    
    print("🔮 Dự báo One-Shot cho AOV...")
    pred_aov = train_and_predict(df, test_df, "AOV", features, "reg:tweedie")
    
    print("\n📊 Kết hợp Revenue = Orders * AOV...")
    final_rev = pred_orders * pred_aov
    
    out = pd.DataFrame({
        "Date": test_dates.strftime("%Y-%m-%d"),
        "Revenue": final_rev
    })
    
    # SỬ DỤNG V18 COGS RATIO ĐỂ ĐỒNG BỘ
    print("⚙️ Đồng bộ COGS Ratio từ V18 DL Anchor...")
    v18_path = V18_DIR / "submission_v18_dl_stack_anchor_a22.csv"
    if v18_path.exists():
        v18 = pd.read_csv(v18_path)
        v18_ratio = v18["COGS"] / v18["Revenue"]
        out["COGS"] = out["Revenue"] * v18_ratio
    else:
        print("⚠️ Không tìm thấy V18, dùng static ratio!")
        out["COGS"] = out["Revenue"] * 0.8862
    
    raw_mean = out["Revenue"].mean()
    print(f"Raw Revenue Mean = {raw_mean:,.0f}")
    
    scale_factor = TARGET_MEAN / raw_mean
    print(f"⚖️ Scale factor: {scale_factor:.4f}")
    
    out_scaled = out.copy()
    out_scaled["Revenue"] = out["Revenue"] * scale_factor
    out_scaled["COGS"] = out["COGS"] * scale_factor
    
    out_name = "submission_v31_promo_titan_scaled.csv"
    out_scaled.to_csv(RESULTS_DIR / out_name, index=False)
    print(f"✅ Đã lưu: {out_name}")

if __name__ == "__main__":
    main()
