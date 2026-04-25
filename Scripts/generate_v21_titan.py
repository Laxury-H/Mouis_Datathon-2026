import pandas as pd
import numpy as np
import xgboost as xgb
from pathlib import Path
import hashlib
from datetime import datetime

# --- CONFIGURATION ---
DATA_DIR = Path("Data")
SCRIPTS_DIR = Path("Scripts")
OUTPUT_DIR = Path("Results/submissions")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = list(range(2000, 2020)) # 20-Seed Bootstrapping
N_ESTIMATORS = 3000
LEARNING_RATE = 0.01
MAX_DEPTH = 10
MULTIPLIER = 1.32
COGS_RATIO = 0.8862

def create_titan_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    V21 Titan Core: 7 'Thập tự chinh' Features
    No smoothing, no Fourier, no lags - Just raw time indices.
    """
    df = df.copy()
    dates = pd.to_datetime(df["Date"])
    
    out = pd.DataFrame(index=df.index)
    out["day_of_year"] = dates.dt.dayofyear
    out["day_of_month"] = dates.dt.day
    out["day_of_week"] = dates.dt.dayofweek
    out["month"] = dates.dt.month
    
    # 7 'Thập tự chinh' specific flags
    out["is_tet"] = ((dates.dt.month == 1) | (dates.dt.month == 2)).astype(int)
    out["is_11_11"] = ((dates.dt.month == 11) & (dates.dt.day == 11)).astype(int)
    out["is_12_12"] = ((dates.dt.month == 12) & (dates.dt.day == 12)).astype(int)
    
    return out

def main():
    print(f"🚀 [V21 TITAN] Khởi động Lò phản ứng hạt nhân...")
    print(f"⚙️ Config: Seeds=20, Estimators={N_ESTIMATORS}, Depth={MAX_DEPTH}, LR={LEARNING_RATE}")
    
    # Load Data
    sales = pd.read_csv(DATA_DIR / "sales.csv")
    sample = pd.read_csv(DATA_DIR / "sample_submission.csv")
    
    # Preprocessing
    X_train = create_titan_features(sales)
    y_train = sales["Revenue"].astype(float)
    X_test = create_titan_features(sample)
    
    all_preds = []
    
    print(f"🧠 Bắt đầu huấn luyện 20-seed Ensemble (MAE Objective)...")
    for i, seed in enumerate(SEEDS):
        start_time = datetime.now()
        model = xgb.XGBRegressor(
            n_estimators=N_ESTIMATORS,
            learning_rate=LEARNING_RATE,
            max_depth=MAX_DEPTH,
            objective="reg:absoluteerror",
            tree_method="hist", # Optimized for speed
            random_state=seed,
            n_jobs=-1,
            verbosity=0
        )
        
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        all_preds.append(pred)
        
        elapsed = (datetime.now() - start_time).total_seconds()
        print(f"  ✅ Seed {i+1}/20 ({seed}) hoàn tất trong {elapsed:.1f}s")
    
    # Ensemble Average
    mean_pred = np.mean(all_preds, axis=0)
    
    # Post-processing: Titan Multiplier
    final_revenue = mean_pred * MULTIPLIER
    final_cogs = final_revenue * COGS_RATIO
    
    # Create Submission
    submission = sample.copy()
    submission["Revenue"] = final_revenue
    submission["COGS"] = final_cogs
    
    # Validation & Stats
    mean_val = submission["Revenue"].mean()
    print(f"\n📊 Thống kê V21 Titan:")
    print(f"   Mean Revenue: {mean_val:,.0f} VND")
    print(f"   Max Revenue:  {submission['Revenue'].max():,.0f} VND")
    
    # Save Output
    out_name = "submission_v21_titan_20seed.csv"
    out_path = OUTPUT_DIR / out_name
    submission.to_csv(out_path, index=False)
    
    # Generate SHA256 for traceability
    sha = hashlib.sha256(out_path.read_bytes()).hexdigest()
    print(f"\n🏆 Đã rèn xong siêu phẩm: {out_name}")
    print(f"🔗 SHA256: {sha}")
    print(f"📍 Lưu tại: {out_path}")

if __name__ == "__main__":
    main()
