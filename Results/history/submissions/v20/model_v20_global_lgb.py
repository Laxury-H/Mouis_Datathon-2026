import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder
from pathlib import Path

DATA_DIR = Path("Data")

def create_global_features(df: pd.DataFrame, drop_na=True) -> pd.DataFrame:
    df = df.copy()
    df['order_date'] = pd.to_datetime(df['order_date'])
    
    # Label Encode Category
    le = LabelEncoder()
    df['category_idx'] = le.fit_transform(df['category'])
    
    # 1. Tạo Lags theo TỪNG DANH MỤC
    # TUYỆT ĐỐI KHÔNG dùng shift() thẳng vì sẽ bị trộn data giữa các danh mục
    lag_features = ['cat_revenue', 'unique_buyers', 'whale_interactions', 'newbie_interactions']
    lags = [1, 7, 14]
    
    for feat in lag_features:
        for lag in lags:
            df[f'{feat}_lag_{lag}'] = df.groupby('category')[feat].shift(lag)
            
    # 2. Tạo Rolling Mean theo TỪNG DANH MỤC
    for w in [3, 7]:
        df[f'revenue_roll_{w}'] = df.groupby('category')['cat_revenue'].transform(
            lambda x: x.shift(1).rolling(w).mean()
        )
        
    # 3. Calendar Features cơ bản
    df['day_of_week'] = df['order_date'].dt.dayofweek
    df['day_of_month'] = df['order_date'].dt.day
    df['month'] = df['order_date'].dt.month
    
    if drop_na:
        return df.dropna().reset_index(drop=True)
    return df.reset_index(drop=True)

def main():
    print("🚀 Khởi tạo Global LightGBM...")
    
    # 1. Load Data
    df = pd.read_csv(DATA_DIR / "category_daily_rfm.csv")
    df = create_global_features(df, drop_na=True)
    
    # 2. Split Holdout (Lấy 365 ngày cuối làm Validation)
    max_date = df['order_date'].max()
    split_date = max_date - pd.Timedelta(days=365)
    
    train_df = df[df['order_date'] < split_date].copy()
    val_df = df[df['order_date'] >= split_date].copy()
    
    features = [c for c in df.columns if c not in [
        'order_date', 'category', 'cat_revenue',
        'unique_buyers', 'whale_interactions', 'newbie_interactions', 'loyal_interactions'
    ]]
    target = 'cat_revenue'
    
    # 3. Khởi tạo Global LightGBM (Tweedie để bắt đỉnh)
    model = lgb.LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=63,
        objective='tweedie',
        tweedie_variance_power=1.35,
        random_state=2026,
        n_jobs=-1
    )
    
    # 4. Huấn luyện (Chỉ định rõ category_idx là biến phân loại)
    # Train trên TOÀN BỘ dữ liệu để dự báo tương lai
    print("🧠 Đang huấn luyện mô hình Global trên toàn bộ dữ liệu...")
    model.fit(
        df[features], df[target],
        categorical_feature=['category_idx']
    )
    
    # 5. Đánh giá: Tạo hàm dự báo đệ quy (Recursive Forecasting)
    print("🔮 Đang dự báo đệ quy cho 548 ngày tương lai...")
    future_dates = pd.date_range(start='2023-01-01', end='2024-07-01')
    categories = df['category'].unique()
    
    # Tạo frame tương lai
    history_df = df.copy()
    
    for date in future_dates:
        new_rows = []
        for cat in categories:
            cat_hist = history_df[history_df['category'] == cat].copy()
            cat_idx = cat_hist['category_idx'].iloc[0]
            
            # Khởi tạo row mới
            row = {'order_date': date, 'category': cat, 'category_idx': cat_idx}
            
            # Điền các biến Time-series naive (dùng lại của tuần trước)
            for feat in ['unique_buyers', 'whale_interactions', 'newbie_interactions', 'loyal_interactions']:
                val_last_week = cat_hist.iloc[-7][feat] if len(cat_hist) >= 7 else 0
                row[feat] = val_last_week
                
            row['cat_revenue'] = np.nan # Cần dự đoán
            new_rows.append(row)
            
        new_df = pd.DataFrame(new_rows)
        history_df = pd.concat([history_df, new_df], ignore_index=True)
        
        # Cập nhật lại toàn bộ features (không drop NAs vì cat_revenue đang trống)
        history_df = create_global_features(history_df, drop_na=False)
        
        # Lấy ra những dòng của ngày hôm nay để predict
        today_idx = history_df['order_date'] == date
        X_today = history_df.loc[today_idx, features]
        
        # Dự đoán
        preds = model.predict(X_today)
        history_df.loc[today_idx, 'cat_revenue'] = np.maximum(preds, 0) # Không cho số âm

    # 6. Gom tổng Doanh thu thành Daily Level và lưu Submission
    future_df = history_df[history_df['order_date'] >= '2023-01-01']
    submission = future_df.groupby('order_date')['cat_revenue'].sum().reset_index()
    submission.columns = ['Date', 'Revenue']
    
    # Format giống sample_submission.csv (Cần thêm COGS - Giả định COGS = Revenue * 0.88 như các phiên bản trước)
    submission['COGS'] = submission['Revenue'] * 0.88
    submission['Date'] = submission['Date'].dt.strftime('%Y-%m-%d')
    
    sub_path = DATA_DIR / "submission_v20_global.csv"
    submission.to_csv(sub_path, index=False)
    print(f"✅ Hoàn tất! Đã lưu {len(submission)} ngày dự báo tại {sub_path}")

    


if __name__ == "__main__":
    main()
