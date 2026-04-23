import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path("Data")
SEED = 2026

PROMO_PRIORS = {}

def init_promo_priors():
    global PROMO_PRIORS
    promos = pd.read_csv(DATA_DIR / "promotions.csv")
    
    # Expand promotions to daily
    daily_promos = []
    for _, row in promos.iterrows():
        dates = pd.date_range(row['start_date'], row['end_date'])
        for d in dates:
            daily_promos.append({
                'date': d,
                'discount_value': row['discount_value'] if row['promo_type'] == 'percentage' else 0, 
            })
    
    dp_df = pd.DataFrame(daily_promos)
    dp_df['doy'] = dp_df['date'].dt.dayofyear
    
    doy_discount = dp_df.groupby('doy')['discount_value'].max().to_dict()
    PROMO_PRIORS['doy_discount'] = doy_discount

def create_deterministic_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    # 1. Label Encode Category
    le = LabelEncoder()
    df['category_idx'] = le.fit_transform(df['category'])
    
    # 2. Time Index
    min_date = pd.to_datetime('2013-01-01') # Constant min date so test set is consistent
    df['time_index'] = (df['order_date'] - min_date).dt.days
    
    # 3. Calendar Basics
    df['day_of_month'] = df['order_date'].dt.day
    df['day_of_week'] = df['order_date'].dt.dayofweek
    df['month'] = df['order_date'].dt.month
    df['doy'] = df['order_date'].dt.dayofyear
    
    # 4. Flags
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['is_payday'] = ((df['day_of_month'] >= 25) | (df['day_of_month'] <= 3)).astype(int)
    df['is_mega_sale'] = (((df['month'] == 11) & (df['day_of_month'] == 11)) | 
                          ((df['month'] == 12) & (df['day_of_month'] == 12))).astype(int)
    
    # 5. Fourier Terms
    df['doy_sin'] = np.sin(2 * np.pi * df['doy'] / 365.25)
    df['doy_cos'] = np.cos(2 * np.pi * df['doy'] / 365.25)
    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7.0)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7.0)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12.0)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12.0)
    
    # 6. Promo Priors
    df['promo_discount'] = df['doy'].map(PROMO_PRIORS.get('doy_discount', {})).fillna(0)
    
    # 7. Cross-features
    df['promo_x_payday'] = df['promo_discount'] * df['is_payday']
    
    return df

def main():
    print("🚀 Khởi tạo V21 ONE-SHOT Category LightGBM...")
    init_promo_priors()
    
    # Load training data
    df = pd.read_csv(DATA_DIR / "category_daily_rfm.csv")
    df['order_date'] = pd.to_datetime(df['order_date'])
    df = df[['order_date', 'category', 'cat_revenue']] 
    
    df = create_deterministic_features(df)
    
    # Split Validation 
    max_date = df['order_date'].max() 
    split_date = max_date - pd.Timedelta(days=365)
    
    train_df = df[df['order_date'] < split_date].copy()
    val_df = df[df['order_date'] >= split_date].copy()
    
    features = [c for c in df.columns if c not in ['order_date', 'category', 'cat_revenue']]
    target = 'cat_revenue'
    
    model = lgb.LGBMRegressor(
        n_estimators=1500,
        learning_rate=0.02, 
        num_leaves=31,      
        objective='tweedie',
        tweedie_variance_power=1.35, 
        colsample_bytree=0.8,
        subsample=0.8,
        random_state=SEED,
        n_jobs=-1,
        verbose=-1
    )
    
    print("🧠 Đang huấn luyện One-Shot Model trên tập Train...")
    model.fit(
        train_df[features], train_df[target],
        eval_set=[(val_df[features], val_df[target])],
        categorical_feature=['category_idx'],
        callbacks=[lgb.early_stopping(100, verbose=False)]
    )
    
    # Validation
    val_df['pred_cat_revenue'] = np.maximum(model.predict(val_df[features]), 0)
    daily_eval = val_df.groupby('order_date').agg(
        true_total=('cat_revenue', 'sum'),
        pred_total=('pred_cat_revenue', 'sum')
    ).reset_index()
    mae = mean_absolute_error(daily_eval['true_total'], daily_eval['pred_total'])
    print(f"🎯 V21 One-Shot MAE trên 365 ngày Validation: {mae:,.0f} VND")
    
    importance = pd.DataFrame({'Feature': features, 'Importance': model.feature_importances_}).sort_values(by='Importance', ascending=False)
    print("\n📊 Top Features Định Mệnh:")
    print(importance.head(7))
    
    # Train FULL model
    print("\n🧠 Đang huấn luyện Full Model cho Submission...")
    model.fit(
        df[features], df[target],
        categorical_feature=['category_idx']
    )
    
    # 🔮 DỰ BÁO ONE-SHOT (Không đệ quy)
    print("🔮 Đang dự báo One-Shot cho 548 ngày tương lai...")
    sample = pd.read_csv(DATA_DIR / "sample_submission.csv")
    future_dates = pd.to_datetime(sample['Date'])
    categories = df['category'].unique()
    
    new_rows = []
    for date in future_dates:
        for cat in categories:
            new_rows.append({'order_date': date, 'category': cat})
            
    future_df = pd.DataFrame(new_rows)
    future_df = create_deterministic_features(future_df)
    
    future_df['pred_cat_revenue'] = np.maximum(model.predict(future_df[features]), 0)
    
    # Gom về Daily
    submission = future_df.groupby('order_date')['pred_cat_revenue'].sum().reset_index()
    submission.columns = ['Date', 'Revenue']
    
    # COGS
    sample['ratio'] = sample['COGS'] / sample['Revenue']
    submission['COGS'] = submission['Revenue'] * sample['ratio'].values
    submission['Date'] = submission['Date'].dt.strftime('%Y-%m-%d')
    
    sub_path = DATA_DIR / "submission_v21_oneshot.csv"
    submission.to_csv(sub_path, index=False)
    print(f"✅ Đã kết xuất siêu Base Model V21 tại {sub_path.name}")
    print(f"Mean V21 Revenue: {submission['Revenue'].mean():.1f}")

if __name__ == "__main__":
    main()
