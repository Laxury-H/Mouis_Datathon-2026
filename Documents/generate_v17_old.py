"""
V17-Turbo — The V9 Killer
Kiến trúc: 100% XGBoost DNA V9 + 5-Seed Ensemble.
Multiplier: 1.32x (Tăng nhẹ để vượt Mean V9).
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

DATA = 'c:/Users/Lax/Downloads/Demo/Antigravity/Data/data2.0/'
GROWTH_MULTIPLIER = 1.32    # Tùy chỉnh để vượt V9
COGS_RATIO = 0.8862
SEEDS = [42, 100, 2024, 777, 9999]

# Load & Prep
sales = pd.read_csv(DATA + 'sales.csv', parse_dates=['Date'])
sales = sales.sort_values('Date').reset_index(drop=True)
sales = sales[sales['Date'] >= '2019-01-01'].copy().reset_index(drop=True)

def build_features(dates_series):
    df = pd.DataFrame({'Date': dates_series})
    df['day_of_year']  = df['Date'].dt.dayofyear
    df['day_of_month'] = df['Date'].dt.day
    df['day_of_week']  = df['Date'].dt.dayofweek
    df['month']        = df['Date'].dt.month
    df['is_tet_holiday'] = ((df['month'] == 1) | (df['month'] == 2)).astype(int)
    df['is_11_11'] = ((df['month'] == 11) & (df['day_of_month'] == 11)).astype(int)
    df['is_12_12'] = ((df['month'] == 12) & (df['day_of_month'] == 12)).astype(int)
    return df

train_feat = build_features(sales['Date'])
feature_cols = [c for c in train_feat.columns if c != 'Date']
X_train = train_feat[feature_cols].values
y_train = sales['Revenue'].values

print(f"🚀 Huấn luyện V17-Turbo (Multiplier={GROWTH_MULTIPLIER}x)...")
models = []
for seed in SEEDS:
    m = xgb.XGBRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8,
        objective='reg:absoluteerror', random_state=seed, verbosity=0
    )
    m.fit(X_train, y_train)
    models.append(m)

sub = pd.read_csv(DATA + 'sample_submission.csv')
sub['Date'] = pd.to_datetime(sub['Date'])
test_feat = build_features(sub['Date'])
X_test = test_feat[feature_cols].values

preds = np.zeros(len(X_test))
for m in models:
    preds += m.predict(X_test)
preds /= len(models)

sub['Revenue'] = preds * GROWTH_MULTIPLIER
sub['COGS']    = sub['Revenue'] * COGS_RATIO

sub['Date'] = sub['Date'].dt.strftime('%Y-%m-%d')
out = DATA + 'submission_v17_turbo.csv'
sub.to_csv(out, index=False)

print(f"\n✅ ĐÃ TẠO: {out}")
print(f"   Mean Revenue: {sub['Revenue'].mean():,.0f} (V9 là 4,197,622)")
print(f"   Multiplier: {GROWTH_MULTIPLIER}x")
