import pandas as pd
import numpy as np
from pathlib import Path
from scipy.optimize import minimize_scalar

DATA_DIR = Path("Data")
SUBMISSIONS_DIR = Path("Results/submissions")

def main():
    print("🚀 Thực hiện Ma thuật Phi tuyến (Peak Boosting) cho a25...")
    
    # Đọc bản blend a25
    sub_a25 = pd.read_csv(SUBMISSIONS_DIR / "submission_v18_dl_stack_anchor_a25.csv")
    
    current_mean = sub_a25['Revenue'].mean()
    target_mean = 4450000.0
    
    print(f"Mean hiện tại: {current_mean:,.0f} VND")
    print(f"Mean mục tiêu: {target_mean:,.0f} VND")
    
    # Xác định ngưỡng (Ngưỡng phân tách ngày thường và ngày Sale)
    # Dùng Percentile 85 (Chỉ khuếch đại top 15% những ngày cao nhất)
    threshold = np.percentile(sub_a25['Revenue'], 85)
    print(f"Ngưỡng Peak (Top 15%): {threshold:,.0f} VND")
    
    # Hàm mục tiêu để tìm hệ số x sao cho Mean = 4.45M
    def error_function(x):
        temp_rev = sub_a25['Revenue'].copy()
        mask = temp_rev > threshold
        # Công thức: Revenue_mới = Revenue_cũ + x * (Revenue_cũ - ngưỡng)
        temp_rev[mask] = temp_rev[mask] + x * (temp_rev[mask] - threshold)
        return abs(temp_rev.mean() - target_mean)

    # Tìm hệ số x tối ưu
    res = minimize_scalar(error_function, bounds=(0, 5), method='bounded')
    optimal_x = res.x
    print(f"Hệ số khuếch đại Peak (x): {optimal_x:.4f}")
    
    # Áp dụng x vào tập kết quả
    sub_boosted = sub_a25.copy()
    mask = sub_boosted['Revenue'] > threshold
    
    # Cập nhật Revenue
    sub_boosted.loc[mask, 'Revenue'] = sub_boosted.loc[mask, 'Revenue'] + optimal_x * (sub_boosted.loc[mask, 'Revenue'] - threshold)
    
    # Cập nhật COGS theo đúng tỷ lệ để không phá vỡ Gross Margin
    ratio = sub_boosted['Revenue'] / sub_a25['Revenue']
    sub_boosted['COGS'] = sub_boosted['COGS'] * ratio
    
    print(f"Mean Revenue mới: {sub_boosted['Revenue'].mean():,.0f} VND")
    
    # Thống kê
    diff = sub_boosted['Revenue'] - sub_a25['Revenue']
    print(f"Số ngày được boost (Peak Days): {mask.sum()} / 548")
    print(f"Mức boost mạnh nhất vào 1 ngày: +{diff.max():,.0f} VND")
    
    # Hiển thị Top 5 ngày biến động mạnh nhất
    top_diffs = sub_boosted.copy()
    top_diffs['diff'] = diff
    top_diffs = top_diffs.sort_values('diff', ascending=False).head(5)
    print("\n--- TOP 5 NGÀY MEGA SALE ĐƯỢC BƠM TIỀN ---")
    for _, row in top_diffs.iterrows():
        print(f"Ngày {row['Date']}: {sub_a25.loc[sub_a25['Date'] == row['Date'], 'Revenue'].values[0]:,.0f} -> {row['Revenue']:,.0f} (+{row['diff']:,.0f})")
    
    # Lưu file
    out_path = SUBMISSIONS_DIR / "submission_a25_peak_boosted.csv"
    sub_boosted.to_csv(out_path, index=False)
    print(f"\n✅ Đã lưu file: {out_path.name}")

if __name__ == "__main__":
    main()
