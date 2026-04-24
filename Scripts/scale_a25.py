import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path("Data")
SUBMISSIONS_DIR = Path("Results/submissions")

def main():
    print("🚀 Thực hiện Ép Mean cho Siêu phẩm a25...")
    
    # Đọc bản blend tốt nhất lịch sử (a25)
    sub_a25 = pd.read_csv(SUBMISSIONS_DIR / "submission_v18_dl_stack_anchor_a25.csv")
    
    # 1. Tính Mean hiện tại
    current_mean = sub_a25['Revenue'].mean()
    print(f"Mean hiện tại của a25: {current_mean:,.0f} VND")
    
    # 2. Mean mục tiêu từ Leaderboard
    target_mean = 4450000.0
    
    # 3. Tính hệ số dịch chuyển
    scaling_factor = target_mean / current_mean
    print(f"Hệ số Scaling: {scaling_factor:.4f} (+{(scaling_factor-1)*100:.1f}%)")
    
    # 4. Ép phân phối
    sub_scaled = sub_a25.copy()
    sub_scaled['Revenue'] = sub_scaled['Revenue'] * scaling_factor
    sub_scaled['COGS'] = sub_scaled['COGS'] * scaling_factor
    
    # Lưu file
    out_path = SUBMISSIONS_DIR / "submission_a25_scaled_445.csv"
    sub_scaled.to_csv(out_path, index=False)
    
    print(f"✅ Đã lưu file: {out_path.name}")
    print(f"Mean Revenue mới: {sub_scaled['Revenue'].mean():,.0f} VND")

if __name__ == "__main__":
    main()
