import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path("Data")
V21_DIR = Path("Results/history/submissions/v21")

def main():
    print("🔀 Blending V21 Scaled + V18 a25...")
    
    # Load cả 2 bản submission
    v21 = pd.read_csv(V21_DIR / "submission_v21_scaled.csv")
    v18 = pd.read_csv(DATA_DIR / "submission_v18_dl_stack_anchor_a25.csv")
    
    print(f"V21 Scaled Mean Revenue: {v21['Revenue'].mean():,.0f}")
    print(f"V18 a25 Mean Revenue:    {v18['Revenue'].mean():,.0f}")
    
    # Thử nhiều tỷ lệ blend
    alphas = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70]
    
    for alpha in alphas:
        blended = v21.copy()
        # alpha = tỷ trọng của V21, (1-alpha) = tỷ trọng của V18
        blended['Revenue'] = alpha * v21['Revenue'] + (1 - alpha) * v18['Revenue']
        blended['COGS'] = alpha * v21['COGS'] + (1 - alpha) * v18['COGS']
        
        mean_rev = blended['Revenue'].mean()
        
        fname = f"submission_blend_v21_{int(alpha*100)}_v18_{int((1-alpha)*100)}.csv"
        out_path = DATA_DIR / fname
        blended.to_csv(out_path, index=False)
        
        print(f"  α={alpha:.2f} | V21:{int(alpha*100)}% + V18:{int((1-alpha)*100)}% | Mean: {mean_rev:,.0f} | -> {fname}")
    
    print(f"\n✅ Đã tạo {len(alphas)} bản blend trong Data/. Chọn bản có Mean gần 4.45M nhất để submit!")

if __name__ == "__main__":
    main()
