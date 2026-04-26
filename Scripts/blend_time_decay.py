import pandas as pd
import numpy as np
from pathlib import Path

def main():
    # Base: 70% V25 (Recursive) + 30% V18 (Deep Learning)
    base_path = Path("Results/submissions/v25_sweeps/submission_v25_blend_v18_a30.csv")
    # Anchor: V28 (One-Shot Components, 683k MAE)
    v28_path = Path("Results/submissions/submission_v28_oneshot_comp_raw.csv")
    
    out_dir = Path("Results/submissions/final_sweeps")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("🚀 TIME-DECAY BLENDING: Fading out Recursive, Fading in One-Shot...")
    
    base = pd.read_csv(base_path)
    v28 = pd.read_csv(v28_path)
    
    n_days = len(base)
    # Tỉ trọng của V28 sẽ tăng dần từ start_w đến end_w
    
    configs = [
        (0.05, 0.40), # Bắt đầu với 5% V28, kết thúc với 40% V28
        (0.10, 0.50), # Bắt đầu với 10% V28, kết thúc với 50% V28
        (0.15, 0.60), # Bắt đầu với 15% V28, kết thúc với 60% V28
    ]
    
    for start_w, end_w in configs:
        blend = base.copy()
        
        # Tạo mảng trọng số tuyến tính tăng dần cho V28
        w_v28 = np.linspace(start_w, end_w, n_days)
        w_base = 1.0 - w_v28
        
        blend["Revenue"] = w_base * base["Revenue"] + w_v28 * v28["Revenue"]
        blend["COGS"] = w_base * base["COGS"] + w_v28 * v28["COGS"]
        
        out_name = f"submission_time_decay_{int(start_w*100)}_{int(end_w*100)}.csv"
        blend.to_csv(out_dir / out_name, index=False)
        print(f"✅ Đã lưu: {out_name} (Mean: {blend['Revenue'].mean():,.0f})")

if __name__ == "__main__":
    main()
