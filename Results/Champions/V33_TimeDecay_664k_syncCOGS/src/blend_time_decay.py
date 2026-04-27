import pandas as pd
import numpy as np
from pathlib import Path

def main():
    # Base: 70% V25 (Recursive) + 30% V18 (Deep Learning)
    base_path = Path("Results/submissions/v25_sweeps/submission_v25_blend_v18_a30.csv")
    # Anchor: V28 (One-Shot Components, 683k MAE)
    v28_path = Path("Results/submissions/submission_v28_oneshot_comp_raw.csv")
    
    # V18 Deep Learning Path to extract Dynamic COGS Ratio
    v18_path = Path("Results/history/submissions/v18/submission_v18_dl_stack_anchor_a22.csv")
    
    out_dir = Path("Results/submissions/final_sweeps")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("🚀 TIME-DECAY BLENDING + SYNC COGS: Fading out Recursive, Fading in One-Shot...")
    
    base = pd.read_csv(base_path)
    v28 = pd.read_csv(v28_path)
    v18 = pd.read_csv(v18_path)
    
    # Tính V18 Dynamic COGS Ratio
    v18_ratio = v18["COGS"] / v18["Revenue"]
    
    n_days = len(base)
    # Tỉ trọng của V28 sẽ tăng dần từ start_w đến end_w
    
    configs = [
        (0.10, 0.50), # 665k benchmark
        (0.00, 0.60), # Maximize V25 at start, heavy V28 at end
        (0.05, 0.55), # Balanced
        (0.00, 0.70), # Extreme drift correction
        (0.10, 0.60), # Safe shift
    ]
    
    for start_w, end_w in configs:
        blend = base.copy()
        
        # Tạo mảng trọng số tuyến tính tăng dần cho V28
        w_v28 = np.linspace(start_w, end_w, n_days)
        w_base = 1.0 - w_v28
        
        # Chỉ blend Revenue
        blend["Revenue"] = w_base * base["Revenue"] + w_v28 * v28["Revenue"]
        
        # Sync COGS bằng V18 DL Ratio
        blend["COGS"] = blend["Revenue"] * v18_ratio
        
        out_name = f"submission_time_decay_{int(start_w*100)}_{int(end_w*100)}_syncCOGS.csv"
        blend.to_csv(out_dir / out_name, index=False)
        print(f"✅ Đã lưu: {out_name} (Mean: {blend['Revenue'].mean():,.0f})")

if __name__ == "__main__":
    main()
