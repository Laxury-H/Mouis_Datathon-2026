import pandas as pd
import numpy as np
from pathlib import Path

def main():
    champ_path = Path("Results/submissions/final_sweeps/submission_time_decay_10_50_syncCOGS.csv")
    out_dir = Path("Results/submissions/mean_sweeps")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    if not champ_path.exists():
        print(f"Cannot find {champ_path}")
        return
        
    df = pd.read_csv(champ_path)
    current_mean = df["Revenue"].mean()
    
    print(f"🚀 SCALING CHAMPION (Current Mean: {current_mean:,.0f}) 🚀")
    
    # Quét các mốc Mean từ 4.20M đến 4.45M
    target_means = [4200000.0, 4250000.0, 4300000.0, 4350000.0, 4400000.0, 4450000.0]
    
    for t_mean in target_means:
        scale = t_mean / current_mean
        scaled_df = df.copy()
        
        scaled_df["Revenue"] = scaled_df["Revenue"] * scale
        scaled_df["COGS"] = scaled_df["COGS"] * scale
        
        out_name = f"submission_champ_664k_scaled_{int(t_mean/1000)}k.csv"
        scaled_df.to_csv(out_dir / out_name, index=False)
        print(f"✅ Đã tạo {out_name} (Scale: {scale:.4f})")

if __name__ == "__main__":
    main()
