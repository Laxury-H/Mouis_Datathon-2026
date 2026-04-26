import pandas as pd
from pathlib import Path

def main():
    # Base: 70% V25 (Recursive) + 30% V18 (Deep Learning)
    base_path = Path("Results/submissions/v25_sweeps/submission_v25_blend_v18_a30.csv")
    # Anchor: V28 (One-Shot Components, 683k MAE)
    v28_path = Path("Results/submissions/submission_v28_oneshot_comp_raw.csv")
    
    out_dir = Path("Results/submissions/god_tier_sweeps")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("🚀 BLENDING PURE GOD TIER: (V25 + V18) + V28 ...")
    
    base = pd.read_csv(base_path)
    v28 = pd.read_csv(v28_path)
    
    # We sweep from 5% to 40% V28
    alphas = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
    
    for alpha in alphas:
        blend = base.copy()
        blend["Revenue"] = (1 - alpha) * base["Revenue"] + alpha * v28["Revenue"]
        blend["COGS"] = (1 - alpha) * base["COGS"] + alpha * v28["COGS"]
        
        out_name = f"submission_god_tier_v28_{int(alpha*100)}.csv"
        blend.to_csv(out_dir / out_name, index=False)
        print(f"✅ Đã lưu: {out_name} (Mean Revenue: {blend['Revenue'].mean():,.0f})")

if __name__ == "__main__":
    main()
