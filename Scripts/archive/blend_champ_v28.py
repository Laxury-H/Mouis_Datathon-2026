import pandas as pd
from pathlib import Path

def main():
    champ_path = Path("Results/Champions/V25_Mega_Blend_671k/submission_champion_671k.csv")
    v28_path = Path("Results/submissions/submission_v28_oneshot_comp_raw.csv")
    out_dir = Path("Results/submissions/mega_sweeps_v28")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("🚀 BLENDING CHAMPION (671k) WITH V28 (One-Shot Components)...")
    
    champ = pd.read_csv(champ_path)
    v28 = pd.read_csv(v28_path)
    
    # 5% to 30% V28
    alphas = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    
    for alpha in alphas:
        blend = champ.copy()
        blend["Revenue"] = (1 - alpha) * champ["Revenue"] + alpha * v28["Revenue"]
        blend["COGS"] = (1 - alpha) * champ["COGS"] + alpha * v28["COGS"]
        
        out_name = f"submission_champ671k_v28_{int(alpha*100)}.csv"
        blend.to_csv(out_dir / out_name, index=False)
        print(f"✅ Đã lưu: {out_name} (Mean Revenue: {blend['Revenue'].mean():,.0f})")

if __name__ == "__main__":
    main()
