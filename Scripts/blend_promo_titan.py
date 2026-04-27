import pandas as pd
from pathlib import Path

def main():
    champ_path = Path("Results/submissions/final_sweeps/submission_time_decay_10_50_syncCOGS.csv")
    titan_path = Path("Results/submissions/submission_v31_promo_titan_scaled.csv")
    
    out_dir = Path("Results/submissions/promo_sweeps")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    if not champ_path.exists() or not titan_path.exists():
        print("Missing files!")
        return
        
    champ = pd.read_csv(champ_path)
    titan = pd.read_csv(titan_path)
    
    # Scale Titan xuống bằng đúng Mean của Champion (4.37M) để không làm sai lệch Mean
    target_mean = champ["Revenue"].mean()
    titan_mean = titan["Revenue"].mean()
    scale = target_mean / titan_mean
    titan["Revenue"] = titan["Revenue"] * scale
    titan["COGS"] = titan["COGS"] * scale
    
    print("🚀 BLENDING 664k CHAMPION WITH V31 PROMO TITAN 🚀")
    
    alphas = [0.05, 0.10, 0.15, 0.20, 0.25]
    for alpha in alphas:
        blend = champ.copy()
        blend["Revenue"] = (1 - alpha) * champ["Revenue"] + alpha * titan["Revenue"]
        blend["COGS"] = (1 - alpha) * champ["COGS"] + alpha * titan["COGS"]
        
        out_name = f"submission_champ_promo_titan_{int(alpha*100)}.csv"
        blend.to_csv(out_dir / out_name, index=False)
        print(f"✅ Đã tạo {out_name} (Titan {int(alpha*100)}%)")

if __name__ == "__main__":
    main()
