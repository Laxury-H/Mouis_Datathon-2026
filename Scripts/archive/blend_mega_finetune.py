import pandas as pd
from pathlib import Path

def main():
    champ_path = Path("Results/submissions/v25_sweeps/submission_v25_blend_v18_a30.csv")
    titan_path = Path("Results/history/submissions/v21/submission_v21_titan_scaled_445.csv")
    out_dir = Path("Results/submissions/mega_sweeps")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("🚀 MEGA BLEND FINE-TUNING...")
    
    champ = pd.read_csv(champ_path)
    titan = pd.read_csv(titan_path)
    
    # 5%, 6%, 7%, 8%, 9% Titan
    alphas = [0.05, 0.06, 0.07, 0.08, 0.09, 0.11, 0.12]
    
    for alpha in alphas:
        blend = champ.copy()
        blend["Revenue"] = (1 - alpha) * champ["Revenue"] + alpha * titan["Revenue"]
        blend["COGS"] = (1 - alpha) * champ["COGS"] + alpha * titan["COGS"]
        
        out_name = f"submission_mega_champ_a{int((1-alpha)*100)}_titan_a{int(alpha*100)}.csv"
        blend.to_csv(out_dir / out_name, index=False)
        print(f"✅ Đã lưu: {out_name} (Mean Revenue: {blend['Revenue'].mean():,.0f})")

if __name__ == "__main__":
    main()
