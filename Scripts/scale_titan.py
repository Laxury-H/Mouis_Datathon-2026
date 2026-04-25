import pandas as pd
import numpy as np
from pathlib import Path

SUBMISSIONS_DIR = Path("Results/submissions")

def main():
    print("🚀 Scaling V21 Titan to Target Mean 4.45M...")
    
    # Load Titan
    titan = pd.read_csv(SUBMISSIONS_DIR / "submission_v21_titan_20seed.csv")
    
    current_mean = titan['Revenue'].mean()
    target_mean = 4450000.0
    
    scaling_factor = target_mean / current_mean
    print(f"Current Mean: {current_mean:,.0f} | Target: {target_mean:,.0f} | Factor: {scaling_factor:.4f}")
    
    titan_scaled = titan.copy()
    titan_scaled['Revenue'] = titan_scaled['Revenue'] * scaling_factor
    titan_scaled['COGS'] = titan_scaled['COGS'] * scaling_factor
    
    out_path = SUBMISSIONS_DIR / "submission_v21_titan_scaled_445.csv"
    titan_scaled.to_csv(out_path, index=False)
    
    print(f"✅ Created: {out_path.name}")

if __name__ == "__main__":
    main()
