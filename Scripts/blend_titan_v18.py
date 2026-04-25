import pandas as pd
import numpy as np
from pathlib import Path

SUBMISSIONS_DIR = Path("Results/submissions")

def main():
    print("🔀 Blending V21 Titan (Scaled) + V18 a25...")
    
    # Load models
    titan = pd.read_csv(SUBMISSIONS_DIR / "submission_v21_titan_scaled_445.csv")
    v18 = pd.read_csv(SUBMISSIONS_DIR / "submission_v18_dl_stack_anchor_a25.csv")
    
    print(f"Titan Scaled Mean: {titan['Revenue'].mean():,.0f}")
    print(f"V18 a25 Mean:      {v18['Revenue'].mean():,.0f}")
    
    # Blend Ratios (Alpha = % Titan, 1-Alpha = % V18)
    alphas = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
    
    for alpha in alphas:
        blended = titan.copy()
        blended['Revenue'] = alpha * titan['Revenue'] + (1 - alpha) * v18['Revenue']
        blended['COGS'] = alpha * titan['COGS'] + (1 - alpha) * v18['COGS']
        
        fname = f"submission_blend_titan_{int(alpha*100)}_v18_{int((1-alpha)*100)}.csv"
        out_path = SUBMISSIONS_DIR / fname
        blended.to_csv(out_path, index=False)
        print(f"  α={alpha:.2f} | Titan:{int(alpha*100)}% + V18:{int((1-alpha)*100)}% | Mean: {blended['Revenue'].mean():,.0f} -> {fname}")

    print("\n✅ Ultimate Blends created in Results/submissions/.")

if __name__ == "__main__":
    main()
