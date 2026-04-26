import pandas as pd
from pathlib import Path

v24_path = Path("Results/submissions/submission_v24_cat_stack_raw.csv")
v18_path = Path("Results/history/submissions/v18/submission_v18_dl_stack_anchor_a22.csv")

if not v24_path.exists() or not v18_path.exists():
    print("Files not found")
else:
    v24 = pd.read_csv(v24_path)
    v18 = pd.read_csv(v18_path)
    
    alpha = 0.50 # 50% V24, 50% V18
    
    out = v24.copy()
    out["Revenue"] = (1 - alpha) * v18["Revenue"] + alpha * v24["Revenue"]
    out["COGS"] = (1 - alpha) * v18["COGS"] + alpha * v24["COGS"]
    
    out_path = Path("Results/submissions/submission_v24_cat_stack_anchor_50.csv")
    out.to_csv(out_path, index=False)
    print(f"Created: {out_path.name}")
