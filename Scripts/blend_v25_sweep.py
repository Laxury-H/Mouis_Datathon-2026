import pandas as pd
from pathlib import Path

# Paths
v25_path = Path("Results/submissions/submission_v25_comp_stack_raw.csv")
v18_path = Path("Results/history/submissions/v18/submission_v18_dl_stack_anchor_a22.csv")
out_dir = Path("Results/submissions/v25_sweeps")
out_dir.mkdir(parents=True, exist_ok=True)

if not v25_path.exists() or not v18_path.exists():
    print("Files not found. Check paths.")
else:
    v25 = pd.read_csv(v25_path)
    v18 = pd.read_csv(v18_path)
    
    alphas = [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8]
    
    print("🚀 [V25 SWEEP] Tạo các bản mix V18 và V25...")
    for alpha in alphas:
        out = v25.copy()
        # Revenue = (1-alpha) * V18 + alpha * V25
        out["Revenue"] = (1 - alpha) * v18["Revenue"] + alpha * v25["Revenue"]
        out["COGS"] = (1 - alpha) * v18["COGS"] + alpha * v25["COGS"]
        
        name = f"submission_v25_blend_v18_a{int(alpha*100):02d}.csv"
        out_path = out_dir / name
        out.to_csv(out_path, index=False)
        print(f"✅ Đã tạo: {name} (V25={int(alpha*100)}%, V18={int((1-alpha)*100)}%)")
        
    print(f"\\n📁 Tất cả file đã lưu tại: {out_dir}")
