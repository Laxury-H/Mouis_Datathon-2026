import pandas as pd
from pathlib import Path

def main():
    champ_path = Path("Results/submissions/v25_sweeps/submission_v25_blend_v18_a30.csv")
    titan_path = Path("Results/history/submissions/v21/submission_v21_titan_scaled_445.csv")
    out_dir = Path("Results/submissions/mega_sweeps")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("🚀 Khởi động MEGA BLEND: Tích hợp One-Shot Titan vào Vua V25...")
    
    if not champ_path.exists() or not titan_path.exists():
        print(f"❌ Thiếu file. Vui lòng kiểm tra lại đường dẫn.")
        print(f"Champ: {champ_path.exists()}")
        print(f"Titan: {titan_path.exists()}")
        return
        
    champ = pd.read_csv(champ_path)
    titan = pd.read_csv(titan_path)
    
    # Tạo các bản blend kết hợp thêm sức mạnh ổn định của Titan
    alphas = [0.05, 0.10, 0.15, 0.20, 0.25]
    
    for alpha in alphas:
        blend = champ.copy()
        # Alpha % Titan + (1 - Alpha) % Champion
        blend["Revenue"] = (1 - alpha) * champ["Revenue"] + alpha * titan["Revenue"]
        blend["COGS"] = (1 - alpha) * champ["COGS"] + alpha * titan["COGS"]
        
        out_name = f"submission_mega_champ_a{int((1-alpha)*100)}_titan_a{int(alpha*100)}.csv"
        blend.to_csv(out_dir / out_name, index=False)
        print(f"✅ Đã lưu: {out_name} (Mean Revenue: {blend['Revenue'].mean():,.0f})")

if __name__ == "__main__":
    main()
