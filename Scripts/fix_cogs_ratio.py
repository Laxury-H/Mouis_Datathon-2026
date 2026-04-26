import pandas as pd
from pathlib import Path

def main():
    print("🚀 Khởi động COGS Ratio Synchronizer...")
    
    # Load the best submission so far
    mega_path = Path("Results/submissions/mega_sweeps/submission_mega_champ_a93_titan_a7.csv")
    
    # Load V18 to get its Deep Learning COGS ratio
    v18_path = Path("Results/history/submissions/v18/submission_v18_dl_stack_anchor_a22.csv")
    
    if not mega_path.exists() or not v18_path.exists():
        print("Missing files")
        return
        
    mega = pd.read_csv(mega_path)
    v18 = pd.read_csv(v18_path)
    
    # Tính toán tỷ lệ biên lợi nhuận (COGS/Revenue) được dự báo bằng Deep Learning từ V18
    v18_ratio = v18["COGS"] / v18["Revenue"]
    
    # Áp dụng tỷ lệ này lên Revenue cực chuẩn của Mega Blend
    mega_synced = mega.copy()
    mega_synced["COGS"] = mega_synced["Revenue"] * v18_ratio
    
    # Cũng áp dụng cho bản 8% Titan
    mega_8_path = Path("Results/submissions/mega_sweeps/submission_mega_champ_a92_titan_a8.csv")
    if mega_8_path.exists():
        mega_8 = pd.read_csv(mega_8_path)
        mega_8["COGS"] = mega_8["Revenue"] * v18_ratio
        mega_8.to_csv(mega_8_path.parent / "submission_mega_champ_a92_titan_a8_syncCOGS.csv", index=False)
        print("✅ Đã lưu: submission_mega_champ_a92_titan_a8_syncCOGS.csv")

    out_name = "submission_mega_champ_a93_titan_a7_syncCOGS.csv"
    mega_synced.to_csv(mega_path.parent / out_name, index=False)
    print(f"✅ Đã lưu: {out_name}")

if __name__ == "__main__":
    main()
