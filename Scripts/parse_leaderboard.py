import pandas as pd
from pathlib import Path
import re

raw_text = """
submission_time_decay_10_50.csv	Complete	-	14s ago	66590355355
submission_god_tier_v28_15.csv	Complete	-	4m ago	66669294384
submission_champ671k_v28_10.csv	Complete	-	11m ago	66735678477
submission_v31_promo_titan_scaled.csv	Complete	-	16m ago	75068564423
submission_v31_promo_titan_scaled.csv	Error	-	24m ago	N/A
submission_mega_champ_a92_titan_a8_syncCOGS.csv	Complete	-	34m ago	67125104958
submission_mega_champ_a93_titan_a7_syncCOGS.csv	Complete	-	35m ago	67124953265
submission_mega_champ_a93_titan_a7.csv	Complete	-	41m ago	67158229653
submission_mega_champ_a92_titan_a8.csv	Complete	-	41m ago	67165236437
submission_v28_oneshot_comp_anchor_30.csv	Complete	Laxurie	1h ago	68338026735
submission_mega_champ_a85_titan_a15.csv	Complete	-	1h ago	67415408873
submission_mega_champ_a90_titan_a10.csv	Complete	-	1h ago	67193841935
submission_v29_mae_comp_anchor_30.csv	Complete	-	1h ago	71909653444
submission_v25_blend_v18_a20.csv	Complete	-	1h ago	67939544643
submission_v27_yoy_stack_anchor_30.csv	Complete	-	1h ago	71888611369
submission_v26_matrix_anchor_30.csv	Complete	Laxurie	2h ago	69180903280
submission_v25_blend_v18_a40.csv	Complete	Laxurie	2h ago	67654195932
submission_v25_comp_stack_raw.csv	Complete	Laxurie	2h ago	83767286809
submission_v26_matrix_raw.csv	Complete	-	2h ago	82574508975
submission_v25_blend_v18_a30.csv	Complete	-	2h ago	67401585682
submission_v25_comp_stack_anchor_50.csv	Complete	-	2h ago	68654839927
submission_v23_recovery_anchor_a25.csv	Complete	Laxurie	9h ago	73014034500
submission_v24_cat_stack_raw.csv	Complete	-	9h ago	76005946670
submission_v24_cat_stack_anchor_50.csv	Complete	-	9h ago	70571507016
submission_v22_titan_stack_anchor_a25.csv	Complete	Laxurie	10h ago	98775895992
submission_v21_titan_20seed.csv	Complete	Laxurie	10h ago	227438067737
submission_v21_titan_scaled_445.csv	Complete	Laxurie	10h ago	90064272127
submission_v18_dl_stack_anchor_a22.csv	Complete	Muối	3d ago	70143278774
submission_v18_dl_stack_anchor_a12.csv	Complete	Muối	3d ago	71375119172
submission_v18_dl_stack_anchor_a30 (1).csv	Complete	Muối	3d ago	76984773492
submission_v18_dl_stack_anchor_a30.csv	Complete	Muối	3d ago	81005827284
submission_v17_6_ratio_doy_g27.csv	Complete	Laxurie	3d ago	74279331278
submission_v17_3_bal75.csv	Complete	Laxurie	3d ago	77477834833
submission_v17_3_safe90.csv	Complete	Laxurie	3d ago	74865335661
submission_v17_2.csv	Complete	Laxurie	3d ago	126039205915
submission_v18_dl_stack_anchor_a25.csv	Complete	-	3d ago	70063766814
submission_v18_dl_stack_anchor_a18.csv	Complete	-	3d ago	70503003152
submission_v18_dl_stack_raw.csv	Complete	-	3d ago	98188242254
submission_v18_dl_stack_anchor_a50.csv	Complete	-	3d ago	86196529233
submission_v17_1b.csv	Complete	-	3d ago	74352536824
submission_v17_1b.csv	Complete	-	3d ago	76692063081
submission_v17_turbo_recover.csv	Complete	-	3d ago	74456081538
submission_v17_1.csv	Complete	-	3d ago	120177498651
submission_v18_scaled_mean_445.csv	Complete	-	3d ago	79260455281
submission_v20_global.csv	Complete	-	3d ago	141339653385
submission_v20_global.csv	Error	-	3d ago	N/A
submission_v18_dl_stack_anchor_a22.csv	Complete	-	3d ago	72242101480
submission_v18_dl_stack_anchor_a30.csv	Complete	-	3d ago	70192958964
submission_v17_5_ratio_doy_g26.csv	Complete	-	3d ago	74279194989
submission_v17_5_ratio_doy_g24.csv	Complete	-	3d ago	74279207281
submission_v17_4_joint_n02.csv	Complete	-	3d ago	74445708312
submission_v17_4_ratio_season_g25.csv	Complete	-	3d ago	74279201135
submission_v18_dl_stack_anchor_a12.csv	Complete	-	3d ago	71405177870
submission_v18_dl_stack_anchor_a11.csv	Complete	-	3d ago	71572664845
submission_v18_dl_stack_anchor_a10.csv	Complete	-	3d ago	71749553887
submission_v18_dl_stack_anchor_a05.csv	Complete	-	3d ago	72874449572
submission.csv	Complete	-	3d ago	100130420451
submission_a25_scaled_445.csv	Complete	-	3d ago	70987314849
submission_blend_v21_70_v18_30.csv	Complete	-	3d ago	73158695577
submission_v21_scaled.csv	Complete	-	3d ago	76703468738
submission_v21_oneshot.csv	Complete	-	3d ago	118412984308
submission.csv	Complete	-	4d ago	106258421656
x.csv	Complete	-	4d ago	122593113792
submission_v20_grandmaster.csv	Complete	Laxurie	5d ago	135036448659
submission_v19_final.csv	Complete	Laxurie	5d ago	78246811618
submission_v17.csv	Complete	Laxurie	5d ago	78871648663
submission_v9.csv	Complete	Laxurie	5d ago	74674623731
submission_v4.csv	Complete	Laxurie	5d ago	126017178501
submission_v3.csv	Complete	Laxurie	5d ago	124438177315
submission_v2.csv	Complete	Laxurie	5d ago	124186935093
submission_v21_titan.csv	Complete	-	5d ago	79648408981
submission_v18_supersurgical.csv	Complete	-	5d ago	77872558586
submission_v18_supersurgical.csv	Error	-	5d ago	N/A
submission_v17.csv	Complete	-	5d ago	77983879706
submission_v17_turbo.csv	Complete	-	5d ago	74154216220
submission_v16_final.csv	Complete	-	5d ago	101044750813
submission_v15_detrended.csv	Complete	-	5d ago	120284499573
submission_v14.csv	Complete	-	5d ago	101802041273
submission.csv	Complete	-	5d ago	122593113792
submission_v13.csv	Complete	-	5d ago	75582410272
submission_v12.csv	Complete	-	5d ago	74936868041
submission_v10.csv	Complete	-	5d ago	77276912661
submission_v8.csv	Complete	-	5d ago	132284458814
submission.csv	Complete	Laxurie	6d ago	152354354029
"""

def main():
    lines = raw_text.strip().split('\n')
    data = []
    
    pattern = re.compile(r"^(\S+\.csv)\s+([A-Za-z]+)\s+(\S+)\s+(.+?)\s+(\d+)$")
    
    for line in lines:
        line = line.strip()
        if not line: continue
        
        match = pattern.search(line)
        if match:
            file_name = match.group(1)
            status = match.group(2)
            user = match.group(3)
            time_ago = match.group(4)
            score_str = match.group(5)
            
            if status != "Complete" or score_str == "N/A":
                continue
                
            try:
                score_float = float(score_str) / 100000.0
                data.append({
                    "File": file_name,
                    "Score": score_float,
                    "Time Ago": time_ago,
                    "User": user
                })
            except ValueError:
                pass

    if not data:
        print("No valid data found")
        return

    df = pd.DataFrame(data)
    # Sắp xếp từ thấp đến cao (càng thấp càng tốt)
    df = df.sort_values("Score", ascending=True).reset_index(drop=True)
    
    # Tạo nội dung Markdown
    md_lines = []
    md_lines.append("# Bảng Vàng Thành Tích (Kaggle Leaderboard)\\n")
    md_lines.append(f"**Tổng số mô hình đã nộp thành công:** {len(df)}\\n")
    md_lines.append("| Hạng | Tên File Submission | Public Score (MAE) | Thời gian nộp |")
    md_lines.append("|---|---|---|---|")
    
    for idx, row in df.iterrows():
        rank = idx + 1
        # highlight top 3
        if rank == 1:
            rank_str = "🏆 1"
            file_str = f"**{row['File']}**"
        elif rank == 2:
            rank_str = "🥈 2"
            file_str = f"**{row['File']}**"
        elif rank == 3:
            rank_str = "🥉 3"
            file_str = f"**{row['File']}**"
        else:
            rank_str = str(rank)
            file_str = row['File']
            
        md_lines.append(f"| {rank_str} | {file_str} | **{row['Score']:,.5f}** | {row['Time Ago']} |")

    out_path = Path("Results/Leaderboard_History.md")
    out_path.write_text("\\n".join(md_lines), encoding="utf-8")
    print("Done")

if __name__ == "__main__":
    main()
