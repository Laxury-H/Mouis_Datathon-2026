# 🏆 Champion: V25 Blend a40 (40% V25 + 60% V18)

**Score**: 676,541 MAE (Public Leaderboard)  
**Date**: April 26, 2026  

## Cơ chế hoạt động (The Mechanics)
Đây là minh chứng kinh điển cho sức mạnh của **Ensemble Stacking (Tương quan âm)**:
* Bản V18 (Dự báo trực tiếp Revenue) đóng vai trò xương sống ổn định.
* Bản V25 Raw (Bóc tách Orders * AOV) tuy có sai số lớn nhưng lại bắt được những đặc trưng phi tuyến tính ẩn giấu.
* Bằng cách điều chỉnh tỷ trọng chệch về V18 (60%), chúng ta giữ được độ ổn định của đường cơ sở, đồng thời tận dụng được các "tín hiệu cực kỳ nhiễu nhưng quý giá" từ V25 (40%).

## Cách tái tạo (Reproducibility)
Bản nộp này được tạo ra từ script `blend_v25_sweep.py` với cấu hình:
- `alpha = 0.4`
- `V18`: `submission_v18_dl_stack_anchor_a22.csv`
- `V25`: `submission_v25_comp_stack_raw.csv`
