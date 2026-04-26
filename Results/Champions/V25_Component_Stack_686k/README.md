# 🏆 Champion: V25 Component Stack

**Score**: 686,548 MAE (Public Leaderboard)  
**Date**: April 26, 2026  

## Cơ chế hoạt động (The Mechanics)
Mô hình này từ bỏ việc bóc tách dữ liệu theo "Danh mục" (Category) vì nó gây nhiễu chéo, thay vào đó đi vào bản chất gốc rễ của dòng tiền:
`Revenue = Total Orders * Average Order Value (AOV)`

1. **Total Orders Model**: Một tập hợp Ensemble (XGBoost, LightGBM, CatBoost - 10 seeds) dự báo số lượng đơn hàng đệ quy (Recursive). Sử dụng lags (1, 2, 3, 7, 14, 28) và rolling means.
2. **AOV Model**: Một tập hợp Ensemble tương tự (nhưng giới hạn độ sâu cây nông hơn vì AOV ít biến động) dự báo giá trị trung bình trên mỗi đơn hàng.
3. **Combination**: Nhân hai dự báo lại với nhau để ra `Raw Revenue`.
4. **Anchor Blend**: Bản tốt nhất đạt `686k` là bản Blend 50/50 giữa `V25 Raw` và cựu vương `V18 (bản a22)`.

## Cách tái tập huấn (Reproducibility)
1. Chỉ cần chạy file `model_v25_components.py` trong thư mục này.
2. Nó sẽ tự động đọc `daily_buyers_aov.csv` từ thư mục `Data/`.
3. Script cũng sẽ tự động xuất ra 2 file nộp (Raw và Anchor).
