# V25 Mega Blend (671k MAE)

## Architecture
Đây là trạng thái tối ưu nhất của toàn bộ dự án tính đến hiện tại, kết hợp "Tam Kiếm Hợp Bích":
1. **V18 Deep Learning Stack (Anchor - 63%)**: Đóng vai trò mỏ neo tạo đường cơ sở ổn định và học được tỷ lệ Gross Margin (COGS/Revenue) cực kỳ chuẩn xác từ dữ liệu lịch sử.
2. **V25 Component Stack (Tweedie - 27%)**: Bóc tách `Revenue = Orders * AOV` và dự báo đệ quy, sử dụng hàm mục tiêu Tweedie để bắt trọn các đỉnh nến siêu nhỏ và biến động vi mô.
3. **V21 Titan One-Shot (Deterministic - 10%)**: Mô hình phi đệ quy hoàn toàn không có sai số tích lũy, đóng vai trò "cân bằng" lại các đoạn trôi dạt của mô hình đệ quy ở cuối chu kỳ.

## Thành tựu
- Kéo MAE xuống **671,249**.
- Kỹ thuật Synchronize COGS: Ép COGS phải đồng bộ hoàn toàn với tỷ lệ dự báo bằng Deep Learning của V18, giúp triệt tiêu sai số do hằng số `0.8862`.

## Đường đi tiếp theo
Khoảng cách đến Top 1 (623k) là 48k MAE. Điều này ngụ ý chúng ta đã bỏ lỡ một siêu biến lượng (super-feature) có khả năng định hình lại chuỗi dự báo. Cần kiểm tra lại toàn bộ Exogenous Features hoặc cấu trúc chuỗi.
