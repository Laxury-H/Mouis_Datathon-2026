import polars as pl
from pathlib import Path

DATA_DIR = Path("Data") 

def build_daily_buyers_aov():
    print("🚀 Khởi chạy Polars Lazy Engine: Tối ưu hoá RAM...")

    # 1. Xử lý orders.csv (Đếm số đơn và số người mua)
    if not (DATA_DIR / "orders.csv").exists():
        print("❌ Không tìm thấy Data/orders.csv")
        return
    orders_lazy = pl.scan_csv(DATA_DIR / "orders.csv")
    daily_orders = (
        orders_lazy
        .with_columns(pl.col("order_date").str.to_datetime().dt.date().alias("date"))
        .group_by("date")
        .agg([
            pl.col("order_id").n_unique().alias("total_orders"),
            pl.col("customer_id").n_unique().alias("unique_buyers")
        ])
    )

    # 2. Lấy Revenue từ sales.csv
    if not (DATA_DIR / "sales.csv").exists():
        print("❌ Không tìm thấy Data/sales.csv")
        return
    sales_lazy = pl.scan_csv(DATA_DIR / "sales.csv")
    sales_daily = (
        sales_lazy
        .with_columns(pl.col("Date").str.to_datetime().dt.date().alias("date"))
        .select(["date", "Revenue"])
    )

    # 3. Xử lý web_traffic.csv
    if (DATA_DIR / "web_traffic.csv").exists():
        web_lazy = pl.scan_csv(DATA_DIR / "web_traffic.csv")
        web_traffic = (
            web_lazy
            .with_columns(pl.col("date").str.to_datetime().dt.date().alias("date"))
            .group_by("date")
            .agg([
                pl.col("sessions").sum().alias("sessions"),
                pl.col("unique_visitors").sum().alias("visitors")
            ])
        )
        # Join Orders, Sales và Web Traffic
        final_lazy = daily_orders.join(sales_daily, on="date", how="left")
        final_lazy = final_lazy.join(web_traffic, on="date", how="left")
    else:
        print("⚠️ Không tìm thấy web_traffic.csv, chỉ trích xuất từ orders và sales...")
        final_lazy = daily_orders.join(sales_daily, on="date", how="left")

    # Tính AOV
    final_lazy = final_lazy.with_columns(
        (pl.col("Revenue") / pl.col("total_orders")).alias("AOV")
    )

    # 4. Thực thi gom nhóm
    print("⏳ Đang tổng hợp dữ liệu...")
    final_df = final_lazy.sort("date").collect()

    # Lưu ra file CSV để dùng cho XGBoost
    out_path = DATA_DIR / "daily_buyers_aov.csv"
    final_df.write_csv(out_path)
    
    print(f"✅ Hoàn tất! Đã lưu {final_df.height} dòng tại {out_path}")
    print(final_df.head(5))

if __name__ == "__main__":
    build_daily_buyers_aov()
