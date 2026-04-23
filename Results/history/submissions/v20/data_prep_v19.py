import polars as pl
from pathlib import Path

DATA_DIR = Path("Data")

def prepare_v19_data():
    print("🚀 Khởi chạy Polars Lazy Engine: Tối ưu hoá RAM cho V19...")

    # 1. Load Lazy Frames
    orders = pl.scan_csv(DATA_DIR / "orders.csv")
    items = pl.scan_csv(DATA_DIR / "order_items.csv")
    prods = pl.scan_csv(DATA_DIR / "products.csv")

    orders = orders.with_columns(
        pl.col("order_date").str.to_datetime().dt.date()
    )

    items = items.with_columns(
        (pl.col("quantity") * pl.col("unit_price")).alias("item_revenue")
    )

    # Tính tổng tiền từng đơn hàng
    order_revenue = (
        items.group_by("order_id")
        .agg(pl.col("item_revenue").sum().alias("order_revenue"))
    )

    orders_with_revenue = orders.join(order_revenue, on="order_id", how="left")
    orders_with_revenue = orders_with_revenue.with_columns(
        pl.col("order_revenue").fill_null(0.0)
    )

    # 2. RFM - Cumulative Sum/Count over customer_id
    orders_rfm = (
        orders_with_revenue
        .sort(["customer_id", "order_date"])
        .with_columns([
            pl.lit(1).cum_sum().over("customer_id").alias("total_orders_to_date"),
            pl.col("order_revenue").cum_sum().over("customer_id").alias("total_spend_to_date"),
        ])
    )

    # Tính Recency (khoảng cách ngày so với đơn trước đó)
    orders_rfm = orders_rfm.with_columns([
        (pl.col("order_date") - pl.col("order_date").shift(1).over("customer_id")).dt.total_days().fill_null(0).alias("days_since_last")
    ])

    # 3. Phân loại Khách hàng (Whales/Newbies)
    # Định nghĩa tạm: Đơn đầu là Newbie, >= 5 đơn hoặc chi > 500k là Whale (có thể tuỳ chỉnh)
    orders_rfm = orders_rfm.with_columns([
        pl.when(pl.col("total_orders_to_date") == 1).then(pl.lit("Newbie"))
        .when((pl.col("total_orders_to_date") >= 5) | (pl.col("total_spend_to_date") > 5000000)).then(pl.lit("Whale"))
        .otherwise(pl.lit("Loyal")).alias("customer_segment")
    ])

    # 4. Join tổng lực để ra Category Level
    final_agg = (
        items
        .join(prods, on="product_id", how="left")
        .join(orders_rfm, on="order_id", how="left")
        .group_by(["order_date", "category"])
        .agg([
            pl.col("item_revenue").sum().alias("cat_revenue"),
            pl.col("customer_id").n_unique().alias("unique_buyers"),
            pl.col("customer_segment").filter(pl.col("customer_segment") == "Whale").count().alias("whale_interactions"),
            pl.col("customer_segment").filter(pl.col("customer_segment") == "Newbie").count().alias("newbie_interactions"),
            pl.col("customer_segment").filter(pl.col("customer_segment") == "Loyal").count().alias("loyal_interactions"),
        ])
    )

    print("⏳ Đang tổng hợp hàng chục triệu dòng dữ liệu...")
    final_df = final_agg.sort(["order_date", "category"]).collect()

    out_path = DATA_DIR / "category_daily_rfm.csv"
    final_df.write_csv(out_path)
    
    print(f"✅ Hoàn tất! Đã lưu {final_df.height} dòng tại {out_path}")
    print(final_df.head(10))

if __name__ == "__main__":
    prepare_v19_data()
