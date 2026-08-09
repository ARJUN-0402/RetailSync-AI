import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from sqlalchemy import create_engine

plt.style.use("seaborn-v0_8-darkgrid")
OUTPUT_DIR = "notebooks/eda_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

engine = create_engine("sqlite:///database/retailsync.db")

def save_plotly(fig, name):
    path = os.path.join(OUTPUT_DIR, f"{name}.html")
    fig.write_html(path)
    print(f"Saved: {path}")

def save_matplotlib(fig, name):
    path = os.path.join(OUTPUT_DIR, f"{name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")

# Load data
products = pd.read_sql("SELECT * FROM products", engine)
stores = pd.read_sql("SELECT * FROM stores", engine)
sales = pd.read_sql("SELECT * FROM sales", engine)
inventory = pd.read_sql("SELECT * FROM inventory", engine)

sales["date"] = pd.to_datetime(sales["date"])
inventory["date"] = pd.to_datetime(inventory["date"])
products["launch_date"] = pd.to_datetime(products["launch_date"])

print("=== DATA SHAPES ===")
print(f"Products: {products.shape}")
print(f"Stores: {stores.shape}")
print(f"Sales: {sales.shape}")
print(f"Inventory: {inventory.shape}")

# ============================================================
# 1. SALES TRENDS OVER TIME
# ============================================================
print("\n=== 1. SALES TRENDS OVER TIME ===")
daily_sales = sales.groupby("date").agg(
    total_quantity=("quantity_sold", "sum"),
    total_revenue=("revenue", "sum"),
    avg_price=("unit_price", "mean"),
    unique_products=("product_id", "nunique"),
).reset_index()

fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_trace(go.Scatter(x=daily_sales["date"], y=daily_sales["total_revenue"], name="Revenue", line=dict(color="#00d4ff")), secondary_y=False)
fig.add_trace(go.Scatter(x=daily_sales["date"], y=daily_sales["total_quantity"], name="Quantity", line=dict(color="#ff6b6b")), secondary_y=True)
fig.update_layout(title_text="Daily Sales Trend (Revenue vs Quantity)", template="plotly_dark", height=500)
fig.update_xaxes(title_text="Date")
fig.update_yaxes(title_text="Revenue ($)", secondary_y=False)
fig.update_yaxes(title_text="Quantity Sold", secondary_y=True)
save_plotly(fig, "01_sales_trend_daily")

# Monthly aggregation
daily_sales["month"] = daily_sales["date"].dt.to_period("M").astype(str)
monthly_sales = daily_sales.groupby("month").agg(
    total_revenue=("total_revenue", "sum"),
    total_quantity=("total_quantity", "sum"),
).reset_index()

fig = px.bar(monthly_sales, x="month", y="total_revenue", title="Monthly Revenue", template="plotly_dark", color="total_revenue", color_continuous_scale="Blues")
save_plotly(fig, "01b_sales_trend_monthly")

# ============================================================
# 2. SEASONAL PATTERNS
# ============================================================
print("\n=== 2. SEASONAL PATTERNS ===")
sales["month"] = sales["date"].dt.month
sales["day_of_week"] = sales["date"].dt.day_name()
sales["quarter"] = sales["date"].dt.quarter

monthly_avg = sales.groupby("month")["quantity_sold"].mean().reset_index()
fig = px.bar(monthly_avg, x="month", y="quantity_sold", title="Average Daily Quantity by Month", labels={"month": "Month", "quantity_sold": "Avg Quantity"}, template="plotly_dark", color="quantity_sold", color_continuous_scale="Viridis")
save_plotly(fig, "02_seasonality_monthly")

dow_avg = sales.groupby("day_of_week")["quantity_sold"].mean().reindex([
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"
]).reset_index()
fig = px.bar(dow_avg, x="day_of_week", y="quantity_sold", title="Average Daily Quantity by Day of Week", labels={"day_of_week": "Day", "quantity_sold": "Avg Quantity"}, template="plotly_dark", color="quantity_sold", color_continuous_scale="Viridis")
save_plotly(fig, "02b_seasonality_dow")

# Category seasonality heatmap
sales_with_cat = sales.merge(products[["product_id", "category"]], on="product_id", how="left")
cat_month = sales_with_cat.groupby(["category", "month"])["quantity_sold"].mean().reset_index()
cat_month_pivot = cat_month.pivot(index="category", columns="month", values="quantity_sold")
fig = px.imshow(cat_month_pivot, labels=dict(x="Month", y="Category", color="Avg Quantity"), title="Category Seasonality Heatmap", template="plotly_dark", color_continuous_scale="RdBu")
save_plotly(fig, "02c_category_seasonality_heatmap")

# ============================================================
# 3. PRODUCT PERFORMANCE
# ============================================================
print("\n=== 3. PRODUCT PERFORMANCE ===")
product_perf = sales.groupby("product_id").agg(
    total_quantity=("quantity_sold", "sum"),
    total_revenue=("revenue", "sum"),
    avg_price=("unit_price", "mean"),
    days_sold=("date", "nunique"),
).reset_index().merge(products[["product_id", "product_name", "category"]], on="product_id")

top_products = product_perf.nlargest(10, "total_revenue")
fig = px.bar(top_products, x="total_revenue", y="product_name", orientation="h", title="Top 10 Products by Revenue", labels={"total_revenue": "Revenue ($)", "product_name": "Product"}, template="plotly_dark", color="category")
save_plotly(fig, "03_top_products_revenue")

category_perf = product_perf.groupby("category").agg(
    total_revenue=("total_revenue", "sum"),
    total_quantity=("total_quantity", "sum"),
    product_count=("product_id", "nunique"),
).reset_index()
fig = px.pie(category_perf, values="total_revenue", names="category", title="Revenue Share by Category", template="plotly_dark")
save_plotly(fig, "03b_category_revenue_share")

# ============================================================
# 4. STORE PERFORMANCE
# ============================================================
print("\n=== 4. STORE PERFORMANCE ===")
store_perf = sales.groupby("store_id").agg(
    total_revenue=("revenue", "sum"),
    total_quantity=("quantity_sold", "sum"),
    avg_order_value=("revenue", "mean"),
    active_days=("date", "nunique"),
).reset_index().merge(stores[["store_id", "store_name", "city", "state", "store_type"]], on="store_id")

fig = px.bar(store_perf, x="total_revenue", y="store_name", orientation="h", title="Store Performance by Revenue", labels={"total_revenue": "Revenue ($)", "store_name": "Store"}, template="plotly_dark", color="store_type")
save_plotly(fig, "04_store_performance")

store_type_perf = store_perf.groupby("store_type").agg(
    total_revenue=("total_revenue", "sum"),
    total_quantity=("total_quantity", "sum"),
    store_count=("store_id", "nunique"),
).reset_index()
fig = px.bar(store_type_perf, x="store_type", y="total_revenue", title="Revenue by Store Type", labels={"store_type": "Store Type", "total_revenue": "Revenue ($)"}, template="plotly_dark", color="store_type")
save_plotly(fig, "04b_store_type_performance")

# ============================================================
# 5. INVENTORY BEHAVIOR
# ============================================================
print("\n=== 5. INVENTORY BEHAVIOR ===")
latest_inv = inventory.loc[inventory.groupby(["product_id", "store_id"])["date"].idxmax()]
latest_inv = latest_inv.merge(products[["product_id", "category"]], on="product_id", how="left")

fig = px.histogram(latest_inv, x="quantity_on_hand", nbins=50, title="Distribution of Current Inventory Levels", labels={"quantity_on_hand": "Quantity On Hand"}, template="plotly_dark", color="category")
save_plotly(fig, "05_inventory_distribution")

inv_summary = latest_inv.groupby("category").agg(
    avg_stock=("quantity_on_hand", "mean"),
    median_stock=("quantity_on_hand", "median"),
    max_stock=("quantity_on_hand", "max"),
).reset_index()
fig = px.bar(inv_summary, x="category", y="avg_stock", title="Average Inventory Level by Category", labels={"category": "Category", "avg_stock": "Avg Stock"}, template="plotly_dark", color="avg_stock", color_continuous_scale="Blues")
save_plotly(fig, "05b_inventory_by_category")

# ============================================================
# 6. REVENUE TRENDS
# ============================================================
print("\n=== 6. REVENUE TRENDS ===")
daily_sales["revenue_7d_ma"] = daily_sales["total_revenue"].rolling(window=7, min_periods=1).mean()
fig = go.Figure()
fig.add_trace(go.Scatter(x=daily_sales["date"], y=daily_sales["total_revenue"], mode="lines", name="Daily Revenue", opacity=0.4, line=dict(color="#00d4ff")))
fig.add_trace(go.Scatter(x=daily_sales["date"], y=daily_sales["revenue_7d_ma"], mode="lines", name="7-Day MA", line=dict(color="#ff6b6b", width=2)))
fig.update_layout(title="Daily Revenue with 7-Day Moving Average", template="plotly_dark", height=500)
save_plotly(fig, "06_revenue_trend_ma")

# Promotional vs non-promotional
promo = sales.groupby("promotion").agg(total_revenue=("revenue", "sum"), total_quantity=("quantity_sold", "sum")).reset_index()
promo["promotion_label"] = promo["promotion"].map({0: "Regular", 1: "Promotional"})
fig = px.pie(promo, values="total_revenue", names="promotion_label", title="Revenue: Promotional vs Regular", template="plotly_dark")
save_plotly(fig, "06b_promotional_revenue_share")

# ============================================================
# 7. DEMAND VARIABILITY
# ============================================================
print("\n=== 7. DEMAND VARIABILITY ===")
daily_product = sales.groupby(["date", "product_id"])["quantity_sold"].sum().reset_index()
variability = daily_product.groupby("product_id")["quantity_sold"].agg(["mean", "std", "count"]).reset_index()
variability["cv"] = variability["std"] / variability["mean"]
variability = variability.merge(products[["product_id", "product_name", "category"]], on="product_id")
variability = variability.sort_values("cv", ascending=False)

fig = px.histogram(variability, x="cv", nbins=40, title="Distribution of Demand Variability (Coefficient of Variation)", labels={"cv": "Coefficient of Variation"}, template="plotly_dark")
save_plotly(fig, "07_demand_variability_dist")

top_cv = variability.head(10)
fig = px.bar(top_cv, x="cv", y="product_name", orientation="h", title="Top 10 Products by Demand Variability", labels={"cv": "CV", "product_name": "Product"}, template="plotly_dark", color="category")
save_plotly(fig, "07b_top_demand_variability")

# ============================================================
# 8. STOCKOUT PATTERNS
# ============================================================
print("\n=== 8. STOCKOUT PATTERNS ===")
stockouts = inventory[inventory["quantity_on_hand"] <= inventory["reorder_point"]].copy()
stockout_freq = stockouts.groupby(["product_id", "store_id"]).size().reset_index(name="stockout_count")
stockout_freq = stockout_freq.merge(products[["product_id", "category"]], on="product_id", how="left")
stockout_summary = stockout_freq.groupby("product_id").agg(
    total_stockouts=("stockout_count", "sum"),
    affected_stores=("store_id", "nunique"),
).reset_index().merge(products[["product_id", "product_name", "category"]], on="product_id").sort_values("total_stockouts", ascending=False)

fig = px.bar(stockout_summary.head(15), x="total_stockouts", y="product_name", orientation="h", title="Top 15 Products by Stockout Frequency", labels={"total_stockouts": "Stockout Snapshots", "product_name": "Product"}, template="plotly_dark", color="category")
save_plotly(fig, "08_stockout_frequency")

# Stockouts by category
stockout_cat = stockout_freq.groupby("category")["stockout_count"].sum().reset_index()
fig = px.pie(stockout_cat, values="stockout_count", names="category", title="Stockout Distribution by Category", template="plotly_dark")
save_plotly(fig, "08b_stockout_by_category")

# ============================================================
# SUMMARY STATISTICS
# ============================================================
print("\n=== SUMMARY STATISTICS ===")
summary = {
    "total_revenue": sales["revenue"].sum(),
    "total_quantity_sold": sales["quantity_sold"].sum(),
    "avg_daily_revenue": daily_sales["total_revenue"].mean(),
    "date_range": f"{sales['date'].min().date()} to {sales['date'].max().date()}",
    "total_products": products["product_id"].nunique(),
    "total_stores": stores["store_id"].nunique(),
    "promotional_sales_pct": sales["promotion"].mean() * 100,
    "stockout_snapshots": len(stockouts),
    "stockout_pct": len(stockouts) / len(inventory) * 100,
    "avg_inventory_level": latest_inv["quantity_on_hand"].mean(),
    "products_with_stockouts": stockout_summary["product_id"].nunique(),
}
for k, v in summary.items():
    print(f"  {k}: {v}")

# Save summary to CSV
pd.DataFrame([summary]).to_csv(os.path.join(OUTPUT_DIR, "eda_summary.csv"), index=False)
print(f"\nSaved summary to {os.path.join(OUTPUT_DIR, 'eda_summary.csv')}")
print("\nEDA complete.")
