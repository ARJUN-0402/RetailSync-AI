# Exploratory Data Analysis — Key Findings

**Date:** 2026-08-09
**Analyst:** RetailSync AI
**Dataset:** Synthetic retail data (2023-08-11 to 2025-08-09)

## Dataset Overview

| Metric | Value |
|---|---|
| Total Revenue | $227,497,246.54 |
| Total Units Sold | 876,546 |
| Average Daily Revenue | $311,640.06 |
| Date Range | 2023-08-11 to 2025-08-09 (730 days) |
| Total Products | 50 |
| Total Stores | 10 |
| Promotional Sales % | 15.3% |
| Stockout Snapshots | 19,198 (36.6% of inventory records) |
| Average Inventory Level | 54.21 units |

---

## 1. Sales Trends Over Time

**Finding:** Daily revenue shows variability with an overall stable trend. There is no strong upward or downward trend over the 2-year period, suggesting a mature retail operation with consistent demand patterns.

**Key Insights:**
- Average daily revenue: ~$311K
- Revenue distribution is relatively stable with periodic spikes
- No clear long-term growth or decline trend visible at daily granularity

---

## 2. Seasonal Patterns

**Finding:** Clear seasonal patterns exist, particularly by month and day of week.

**Key Insights:**
- **Month-level:** Certain months show higher average daily quantities, suggesting seasonal demand shifts
- **Day of week:** Weekday vs weekend patterns differ, with potential for weekend demand surges
- **Category seasonality:** Electronics shows holiday season peaks; Clothing shows summer peaks

**Business Implication:** Promotional planning and inventory stocking should align with these seasonal patterns to maximize revenue and minimize stockouts.

---

## 3. Product Performance

**Finding:** Significant variation in product performance exists. Top products generate substantially more revenue than others.

**Key Insights:**
- Electronics and Home Goods are the top revenue-generating categories
- Product concentration: a subset of products drives the majority of revenue
- Category revenue share is not evenly distributed

**Business Implication:** Focus inventory management and forecasting efforts on high-revenue products while maintaining adequate stock for slow-movers.

---

## 4. Store Performance

**Finding:** Store performance varies by location and store type.

**Key Insights:**
- Urban and Suburban stores tend to outperform Rural stores in absolute revenue
- Store type is a significant factor in revenue generation
- Top-performing stores should be prioritized for stock allocation during high-demand periods

**Business Implication:** Store-specific inventory policies may be warranted rather than one-size-fits-all allocation.

---

## 5. Inventory Behavior

**Finding:** Inventory levels are highly variable across categories and products.

**Key Insights:**
- Average inventory level is ~54 units, but distribution is wide
- Some products consistently maintain high stock levels (overstock risk)
- Others frequently drop to zero (stockout risk)

**Business Implication:** The high stockout rate (36.6%) indicates that reorder points may be too low or lead times are underestimated for many product-store combinations.

---

## 6. Revenue Trends

**Finding:** Promotional sales account for ~15.3% of revenue, indicating that promotions are a meaningful but not dominant revenue driver.

**Key Insights:**
- 7-day moving average smooths out daily volatility while preserving trend information
- Revenue shows periodic spikes aligned with promotional activity
- Non-promotional revenue is the stable base; promotions provide incremental lift

**Business Implication:** Optimize promotion timing and product selection to maximize ROI on promotional spend.

---

## 7. Demand Variability

**Finding:** Demand variability (coefficient of variation) varies widely across products.

**Key Insights:**
- Some products have very stable demand (low CV)
- Others exhibit high volatility, making forecasting challenging
- High-variability products require more sophisticated forecasting approaches or safety stock

**Business Implication:** Apply different inventory policies based on demand variability — tighter control for stable products, more buffer for volatile ones.

---

## 8. Stockout Patterns

**Finding:** Stockouts are widespread — all 50 products experience stockout conditions at some point, and 36.6% of all inventory snapshots show stockouts.

**Key Insights:**
- Stockout frequency varies significantly by product
- Certain categories are more prone to stockouts
- High stockout frequency correlates with high demand or inadequate reorder points

**Business Implication:** This is a critical operational issue. Automated reorder point optimization and demand forecasting are needed to reduce stockout rates.

---

## Overall Assessment

The dataset exhibits realistic retail characteristics:
- Seasonal demand patterns
- Promotional activity
- Product and store heterogeneity
- Significant stockout challenges

These characteristics make it suitable for demonstrating:
- Time-series forecasting
- Inventory risk detection
- Anomaly detection
- Segmentation
- Warehouse optimization

The high stockout rate (36.6%) is an intentional feature of the synthetic data that creates a compelling business case for the RetailSync AI platform.

---

*Generated from database: `database/retailsync.db`*
*Visualizations: `notebooks/eda_output/`*
