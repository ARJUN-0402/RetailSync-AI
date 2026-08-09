# RetailSync AI - Dashboard User Guide

## Overview

The RetailSync AI dashboard is an interactive web application built with Streamlit that provides visual access to all analytics outputs. The dashboard features a dark, modern, enterprise-grade design with 7 main pages.

## Accessing the Dashboard

```bash
# From the project root directory
streamlit run dashboard/app.py
```

The dashboard will open in your default web browser at `http://localhost:8501`.

## Dashboard Pages

### 1. Executive Overview

**Purpose:** High-level summary of business metrics and pipeline status.

**Features:**
- KPI cards showing total revenue, units sold, products, stores
- 14-day forecast summary
- Stockout and anomaly counts
- ML model status indicators
- Business problem description
- Engineering approach overview
- Measurable outcomes

**Use Cases:**
- Executive briefings
- Pipeline health checks
- Quick business review

### 2. Demand Forecast

**Purpose:** Explore demand forecasts for specific product-store combinations.

**Features:**
- Product and store selection dropdowns
- Forecast horizon slider (1-14 days)
- Time series chart of forecasted demand
- Historical vs forecasted comparison
- Summary statistics (total demand, revenue, average)

**Use Cases:**
- Inventory planning
- Revenue projection
- Demand pattern analysis

**How to Use:**
1. Select a product from the dropdown
2. Select a store from the dropdown
3. Adjust the forecast horizon slider
4. View the forecast chart and statistics

### 3. Inventory Intelligence

**Purpose:** Monitor inventory risks across all product-store combinations.

**Features:**
- Risk distribution pie charts (stockout and overstock)
- Critical items table
- Inventory alerts table with filtering
- Risk level badges (HIGH, MEDIUM, LOW)

**Use Cases:**
- Stockout prevention
- Overstock reduction
- Reorder planning

**How to Use:**
1. Review the risk distribution charts
2. Examine critical items requiring immediate action
3. Filter alerts by type (Stockout Risk, Overstock Risk, etc.)
4. Check recommended actions for each alert

### 4. Demand Anomalies

**Purpose:** Identify and investigate unusual demand patterns.

**Features:**
- Anomaly timeline bar chart
- Top anomalous products chart
- Anomaly details table with filtering
- Z-score and anomaly type display

**Use Cases:**
- Root cause analysis
- Promotion effectiveness tracking
- Supply chain disruption detection

**How to Use:**
1. Review the daily anomaly count timeline
2. Identify top anomalous products
3. Filter by anomaly type (Demand Spike, Unusual Pattern)
4. Drill down into specific anomaly records

### 5. Segmentation

**Purpose:** Explore product, store, and warehouse segments.

**Features:**
- Segmentation type selector (Products, Stores, Warehouses)
- Cluster distribution pie charts
- Scatter plots for cluster visualization
- Segment details table

**Use Cases:**
- Targeted marketing
- Resource allocation
- Inventory policy optimization

**How to Use:**
1. Select segmentation type
2. Review cluster distribution
3. Analyze cluster characteristics in scatter plots
4. Export segment data for further analysis

### 6. Warehouse Intelligence

**Purpose:** Monitor warehouse utilization and capacity.

**Features:**
- Utilization metrics cards
- Bar chart of utilization percentages
- Warehouse details table with recommendations

**Use Cases:**
- Capacity planning
- Cost optimization
- Network redesign

**How to Use:**
1. Review overall utilization metrics
2. Identify underutilized or overutilized warehouses
3. Check recommendations for each warehouse
4. Plan capacity adjustments

### 7. Data Explorer

**Purpose:** Browse raw data tables and run quick queries.

**Features:**
- Table selector dropdown
- Interactive data table with sorting
- Quick statistics (shape, columns)

**Use Cases:**
- Data validation
- ad-hoc analysis
- Data quality checks

**How to Use:**
1. Select a table from the dropdown
2. Browse the data in the interactive table
3. Check quick statistics at the bottom

## Navigation

- Use the sidebar radio buttons to switch between pages
- The sidebar also shows the technology stack and version information
- Dashboard state persists while navigating between pages

## Interpreting Results

### Risk Levels

- **HIGH (Red):** Immediate action required
- **MEDIUM (Orange):** Action needed soon
- **LOW (Green):** Monitor, no immediate action

### Forecast Confidence

- Forecasts are based on historical averages
- Higher confidence for high-volume products
- Lower confidence for zero-inflated, low-volume items

### Anomaly Severity

- **Z > 3:** Significant spike/drop
- **Z > 2:** Unusual activity
- **Ensemble agreement:** 2+ methods must agree for high confidence

## Troubleshooting

### Dashboard won't start

```bash
# Ensure virtual environment is activated
venv\Scripts\activate  # Windows
source venv/bin/activate  # macOS/Linux

# Check Streamlit installation
pip install streamlit

# Run with explicit port
streamlit run dashboard/app.py --server.port 8501
```

### Data not loading

- Ensure all pipeline scripts have been run
- Check that `data/processed/` contains CSV files
- Verify `database/retailsync.db` exists

### Slow performance

- Data is cached for 5 minutes; wait for cache to refresh
- Large CSV files may take time to load initially
- Consider reducing dataset size for faster iteration

## Tips for Effective Use

1. **Start with Executive Overview** to understand the big picture
2. **Use Demand Forecast** for specific product-store planning
3. **Check Inventory Intelligence** daily for stockout prevention
4. **Review Anomalies** weekly to identify trends
5. **Explore Segmentation** for strategic planning
6. **Monitor Warehouses** for capacity optimization
7. **Use Data Explorer** for ad-hoc analysis

## Keyboard Shortcuts

- `R` — Rerun the dashboard
- `C` — Clear cache
- `Ctrl+C` — Stop the dashboard (in terminal)

## Feedback

For issues or feature requests, please open an issue on GitHub.
