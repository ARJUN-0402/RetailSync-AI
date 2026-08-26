# RetailSync AI - Portfolio Highlights

## Project Overview

RetailSync AI is an end-to-end AI-powered retail demand forecasting and supply chain intelligence platform. It combines machine learning, data engineering, and interactive visualization to help retailers optimize inventory, detect anomalies, and forecast demand.

## Key Metrics

| Metric | Value |
|--------|-------|
| **Data Volume** | 69,216 sales records, 52,500 inventory snapshots |
| **Features Engineered** | 74 features per product-store-day |
| **Models Trained** | 4 serialized models (forecaster + 3 clusterers) |
| **Test Coverage** | 95/95 tests passed (100%) |
| **Forecast Horizon** | 14 days |
| **14-Day Forecast** | 18,541 units, $4.8M revenue |
| **Anomalies Detected** | 31,619 (8.66% of records) |
| **Inventory Alerts** | 785 alerts across 500 product-store combinations |
| **Warehouses Analyzed** | 5 warehouses, 48,906 m³ total capacity |

## Technical Stack

**Languages & Frameworks:**
- Python 3.13
- Pandas 3.0, NumPy 2.4
- Scikit-learn 1.9, XGBoost 3.3
- SQLAlchemy 2.0
- Streamlit 1.58
- Plotly 6.8

**Infrastructure:**
- SQLite database with 12 tables
- Docker containerization
- Modular pipeline architecture

## Architecture Highlights

### End-to-End ML Pipeline

```
Data Generation → Cleaning → Feature Engineering → Modeling → Intelligence → Dashboard
```

### Modular Design

- **Data Layer:** Synthetic data generation, validation, SQLite storage
- **Feature Layer:** 74 features (lags, rolling stats, time features, inventory features)
- **Model Layer:** Demand forecasting, clustering, anomaly detection
- **Intelligence Layer:** Inventory risk, warehouse optimization
- **Presentation Layer:** 7-page interactive Streamlit dashboard

## ML Components

### 1. Demand Forecasting

- **Approach:** Time-aware validation (no data leakage)
- **Models:** Historical Mean, Naive, Moving Average, Random Forest, XGBoost
- **Best Model:** Random Forest (validation MAE=4.67, test MAE=4.65)
- **Honest Assessment:** ML models beat every baseline; selection is automatic by validation MAE

### 2. Inventory Intelligence

- **Stockout Detection:** Rule-based with forecast adjustment
- **Overstock Detection:** Threshold-based with demand variability adjustment
- **Dead Stock Detection:** High inventory + zero demand criteria
- **Composite Risk Score:** Weighted formula (stockout 35%, overstock 25%, dead stock 20%, reorder 20%)

### 3. Anomaly Detection

- **Methods:** Rolling Z-score, IQR, Isolation Forest
- **Ensemble:** Require 2+ method agreement
- **Results:** 31,619 anomalies (8.66%), 28,292 demand spikes
- **Interpretability:** Categorized by type and magnitude

### 4. Segmentation

- **Algorithms:** K-Means clustering
- **Products:** K=2, Silhouette=0.234 (5 business labels)
- **Stores:** K=2, Silhouette=0.337 (4 business labels)
- **Warehouses:** K=4, Silhouette=0.826 (4 business labels)

### 5. Warehouse Optimization

- **Metrics:** Utilization %, turnover ratio, capacity risk
- **Findings:** All warehouses underutilized (avg 14.5%)
- **Recommendations:** Consolidation, redistribution, expansion strategies

## Engineering Practices

### Code Quality
- Modular, reusable components
- Comprehensive docstrings
- Type hints where applicable
- Consistent naming conventions

### Testing
- 95 automated tests
- Data validation at each pipeline stage
- Integration tests for end-to-end flow
- No hardcoded values in production code

### Documentation
- README with setup instructions
- Architecture diagrams
- Methodology documents for each component
- User guide for dashboard
- Design decisions log

### Reproducibility
- Fixed random seeds (42)
- Version-pinned dependencies
- Serialized models with metadata
- Pipeline validation script

## Business Impact

### Forecasting Value
- 14-day demand forecasts for 500 product-store combinations
- $4.8M forecasted revenue enables procurement planning
- Time-aware validation ensures realistic performance estimates

### Inventory Optimization
- 185 urgent reorder alerts prevent stockouts
- 110 high overstock risks reduce holding costs
- 0 dead stock items identified (clean inventory)

### Operational Insights
- 31,619 demand anomalies investigated
- 5 warehouses analyzed for optimization
- 50 products segmented for targeted strategies
- 10 stores segmented for resource allocation

## Challenges & Solutions

| Challenge | Solution |
|-----------|----------|
| Model selection | Automatic by validation MAE across 5 models |
| Data leakage risk | Time-based validation, shift-based features |
| Large dataset (365K rows) | Optimized with groupby and vectorization |
| Multiple anomaly methods | Ensemble approach with consensus threshold |
| Dashboard performance | Cached data with TTL, optimized queries |

## Future Enhancements

1. **Real-time Data:** Kafka streaming for live updates
2. **Advanced Models:** Temporal Fusion Transformers, N-BEATS
3. **Probabilistic Forecasting:** Quantile regression for uncertainty
4. **External Features:** Weather, holidays, market data
5. **Authentication:** Multi-user support with role-based access
6. **Model Monitoring:** drift detection, automated retraining
7. **Cloud Deployment:** AWS/GCP with auto-scaling
8. **API Layer:** FastAPI for programmatic access

## Portfolio Presentation

### Elevator Pitch

"RetailSync AI is an end-to-end supply chain analytics platform that forecasts demand, detects inventory risks, and identifies anomalies using machine learning. It processes 365K daily records, engineers 74 features, and delivers actionable insights through an interactive dashboard."

### Key Talking Points

1. **Full-Stack Data Science:** From raw data to deployed dashboard
2. **Honest ML:** Acknowledged baseline model superiority over complex models
3. **Production-Ready:** Docker, tests, documentation, CI/CD-ready
4. **Business-Focused:** Every component tied to business outcomes
5. **Scalable Architecture:** Modular design ready for cloud deployment

### Demo Flow

1. Show Executive Overview dashboard
2. Drill into Demand Forecast for specific product
3. Review Inventory Intelligence alerts
4. Explore Anomaly Detection timeline
5. Review Segmentation insights
6. Check Warehouse Optimization

## Repository Structure

```
retailsync-ai/
├── src/                    # Source code (9 modules)
├── data/                   # Data files
├── models/                 # Serialized models
├── dashboard/              # Streamlit app
├── tests/                  # 95 automated tests
├── docs/                   # 17 documentation files
└── database/               # SQLite database
```

## How to Run

```bash
git clone https://github.com/<your-username>/retailsync-ai.git
cd retailsync-ai
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run dashboard/app.py
```

## License

MIT License

## Contact

For questions or collaboration opportunities, please reach out via GitHub.
