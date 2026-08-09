# RetailSync AI - Design Decisions

## Overview

This document captures key design decisions made during the development of RetailSync AI, including trade-offs, alternatives considered, and rationale.

## 1. Synthetic Data vs Real Data

**Decision:** Use synthetic data instead of real retail data.

**Rationale:**
- No access to proprietary retail datasets
- Ability to control data characteristics (zero-inflation, seasonality)
- Reproducibility for portfolio demonstration
- Avoids data privacy and licensing issues

**Trade-offs:**
- Pro: Full control over data generation, reproducible results
- Con: May not capture real-world complexity and noise

## 2. SQLite vs PostgreSQL

**Decision:** Use SQLite for development and demonstration.

**Rationale:**
- Zero configuration required
- Single file for easy sharing
- Sufficient performance for 50 products × 10 stores
- SQLAlchemy 2.0 compatible

**Trade-offs:**
- Pro: Simple setup, portable
- Con: Not suitable for concurrent writes or production scale

**Future:** Migrate to PostgreSQL for production deployment.

## 3. Time-Based Train/Validation/Test Split

**Decision:** Use time-based split instead of random split.

**Rationale:**
- Prevents data leakage in time-series forecasting
- Simulates real-world forecasting scenario
- Industry standard for demand forecasting

**Split:**
- Train: 2023-08-11 to 2024-12-31 (254,500 rows)
- Validation: 2025-01-01 to 2025-06-09 (80,000 rows)
- Test: 2025-06-10 to 2025-08-09 (30,500 rows)

## 4. Baseline Mean as Best Forecasting Model

**Decision:** Use historical mean as the production forecasting model.

**Rationale:**
- High zero-inflation (81%) makes ML models struggle
- Baseline Mean achieved lowest MAE (4.09) on validation set
- Honest assessment of model performance
- Avoids overfitting to noise

**Trade-offs:**
- Pro: Simple, interpretable, no overfitting
- Con: Lower accuracy than potential advanced models

**Future:** Implement zero-inflated models (Temporal Fusion Transformers, N-BEATS).

## 5. Ensemble Anomaly Detection

**Decision:** Use ensemble of Z-score, IQR, and Isolation Forest.

**Rationale:**
- Each method captures different anomaly patterns
- Reduces false positives through consensus
- Z-score: univariate spikes/drops
- IQR: distribution-based outliers
- Isolation Forest: multivariate anomalies

**Threshold:** Require 2+ methods to agree for "Anomaly" classification.

## 6. K-Means for Clustering

**Decision:** Use K-Means for product, store, and warehouse segmentation.

**Rationale:**
- Simple and fast
- Easy to interpret clusters
- Business-aligned labels
- Sufficient for small datasets (50, 10, 5 entities)

**Alternatives Considered:**
- DBSCAN: Too sensitive to parameters
- Hierarchical: Slow for larger datasets
- Gaussian Mixture: Overkill for this use case

**Optimal K:**
- Products: K=2 (Silhouette=0.234)
- Stores: K=2 (Silhouette=0.337)
- Warehouses: K=4 (Silhouette=0.826)

## 7. 74 Features for Forecasting

**Decision:** Engineer 74 features including lags, rolling stats, time features, inventory, and price features.

**Rationale:**
- Captures multiple aspects of demand behavior
- Time-aware validation ensures no data leakage
- Balance between complexity and interpretability

**Feature Categories:**
- Lag features: 8
- Rolling features: 12
- Expanding features: 2
- Time features: 12
- Price features: 2
- Promotion features: 3
- Inventory features: 7
- Demand variability: 2
- Aggregate features: 2
- Targets: 6

## 8. Streamlit for Dashboard

**Decision:** Use Streamlit instead of React or other frontend frameworks.

**Rationale:**
- Rapid development (Python-only)
- Seamless integration with Pandas and Plotly
- Easy deployment (`streamlit run`)
- Suitable for single-user analytics

**Trade-offs:**
- Pro: Fast development, Python ecosystem
- Con: Limited customization, not ideal for production multi-user

**Future:** Consider FastAPI + React for production deployment.

## 9. Dark Enterprise Theme

**Decision:** Use dark theme with cyan accents for dashboard.

**Rationale:**
- Modern, professional appearance
- Reduces eye strain for extended use
- High contrast for data visualization
- Popular in enterprise analytics tools

**Colors:**
- Background: #0e1117
- Sidebar: #1a1d23
- Accent: #00d4ff (cyan)
- Risk colors: Red (#ff4757), Orange (#ffa502), Green (#2ed573)

## 10. No Real-Time Data

**Decision:** Use batch processing instead of real-time streaming.

**Rationale:**
- Synthetic data is static
- Portfolio demonstration doesn't require real-time
- Simpler architecture and testing

**Future:** Add Kafka or similar for real-time ingestion.

## 11. No External Features

**Decision:** Exclude weather, holidays, and market data.

**Rationale:**
- Synthetic data doesn't include these features
- Keeps the project focused
- Avoids external API dependencies

**Future:** Add holiday calendar and weather API integration.

## 12. Joblib for Model Serialization

**Decision:** Use joblib instead of pickle or ONNX.

**Rationale:**
- Efficient for scikit-learn models
- Handles large numpy arrays efficiently
- Simple API

**Trade-offs:**
- Pro: Fast, Python-native
- Con: Not language-agnostic

**Future:** Consider ONNX for model portability.

## 13. SQLAlchemy 2.0

**Decision:** Use SQLAlchemy 2.0 for database access.

**Rationale:**
- Modern async support (future-proofing)
- Improved type hints and IDE support
- Better performance than 1.x
- Industry standard ORM

## 14. Weekly Inventory Snapshots

**Decision:** Use weekly inventory snapshots instead of daily.

**Rationale:**
- Reduces dataset size (52,500 vs 365,000 records)
- More realistic for retail operations
- Forward-filling for daily estimates

**Trade-offs:**
- Pro: Smaller dataset, realistic cadence
- Con: Less granular inventory visibility

## 15. Composite Risk Score Weights

**Decision:** Use weighted composite score for inventory risk.

**Formula:**
```
composite_risk_score = (
    stockout_score * 0.35 +
    overstock_score * 0.25 +
    dead_stock_score * 0.20 +
    reorder_score * 0.20
)
```

**Rationale:**
- Stockouts weighted highest (35%) due to revenue impact
- Overstock and reorder equally important (25% and 20%)
- Dead stock included but lower weight (20%)
