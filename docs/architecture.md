# RetailSync AI - System Architecture

## Overview

RetailSync AI is built using a modular, layered architecture that separates concerns and enables independent development and testing of each component.

## Architecture Layers

```
┌─────────────────────────────────────────────────────────────┐
│                    Presentation Layer                        │
│                  (Streamlit Dashboard)                       │
├─────────────────────────────────────────────────────────────┤
│                    Intelligence Layer                        │
│    (Forecasting, Anomaly Detection, Inventory, Clustering)  │
├─────────────────────────────────────────────────────────────┤
│                      Model Layer                             │
│         (Serialized models, Feature pipelines)              │
├─────────────────────────────────────────────────────────────┤
│                      Feature Layer                           │
│        (Feature engineering, transformations)               │
├─────────────────────────────────────────────────────────────┤
│                       Data Layer                             │
│     (Raw data, Cleaned data, Database, SQLAlchemy)          │
└─────────────────────────────────────────────────────────────┘
```

## Component Diagram

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Raw Data   │────▶│   Cleaning   │────▶│   Features   │
│  (CSV files) │     │   Pipeline   │     │   (74 cols)  │
└──────────────┘     └──────────────┘     └──────┬───────┘
                                                 │
                    ┌──────────────┐             │
                    │  SQLAlchemy  │◀────────────┘
                    │   (ORM)      │
                    └──────┬───────┘
                           │
              ┌────────────┴────────────┐
              │                         │
    ┌─────────▼─────────┐     ┌─────────▼─────────┐
    │  Demand Forecaster │     │  Inventory Risk   │
    │   (Baseline Mean)  │     │   Detection       │
    └─────────┬─────────┘     └─────────┬─────────┘
              │                         │
    ┌─────────▼─────────┐     ┌─────────▼─────────┐
    │  Anomaly Detection│     │   Clustering      │
    │ (Z-score + IQR +  │     │  (K-Means K=2,4)  │
    │  Isolation Forest)│     └─────────┬─────────┘
    └─────────┬─────────┘               │
              │                         │
              └────────────┬────────────┘
                           │
              ┌────────────▼────────────┐
              │   Warehouse Optimization │
              │   (Utilization Analysis) │
              └────────────┬────────────┘
                           │
              ┌────────────▼────────────┐
              │   Dashboard (Streamlit)  │
              │   7 Interactive Pages    │
              └─────────────────────────┘
```

## Data Flow

### 1. Data Ingestion

```
Raw CSV files
    ↓ [generate_dataset.py]
Synthetic data (69,216 sales, 52,500 inventory)
    ↓ [ingest.py]
Cleaned data with validation
    ↓ [init_db.py]
SQLite database (retailsync.db)
```

### 2. Feature Engineering

```
Database tables
    ↓ [feature_engineering.py]
Daily product-store level dataset (365,000 rows)
    ├── Lag features (1d, 7d, 14d, 28d)
    ├── Rolling statistics (7d, 14d, 28d)
    ├── Expanding statistics
    ├── Time features (cyclical encoding)
    ├── Price and promotion features
    ├── Inventory features
    ├── Demand variability features
    └── Target variables (1d, 7d, 14d ahead)
```

### 3. Model Training

```
Features + Targets
    ↓ [demand_forecaster.py]
Train/Val/Test split (time-based)
    ↓
Baseline models (Mean, Naive, MA)
    ↓
ML models (Random Forest, XGBoost)
    ↓
Best model: Baseline Mean
    ↓ [joblib.dump]
models/demand_forecaster.pkl
```

### 4. Intelligence Generation

```
Features
    ├── [forecast_pipeline.py] → 14-day forecasts (7,000 rows)
    ├── [inventory_intelligence.py] → Risk alerts (785 alerts)
    ├── [anomaly_detection.py] → Anomalies (31,619 records)
    ├── [segmentation.py] → Product/Store/Warehouse clusters
    └── [warehouse_optimization.py] → Utilization analysis (5 warehouses)
```

### 5. Dashboard

```
All outputs + Models
    ↓ [load_data()]
DataFrames loaded into memory
    ↓
7 interactive pages:
    ├── Executive Overview
    ├── Demand Forecast
    ├── Inventory Intelligence
    ├── Demand Anomalies
    ├── Segmentation
    ├── Warehouse Intelligence
    └── Data Explorer
```

## Technology Decisions

### Why SQLite?

- **Simplicity:** No separate server process required
- **Portability:** Single file database, easy to share
- **Performance:** Sufficient for 50 products × 10 stores × 730 days
- **Compatibility:** SQLAlchemy 2.0 support

### Why Streamlit?

- **Rapid Development:** Python-only, no frontend framework needed
- **Interactivity:** Built-in widgets for filtering and exploration
- **Deployment:** Easy to deploy with `streamlit run`
- **Integration:** Works seamlessly with Pandas and Plotly

### Why Baseline Mean for Forecasting?

- **Data Characteristics:** 81% zero-inflation makes ML models struggle
- **Honest Assessment:** Baseline outperforms XGBoost and Random Forest
- **Business Value:** Even simple forecasts provide value when combined with other intelligence

### Why K-Means for Clustering?

- **Simplicity:** Easy to interpret and implement
- **Scalability:** Fast even with larger datasets
- **Business Alignment:** Clear cluster definitions with labels
- **Alternatives Considered:** DBSCAN (too sensitive to parameters), Hierarchical (slow for large datasets)

### Why Ensemble Anomaly Detection?

- **Robustness:** Combining methods reduces false positives
- **Interpretability:** Each method provides different insights
- **Coverage:** Z-score captures univariate spikes, Isolation Forest captures multivariate patterns
- **Consensus:** Requiring 2+ method agreement ensures high-confidence anomalies

## Database Schema

```sql
-- Core entities
products (50 rows)
stores (10 rows)
suppliers (8 rows)
warehouses (5 rows)

-- Transactions
sales (69,216 rows)
inventory (52,500 rows)

-- Analytics
inventory_alerts (785 rows)
anomaly_flags (31,619 rows)
product_segments (50 rows)
store_segments (10 rows)
warehouse_segments (5 rows)
warehouse_optimization (5 rows)
```

## Model Artifacts

```
models/
├── demand_forecaster.pkl    # Baseline Mean model + metadata
├── product_clusterer.pkl    # K-Means (K=2) + scaler
├── store_clusterer.pkl      # K-Means (K=2) + scaler
└── warehouse_clusterer.pkl  # K-Means (K=4) + scaler
```

Each model package contains:
- Trained model object
- Feature columns list
- Scaler (for clustering)
- Metadata (metrics, training date, etc.)

## Caching Strategy

- **@st.cache_resource:** Database engine, loaded models (persist across reruns)
- **@st.cache_data:** DataFrames with TTL=300s (refresh periodically)

## Security Considerations

- No sensitive data in the repository
- Database file excluded from Git
- Environment variables for configuration
- No authentication implemented (single-user local deployment)

## Scalability Considerations

### Current Limitations

- SQLite: Not suitable for concurrent writes
- In-memory DataFrames: Limited by available RAM
- Streamlit: Single-user sessions

### Scaling Path

1. **Database:** Migrate to PostgreSQL for concurrent access
2. **Processing:** Use Dask or Spark for larger datasets
3. **Deployment:** Deploy Streamlit on AWS/GCP with load balancer
4. **Real-time:** Add Kafka for streaming data ingestion
5. **Models:** Implement model versioning with MLflow

## Design Patterns

- **Repository Pattern:** Database access abstracted through SQLAlchemy
- **Factory Pattern:** Model loading and instantiation
- **Pipeline Pattern:** Sequential data processing stages
- **Observer Pattern:** Dashboard updates when data changes
- **Strategy Pattern:** Multiple anomaly detection methods

## Error Handling

- Data validation at each pipeline stage
- Graceful degradation when data is missing
- Comprehensive logging for debugging
- Test suite catches regressions

## Performance

- Feature engineering: ~60 seconds for 365,000 rows
- Model training: ~3 seconds for XGBoost, ~78 seconds for Random Forest
- Anomaly detection: ~5 seconds for ensemble method
- Dashboard load: ~2-3 seconds with caching
