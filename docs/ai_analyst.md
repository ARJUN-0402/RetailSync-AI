# AI Analyst Layer

## Architecture

The AI Analyst layer adds a modular, read-only natural-language interface on top of the existing RetailSync AI data and models. It never invents values; every answer is grounded in actual application data.

```
┌─────────────────────────────────────────────────────────────┐
│                    Streamlit AI Analyst Page                 │
│              (chat interface + suggested questions)           │
├─────────────────────────────────────────────────────────────┤
│                    Orchestrator (orchestrator.py)            │
│  - Prompt construction                                       │
│  - Tool-calling loop (max 4 rounds)                          │
│  - RAG context injection                                     │
│  - Grounding enforcement                                     │
├─────────────────────────────────────────────────────────────┤
│                        Tools (tools.py)                      │
│  Read-only data access functions reusing existing modules:   │
│  - get_sales_trends                                          │
│  - get_forecasts                                             │
│  - get_inventory_snapshot                                    │
│  - get_stockout_risks                                        │
│  - get_overstock_risks                                       │
│  - get_anomalies                                             │
│  - get_product_segments                                      │
│  - get_store_segments                                        │
│  - get_warehouse_performance                                 │
│  - get_executive_kpis                                        │
│  - get_reorder_recommendations                               │
│  - get_forecast_explanation                                  │
├─────────────────────────────────────────────────────────────┤
│                    RAG Layer (retriever.py)                  │
│  Lightweight keyword-based retrieval over docs/              │
├─────────────────────────────────────────────────────────────┤
│                    LLM Provider (openai/anthropic/ollama)   │
└─────────────────────────────────────────────────────────────┘
```

## RAG Flow

1. User asks a question.
2. The retriever tokenizes the query and scores all documentation paragraphs from `docs/*.md`.
3. Top-k chunks (default 3) are formatted as context.
4. The orchestrator injects the context into the system prompt.
5. The LLM generates an answer constrained to the retrieved context and tool results.

## Available Tools

| Tool | Description |
|------|-------------|
| `get_sales_trends` | Recent sales revenue and quantity by product/store/date range |
| `get_forecasts` | 14-day demand forecasts with revenue projections |
| `get_inventory_snapshot` | Latest inventory levels, reorder points, max stock |
| `get_stockout_risks` | HIGH/MEDIUM stockout risk items with reasoning |
| `get_overstock_risks` | HIGH/MEDIUM overstock risk items with reasoning |
| `get_anomalies` | Demand spikes, drops, and unusual patterns |
| `get_product_segments` | Product cluster labels (High-Volume, Slow-Moving, etc.) |
| `get_store_segments` | Store cluster labels (High-Performance, Low-Performance, etc.) |
| `get_warehouse_performance` | Warehouse utilization and capacity risk |
| `get_executive_kpis` | Top-level KPIs computed from existing business_metrics modules |
| `get_reorder_recommendations` | Reorder quantities, urgency, and reasoning |
| `get_forecast_explanation` | SHAP-based natural-language explanation for a forecast |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `RETAILSYNC_AI_API_KEY` | (empty) | API key for the LLM provider |
| `RETAILSYNC_AI_PROVIDER` | `openai` | LLM provider (`openai`, `anthropic`, `ollama`) |
| `RETAILSYNC_AI_MODEL` | `gpt-4o-mini` | Model name |
| `RETAILSYNC_AI_BASE_URL` | (empty) | Optional base URL (for Ollama or proxies) |
| `RETAILSYNC_AI_OFFLINE_MODE` | `false` | Run without LLM (rule-based fallback) |
| `RETAILSYNC_AI_DISABLE_TOOLS` | `false` | Disable tool calling |
| `RETAILSYNC_AI_DISABLE_RAG` | `false` | Disable documentation retrieval |

## Supported Questions

The AI Analyst can answer questions about:

- **Stockouts:** "Why are stockouts increasing?" → inspects inventory alerts and trends
- **Overstock:** "Which products are overstocked?" → inspects overstock risk data
- **Reorder:** "Which products should I reorder?" → inspects forecasts, inventory, lead times, safety stock
- **Anomalies:** "Explain this anomaly." → inspects anomaly flags with Z-score, IQR, and Isolation Forest details
- **Inventory risk:** "Which store has the highest inventory risk?" → inspects composite risk scores
- **Demand:** "Why is demand expected to increase next week?" → inspects 14-day forecasts
- **Forecasts:** "What is the 14-day demand forecast?" → returns aggregated forecast totals
- **Warehouses:** "Which warehouses are near capacity?" → inspects utilization and capacity risk
- **Segments:** "Which products are high-volume?" → inspects cluster labels
- **KPIs:** "What do the executive KPIs show?" → computes and returns executive metrics

## Limitations

- The AI is **read-only**. It cannot modify inventory, place orders, or change data.
- Forecast explanations require SHAP and a trained model.
- In offline mode, answers are rule-based summaries, not LLM-generated prose.
- Answers are only as good as the underlying data quality.
- The RAG layer is keyword-based; it may miss relevant docs for very abstract questions.

## Privacy / Security

- No API keys are stored in the repository.
- All LLM calls are configured via environment variables.
- The AI analyst does not bypass the data-access layer; it uses the same tools as the dashboard.
- Tool results are limited to aggregated summaries to avoid sending excessive raw data to the LLM.
