# Model Explainability (SHAP)

A modular, training-agnostic explainability layer for the RetailSync AI demand
forecaster. It uses [SHAP](https://shap.readthedocs.io/) (SHapley Additive
exPlanations) to answer three business questions:

1. **Globally** — which features influence demand predictions across the whole model?
2. **Locally** — which features influenced one specific forecast?
3. **Directionally** — why is demand expected to increase or decrease?

It reuses the already-trained model package (`models/demand_forecaster.pkl`); it
**never retrains** the model and the forecasting pipeline is untouched.

---

## What SHAP is doing here

SHAP assigns each feature an *attribution* (a SHAP value, in the same units as
the prediction — demand in units) for a given prediction. The attributions sum
to the difference between the model's prediction and its baseline ("expected")
value:

```
prediction = expected_value + sum(SHAP_i for all features i)
```

So a positive SHAP value means the feature **pushed demand up** relative to the
model's typical demand; a negative value means it **pulled demand down**.

For the supported tree-based models we use `shap.TreeExplainer`, which computes
exact Shapley values directly from the tree structure (fast, no approximation).

---

## Global vs. local explainability

| View | Question it answers | Output |
|------|--------------------|--------|
| **Global** | Which features matter most *in general*? | Mean `|SHAP|` per feature, ranked; a SHAP beeswarm summary. |
| **Local** | Why is *this* forecast what it is? | Per-feature SHAP values for one row; top positive/negative drivers; a waterfall; a plain-English sentence. |

- **Global feature importance** = average absolute SHAP value over a sample of
  background rows. It is model-aware (unlike Gini importance) and works even for
  correlated features.
- **Local explanation** = the SHAP vector for a single instance. The biggest
  positive entries are the drivers that *increased* the forecast; the biggest
  negative entries are the drivers that *decreased* it.

---

## How business explanations are generated

`build_explanation()` turns a local `LocalExplanation` into a sentence **dynamically**:

1. It reads the model's actual SHAP values and the **real feature values** for
   the row (no hard-coded text, no hard-coded importances).
2. It maps each feature to a human-readable label (see
   `src/explainability/feature_descriptions.py`) and compares the value to the
   background median to say whether it is "above/below normal".
3. It opens with the direction ("increase"/"decrease") plus the predicted vs.
   typical demand numbers, then lists the top positive drivers and the top
   negative drivers.

Example (generated live, not hand-written):

> Demand is expected to increase for product P001 at store S001 (25.4 units vs.
> a typical 8.4 units, a 16.9-unit uplift). It is expected to increase primarily
> because Demand Rolling Median 28D is above normal, Revenue Lag 7D is above
> normal, and store-type usual demand is above normal. This is partly tempered by
> demand variability is above normal, reorder point is above normal, and Demand
> Rolling Std 14D is above normal.

Because the text is built from `feature → label → value-vs-median`, the same code
produces correct, different explanations for every product, store, and date.

---

## Supported models

Detected automatically in `detect_explainer_kind()`:

| Family | Explainer | Notes |
|--------|-----------|-------|
| RandomForest, ExtraTrees, GradientBoosting, HistGradientBoosting, XGBoost, LightGBM | `TreeExplainer` | Exact, fast. Primary path (the saved model is a RandomForest). |
| LinearRegression, Ridge, Lasso, ElasticNet, SGDRegressor, LinearSVR, LogisticRegression | `LinearExplainer` | Requires a background sample (provided automatically). |
| SVR / KNeighbors / single DecisionTree | `KernelExplainer` | Allowed but expensive — not recommended for large background sets. |
| Anything else | — | Raises `UnsupportedModelError`; the UI degrades gracefully. |

If SHAP itself is not installed, `detect_explainer_kind()` raises
`UnsupportedModelError` and the dashboard shows a friendly message instead of
crashing.

---

## Where the code lives

```
src/explainability/
    exceptions.py            # ExplainabilityError hierarchy
    shap_explainer.py        # ExplainabilityEngine: global + local SHAP, sampling, caching
    feature_descriptions.py  # feature -> business label / value phrase
    explanation.py           # build_explanation() natural-language generator
    visualizations.py        # Plotly charts (importance bar, beeswarm, waterfall, driver bars)
    __init__.py              # public API

dashboard/
    explainability_page.py   # render_explainability_page() + render_why_forecast()
    app.py                   # wires the new "🧠 Model Explainability" nav entry + "Why this forecast?"
```

The explainability code is fully decoupled from `src/forecasting/*` (training).

---

## Performance controls

SHAP can be expensive, so the layer is bounded:

- **Cached background sample** — `ExplainabilityEngine` samples a fixed-size
  background set once and reuses it (`background_sample_size`, default 100).
- **Sampling** — global importance is computed over a bounded, configurable
  sample (`global_sample_size`, default 200; UI slider 50–500).
- **Local = 1 row** — only one instance is explained per request.
- **Caching** — global SHAP values are memoised per (size, data) inside the
  engine; the Streamlit engine itself is cached with `@st.cache_resource` so the
  explainer object is not rebuilt on every rerun.

---

## Error handling

The layer never crashes the app. Failures map to typed exceptions that the UI
catches and renders as messages:

- **Missing model** → `ModelLoadError` (and a dashboard banner to run the pipeline).
- **Unsupported model** → `UnsupportedModelError` (friendly "not supported" note).
- **Missing features** → `MissingFeaturesError` (e.g. an instance lacking columns).
- **Invalid input** → surfaced as `MissingFeaturesError` / `FeatureContributionError`.
- **SHAP runtime failure** → `FeatureContributionError` (dashboard shows the error).

---

## Usage (Python)

```python
from src.explainability import load_explainability_engine, build_explanation

engine = load_explainability_engine()            # reuses trained model + features
global_exp = engine.global_importance(sample_size=150)
print(global_exp.top(10))

row = features_df[engine.feature_cols].iloc[[0]]
local = engine.explain_instance(row)
print(local.top_positive(3), local.top_negative(3))
print(build_explanation(local, context={"product_id": "P001", "store_id": "S001"}))
```

## Usage (Dashboard)

- **Model Explainability** page: pick the model, explore global feature
  importance + SHAP summary, then pick a product/store/date and click
  *Explain this forecast* for the local breakdown and plain-English reason.
- **Demand Forecast** page: each forecast now shows a lightweight
  *"Why this forecast?"* panel with the generated explanation and top drivers.

---

## Known limitations

- **Global importance is sample-based.** Mean `|SHAP|` depends on the background
  rows sampled; with a small sample the ranking can shift slightly. Increase the
  sample size for stability.
- **Beeswarm colouring** uses the feature's raw value magnitude, not a
  percentile scale, so features on very different scales are not directly
  comparable by colour.
- **KernelExplainer** (SVR/KNN) is allowed but slow on large backgrounds; prefer
  tree/linear models for interactive use.
- **Natural-language phrases** are generated from a feature-label dictionary plus
  value-vs-median comparison. They are business-friendly but not a substitute for
  a domain review of edge cases (e.g. extreme outliers).
- **Local explanations** explain the model's *input* row; for the forward forecast
  the "Why this forecast?" panel uses the latest known conditions as the model's
  input basis for the next day.
- Requires `shap` (added to `requirements.txt`); without it the layer reports
  unsupported rather than failing.
