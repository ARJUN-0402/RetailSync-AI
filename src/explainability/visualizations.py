"""Plotly visualisations for SHAP explanations.

All charts are built with Plotly (matching the rest of the dashboard) and are
derived purely from real model SHAP outputs - no synthetic data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .shap_explainer import GlobalExplanation, LocalExplanation

_PLOT_BG = "#1a1d23"
_ACCENT = "#00d4ff"
_POS = "#2ed573"
_NEG = "#ff4757"


def _dark_layout(**overrides) -> dict:
    layout = {
        "template": "plotly_dark",
        "paper_bgcolor": _PLOT_BG,
        "plot_bgcolor": _PLOT_BG,
        "font": {"color": "#ffffff"},
        "margin": {"l": 40, "r": 20, "t": 50, "b": 40},
    }
    layout.update(overrides)
    return layout


def global_importance_chart(global_exp: GlobalExplanation, top_n: int = 20) -> go.Figure:
    """Horizontal bar chart of mean |SHAP| per feature (global ranking)."""
    ranking = global_exp.top(top_n).iloc[::-1]  # smallest on bottom
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=ranking["mean_abs_shap"],
            y=ranking["feature"],
            orientation="h",
            marker={"color": ranking["mean_abs_shap"], "colorscale": "Tealgrn"},
            name="mean |SHAP|",
        )
    )
    fig.update_layout(
        _dark_layout(
            title=f"Global Feature Importance (mean |SHAP| over {global_exp.sample_size} samples)",
            xaxis_title="mean |SHAP| contribution to demand",
            yaxis_title=None,
            height=max(400, top_n * 22),
        )
    )
    return fig


def shap_summary_chart(
    global_exp: GlobalExplanation,
    shap_values: np.ndarray,
    feature_values: pd.DataFrame,
    max_features: int = 15,
    sample: int | None = None,
) -> go.Figure:
    """SHAP summary (beeswarm) chart, implemented with Plotly.

    For each top feature we plot its SHAP value (x) against a jittered rank (y),
    coloured by the feature's actual value - the canonical SHAP summary view.
    """
    features = global_exp.top(max_features)["feature"].tolist()
    values = np.asarray(shap_values, dtype=float)
    feat_mat = feature_values[features].to_numpy(dtype=float)

    # Bound the number of points for performance.
    n = values.shape[0]
    if sample is not None and n > sample:
        rng = np.random.default_rng(0)
        idx = rng.choice(n, size=sample, replace=False)
        values = values[idx]
        feat_mat = feat_mat[idx]

    fig = go.Figure()
    rng = np.random.default_rng(1)
    for rank, feat in enumerate(features):
        sv = values[:, rank]
        fv = feat_mat[:, rank]
        jitter = rng.uniform(-0.32, 0.32, size=len(sv))
        fig.add_trace(
            go.Scatter(
                x=sv,
                y=[rank + jitter[i] for i in range(len(sv))],
                mode="markers",
                marker={
                    "size": 5,
                    "color": fv,
                    "colorscale": "Viridis",
                    "opacity": 0.7,
                    "showscale": rank == 0,
                },
                name=feat,
                showlegend=False,
                hovertemplate=f"{feat}<br>SHAP: %{{x:.3f}}<br>value: %{{marker.color:.2f}}<extra></extra>",
            )
        )

    fig.update_layout(
        _dark_layout(
            title="SHAP Summary (beeswarm) - feature value vs. SHAP contribution",
            xaxis_title="SHAP value (impact on predicted demand)",
            yaxis_title=None,
            height=max(400, max_features * 28),
            yaxis={"tickmode": "array", "tickvals": list(range(len(features))), "ticktext": features},
        )
    )
    return fig


def local_waterfall_chart(expl: LocalExplanation, top_n: int = 12) -> go.Figure:
    """Waterfall of SHAP contributions for a single forecast."""
    top = expl.contributions(top_n)
    measure = ["relative"] * (len(top) + 1)
    x = [expl.expected_value]
    y = ["baseline"]
    text = [f"{expl.expected_value:.2f}"]

    for c in top:
        x.append(c.shap_value)
        y.append(c.feature)
        text.append(f"{c.shap_value:+.2f}")
    x.append(expl.predicted_value)
    y.append("prediction")
    text.append(f"{expl.predicted_value:.2f}")
    measure[-1] = "total"

    fig = go.Figure(
        go.Waterfall(
            orientation="v",
            measure=measure,
            x=y,
            y=x,
            text=text,
            connector={"line": {"color": "#555"}},
            increasing={"marker": {"color": _POS}},
            decreasing={"marker": {"color": _NEG}},
            totals={"marker": {"color": _ACCENT}},
        )
    )
    fig.update_layout(
        _dark_layout(
            title="Local Explanation - how each feature moves the forecast",
            yaxis_title="demand (units)",
            xaxis_title=None,
            height=max(400, top_n * 26),
        )
    )
    return fig


def driver_bars_chart(expl: LocalExplanation, top_n: int = 10) -> go.Figure:
    """Bar chart of the strongest positive and negative drivers combined."""
    pos = expl.top_positive(top_n)
    neg = expl.top_negative(top_n)
    items = (pos + neg)[
        : top_n * 2
    ]
    items.sort(key=lambda c: c.shap_value, reverse=True)
    if not items:
        fig = go.Figure()
        fig.update_layout(_dark_layout(title="No feature drivers identified"))
        return fig

    labels = [c.feature for c in items]
    vals = [c.shap_value for c in items]
    colors = [_POS if v >= 0 else _NEG for v in vals]
    labels = labels[::-1]
    vals = vals[::-1]
    colors = colors[::-1]

    fig = go.Figure(
        go.Bar(
            x=vals,
            y=labels,
            orientation="h",
            marker_color=colors,
            text=[f"{v:+.2f}" for v in vals],
            textposition="outside",
        )
    )
    fig.update_layout(
        _dark_layout(
            title="Top drivers (green = increases demand, red = decreases)",
            xaxis_title="SHAP contribution (units)",
            yaxis_title=None,
            height=max(400, len(labels) * 22),
        )
    )
    return fig
