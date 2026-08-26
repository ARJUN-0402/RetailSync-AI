"""Natural-language explanation generator for SHAP local explanations.

Turns a :class:`LocalExplanation` (feature contributions) into a business-friendly
sentence. The text is generated dynamically from the model's actual SHAP values
and the real feature values - nothing about the explanation is hard-coded, so it
adapts to whatever the model is currently predicting.
"""

from __future__ import annotations

from .feature_descriptions import describe_contribution
from .shap_explainer import LocalExplanation


def _join_phrases(phrases: list[str]) -> str:
    if not phrases:
        return ""
    if len(phrases) == 1:
        return phrases[0]
    if len(phrases) == 2:
        return f"{phrases[0]} and {phrases[1]}"
    return ", ".join(phrases[:-1]) + f", and {phrases[-1]}"


def _contribution_phrase(expl: LocalExplanation, feature: str) -> str:
    value = expl.feature_values.get(feature, float("nan"))
    median = expl.feature_medians.get(feature)
    return describe_contribution(feature, value, median)


def build_explanation(
    expl: LocalExplanation,
    context: dict | None = None,
    top_n: int = 3,
    include_numbers: bool = True,
) -> str:
    """Generate a human-readable explanation for a single forecast.

    Parameters
    ----------
    expl:
        The local SHAP explanation produced by :class:`ExplainabilityEngine`.
    context:
        Optional labels such as ``{"product_id": ..., "store_id": ...}`` used to
        personalise the opening sentence.
    top_n:
        How many positive/negative drivers to mention.
    include_numbers:
        Whether to include the predicted vs. typical demand figures.
    """
    direction = expl.direction
    base = expl.expected_value
    pred = expl.predicted_value
    diff = pred - base

    scope = ""
    if context:
        bits = []
        if context.get("product_id"):
            bits.append(f"product {context['product_id']}")
        if context.get("store_id"):
            bits.append(f"store {context['store_id']}")
        if bits:
            scope = " for " + " at ".join(bits) if len(bits) == 2 else " for " + bits[0]

    number_clause = (
        f" ({pred:.1f} units vs. a typical {base:.1f} units, a "
        f"{abs(diff):.1f}-unit {'uplift' if diff >= 0 else 'reduction'})"
        if include_numbers
        else ""
    )

    sentences: list[str] = []
    opening = (
        f"Demand is expected to {direction}{scope}{number_clause}."
    )
    sentences.append(opening)

    pos = expl.top_positive(top_n)
    neg = expl.top_negative(top_n)

    if pos:
        pos_phrases = [_contribution_phrase(expl, c.feature) for c in pos]
        verb = "increase" if direction == "increase" else "be higher"
        lead = (
            f"It is expected to {verb} primarily because "
            if direction == "increase"
            else f"It is {verb} than usual primarily because "
        )
        sentences.append(lead + _join_phrases(pos_phrases) + ".")

    if neg:
        neg_phrases = [_contribution_phrase(expl, c.feature) for c in neg]
        if direction == "increase":
            sentences.append(
                "This is partly tempered by " + _join_phrases(neg_phrases) + "."
            )
        else:
            sentences.append(
                "It is held down further by " + _join_phrases(neg_phrases) + "."
            )

    if not pos and not neg:
        sentences.append(
            "No single feature dominates this prediction; the forecast reflects the "
            "model baseline with little feature-specific deviation."
        )

    return " ".join(sentences)


def explain_forecast(
    expl: LocalExplanation,
    context: dict | None = None,
    top_n: int = 3,
) -> str:
    """Thin wrapper kept for readability at call sites."""
    return build_explanation(expl, context=context, top_n=top_n)
