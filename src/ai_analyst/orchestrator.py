"""LLM orchestration for the AI analyst layer.

Handles prompt construction, tool calling, response grounding, and safety.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from .config import AIAnalystConfig
from .exceptions import (
    LLMProviderError,
    MissingConfigurationError,
    RetrievalError,
    ToolExecutionError,
)
from .prompts import SYSTEM_PROMPT, build_tool_prompt
from .retriever import format_context, retrieve
from .tools import TOOLS, TOOL_REGISTRY, execute_tool

logger = logging.getLogger(__name__)


# ============================================================
# PROMPT CONSTRUCTION
# ============================================================

def _build_system_prompt(config: AIAnalystConfig) -> str:
    prompt = SYSTEM_PROMPT
    if config.enable_tools:
        prompt += "\n\n" + build_tool_prompt(TOOLS)
    return prompt


def _build_user_message(question: str, context: str, tool_results: list[dict]) -> str:
    parts = [f"User question: {question}"]
    if context:
        parts.append(f"\nRetrieved documentation context:\n{context}")
    if tool_results:
        parts.append("\nTool results:")
        for i, result in enumerate(tool_results, 1):
            tool_name = result.get("tool", "unknown")
            data = result.get("result", {})
            summary = data.get("summary", data.get("data", {}))
            if isinstance(summary, dict):
                summary_str = json.dumps(summary, default=str)
            else:
                summary_str = str(summary)
            parts.append(f"\nTool {i} ({tool_name}): {summary_str[:4000]}")
    return "\n".join(parts)


# ============================================================
# LLM CALL
# ============================================================

def _call_llm(system_prompt: str, user_message: str, config: AIAnalystConfig) -> str:
    if config.offline_mode or not config.is_configured:
        raise MissingConfigurationError("LLM is not configured. Set RETAILSYNC_AI_API_KEY or enable offline mode.")

    provider = config.llm_provider.lower()
    api_key = config.llm_api_key
    base_url = config.llm_base_url
    model = config.llm_model

    if provider == "openai":
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise LLMProviderError("openai package is not installed.") from exc
        client = OpenAI(api_key=api_key, base_url=base_url or None)
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                temperature=config.llm_temperature,
                max_tokens=config.llm_max_tokens,
            )
            return response.choices[0].message.content or ""
        except Exception as exc:
            raise LLMProviderError(f"OpenAI API error: {exc}") from exc

    if provider == "anthropic":
        try:
            import anthropic
        except ImportError as exc:
            raise LLMProviderError("anthropic package is not installed.") from exc
        client = anthropic.Anthropic(api_key=api_key)
        try:
            response = client.messages.create(
                model=model,
                max_tokens=config.llm_max_tokens,
                temperature=config.llm_temperature,
                system=system_prompt,
                messages=[{"role": "user", "content": user_message}],
            )
            return response.content[0].text
        except Exception as exc:
            raise LLMProviderError(f"Anthropic API error: {exc}") from exc

    if provider == "ollama":
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise LLMProviderError("openai package is not installed.") from exc
        client = OpenAI(base_url=base_url or "http://localhost:11434/v1", api_key="ollama")
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                temperature=config.llm_temperature,
                max_tokens=config.llm_max_tokens,
            )
            return response.choices[0].message.content or ""
        except Exception as exc:
            raise LLMProviderError(f"Ollama API error: {exc}") from exc

    raise LLMProviderError(f"Unsupported LLM provider: {provider}")


# ============================================================
# TOOL CALLING LOOP
# ============================================================

def _extract_tool_calls(text: str) -> list[dict[str, Any]]:
    """Extract tool calls from LLM response text.

    Supports two formats:
    1. JSON array: [{"name": "tool_name", "arguments": {...}}]
    2. Markdown code blocks with JSON.
    """
    calls = []
    try:
        match = re.search(r"\[.*?\]", text, re.DOTALL)
        if match:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, list):
                for item in parsed:
                    if isinstance(item, dict) and "name" in item:
                        calls.append(item)
    except (json.JSONDecodeError, AttributeError):
        pass
    return calls


def _run_tool_rounds(question: str, config: AIAnalystConfig) -> tuple[str, list[str]]:
    """Run tool-calling rounds and return the final answer and sources."""
    system_prompt = _build_system_prompt(config)
    tool_results: list[dict[str, Any]] = []
    sources: list[str] = []

    for round_num in range(config.max_tool_rounds):
        user_message = _build_user_message(question, "", tool_results)
        response = _call_llm(system_prompt, user_message, config)
        tool_calls = _extract_tool_calls(response)
        if not tool_calls:
            return response, sources
        new_results = []
        for call in tool_calls:
            name = call.get("name", "")
            arguments = call.get("arguments", {})
            if name not in TOOL_REGISTRY:
                continue
            try:
                result = execute_tool(name, arguments)
                new_results.append({"tool": name, "result": result})
                if "error" not in result:
                    if name == "get_forecast_explanation" and "data" in result:
                        sources.append(f"SHAP explanation for {arguments.get('product_id')}-{arguments.get('store_id')}")
                    elif name == "get_executive_kpis":
                        sources.append("Executive KPIs computed from business_metrics module")
                    else:
                        sources.append(f"Tool: {name}")
            except ToolExecutionError as exc:
                new_results.append({"tool": name, "result": {"error": str(exc)}})
        tool_results.extend(new_results)
        if not new_results:
            break
    final_message = _build_user_message(question, "", tool_results)
    final_response = _call_llm(system_prompt, final_message, config)
    return final_response, sources


# ============================================================
# MAIN ORCHESTRATOR
# ============================================================

def ask(question: str, config: AIAnalystConfig | None = None) -> dict[str, Any]:
    """Ask a natural-language question and get a grounded answer.

    Returns a dict with:
        - answer: The generated answer text.
        - sources: List of data sources used.
        - error: Optional error message.
    """
    if config is None:
        config = AIAnalystConfig.from_env()
    if not question.strip():
        return {"answer": "Please ask a question about your retail data.", "sources": [], "error": None}

    try:
        _context = ""
        if config.enable_rag:
            try:
                chunks = retrieve(question, top_k=config.retrieval_top_k, max_chunk_chars=config.retrieval_chunk_size)
                _context = format_context(chunks)
            except Exception as exc:
                logger.warning("RAG retrieval failed: %s", exc)
                raise RetrievalError(f"Documentation retrieval failed: {exc}") from exc

        if config.enable_tools and not config.offline_mode:
            try:
                answer, sources = _run_tool_rounds(question, config)
                return {"answer": answer, "sources": sources, "error": None}
            except MissingConfigurationError:
                pass

        answer, sources = _run_offline_answer(question, config)
        return {"answer": answer, "sources": sources, "error": None}

    except MissingConfigurationError as exc:
        return {
            "answer": (
                "AI Analyst is not configured. Set the RETAILSYNC_AI_API_KEY environment variable "
                "or enable offline mode. The dashboard remains fully functional without AI features."
            ),
            "sources": [],
            "error": str(exc),
        }
    except RetrievalError as exc:
        return {"answer": f"Could not retrieve documentation: {exc}", "sources": [], "error": str(exc)}
    except LLMProviderError as exc:
        return {"answer": f"LLM provider error: {exc}", "sources": [], "error": str(exc)}
    except Exception as exc:
        logger.exception("Unexpected error in AI analyst")
        return {"answer": f"An unexpected error occurred: {exc}", "sources": [], "error": str(exc)}


def _run_offline_answer(question: str, config: AIAnalystConfig) -> tuple[str, list[str]]:
    """Fallback rule-based responses when LLM is unavailable."""
    q = question.lower()
    sources: list[str] = []

    if "stockout" in q or "out of stock" in q:
        result = TOOL_REGISTRY["get_stockout_risks"]()
        if "error" in result:
            return result["error"], sources
        high = result["summary"]["high_risk"]
        med = result["summary"]["medium_risk"]
        return (
            f"There are {high} items at HIGH stockout risk and {med} at MEDIUM risk. "
            "Immediate restock is recommended for HIGH-risk items.",
            ["inventory_alerts (stockout_risk)"],
        )

    if "overstock" in q or "excess" in q:
        result = TOOL_REGISTRY["get_overstock_risks"]()
        if "error" in result:
            return result["error"], sources
        high = result["summary"]["high_risk"]
        return (
            f"There are {high} items at HIGH overstock risk. "
            "Consider promotions, redistribution, or reducing incoming orders.",
            ["inventory_alerts (overstock_risk)"],
        )

    if "reorder" in q or "restock" in q:
        result = TOOL_REGISTRY["get_reorder_recommendations"]()
        if "error" in result:
            return result["error"], sources
        s = result["summary"]
        return (
            f"{s.get('critical', 0)} critical, {s.get('urgent', 0)} urgent, and {s.get('soon', 0)} soon reorder recommendations. "
            "Check the Business Intelligence page for full details.",
            ["reorder_recommendations"],
        )

    if "anomal" in q or "spike" in q:
        result = TOOL_REGISTRY["get_anomalies"]()
        if "error" in result:
            return result["error"], sources
        spikes = result["summary"].get("spikes", 0)
        return (
            f"{result['summary']['total_returned']} recent anomalies detected, including {spikes} demand spikes. "
            "Review the Demand Anomalies page for full details.",
            ["anomalies.csv"],
        )

    if "forecast" in q or "demand" in q:
        result = TOOL_REGISTRY["get_forecasts"]()
        if "error" in result:
            return result["error"], sources
        return (
            f"14-day forecast: {result['summary']['total_forecast_demand']:,.0f} units, "
            f"${result['summary']['total_forecast_revenue']:,.2f} revenue across "
            f"{result['summary']['product_store_combinations']} product-store combinations.",
            ["forecasts_next_14d.csv"],
        )

    if "revenue" in q or "sales" in q:
        result = TOOL_REGISTRY["get_sales_trends"]()
        if "error" in result:
            return result["error"], sources
        return (
            f"Total revenue (last 30 days): ${result['summary']['total_revenue']:,.0f}. "
            f"Total quantity sold: {result['summary']['total_quantity']:,.0f} units.",
            ["sales.csv"],
        )

    return (
        "I can help with stockout risks, overstock situations, reorder needs, "
        "anomalies, forecasts, revenue, and product/store segmentation. "
        "Try asking a specific question about one of these topics.",
        [],
    )
