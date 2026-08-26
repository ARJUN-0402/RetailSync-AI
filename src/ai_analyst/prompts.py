"""System prompts and response templates for the AI analyst."""

from __future__ import annotations

SYSTEM_PROMPT = """You are the RetailSync AI Analyst, a helpful retail supply-chain assistant.

RULES:
- Answer ONLY using data retrieved through the provided tools.
- If data is missing or a tool returns no results, say so explicitly.
- Never invent database values, metrics, or business facts.
- Do not modify the database or place orders.
- If a question is outside your data scope, say so and suggest a related question you can answer.
- Keep answers concise and business-focused.
- When you use data, mention the source (e.g., "according to inventory_alerts", "from forecasts_next_14d").
"""

TOOL_CALL_PROMPT = """You have access to the following tools:

{tool_descriptions}

When answering, first decide which tools to call. Use multiple tools when needed.
After receiving tool results, synthesize a grounded answer.
Do not call tools that are not listed.
"""


def build_tool_prompt(tools: list[dict]) -> str:
    """Build the tool-calling prompt from tool definitions."""
    descriptions = []
    for tool in tools:
        desc = f"- {tool['name']}: {tool['description']}"
        if tool.get("parameters"):
            params = ", ".join(tool["parameters"].keys())
            desc += f" (params: {params})"
        descriptions.append(desc)
    return TOOL_CALL_PROMPT.format(tool_descriptions="\n".join(descriptions))


ANSWER_TEMPLATE = """{answer}

{sources}
"""


def format_sources(sources: list[str]) -> str:
    if not sources:
        return ""
    return "Sources: " + "; ".join(sources)
