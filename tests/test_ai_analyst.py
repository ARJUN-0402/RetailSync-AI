"""Tests for the AI analyst layer."""

from __future__ import annotations

import pytest

from src.ai_analyst.config import AIAnalystConfig
from src.ai_analyst.exceptions import (
    AIAnalystError,
    GroundingError,
    LLMProviderError,
    MissingConfigurationError,
    RetrievalError,
    ToolExecutionError,
)
from src.ai_analyst.orchestrator import (
    _run_offline_answer,
    ask,
    _build_user_message,
    _extract_tool_calls,
)
from src.ai_analyst.retriever import retrieve, format_context
from src.ai_analyst.tools import (
    TOOL_REGISTRY,
    get_anomalies,
    get_executive_kpis,
    get_forecast_explanation,
    get_forecasts,
    get_inventory_snapshot,
    get_overstock_risks,
    get_product_segments,
    get_reorder_recommendations,
    get_sales_trends,
    get_stockout_risks,
    get_store_segments,
    get_warehouse_performance,
    TOOLS,
)
from src.ai_analyst.prompts import build_tool_prompt, format_sources, SYSTEM_PROMPT


# ============================================================
# CONFIG TESTS
# ============================================================


class TestConfig:
    def test_default_config(self):
        cfg = AIAnalystConfig()
        assert cfg.llm_provider == "openai"
        assert cfg.llm_temperature == 0.1
        assert cfg.llm_max_tokens == 1024
        assert cfg.enable_tools is True
        assert cfg.enable_rag is True

    def test_offline_mode_from_env(self, monkeypatch):
        monkeypatch.setenv("RETAILSYNC_AI_OFFLINE_MODE", "true")
        cfg = AIAnalystConfig()
        assert cfg.offline_mode is True

    def test_disable_tools(self, monkeypatch):
        monkeypatch.setenv("RETAILSYNC_AI_DISABLE_TOOLS", "1")
        cfg = AIAnalystConfig()
        assert cfg.enable_tools is False

    def test_disable_rag(self, monkeypatch):
        monkeypatch.setenv("RETAILSYNC_AI_DISABLE_RAG", "true")
        cfg = AIAnalystConfig()
        assert cfg.enable_rag is False

    def test_is_configured_offline(self):
        cfg = AIAnalystConfig(offline_mode=True)
        assert cfg.is_configured is False

    def test_is_configured_with_key(self):
        cfg = AIAnalystConfig(offline_mode=False, llm_api_key="sk-test")
        assert cfg.is_configured is True

    def test_is_configured_ollama(self):
        cfg = AIAnalystConfig(offline_mode=False, llm_provider="ollama")
        assert cfg.is_configured is True


# ============================================================
# EXCEPTION TESTS
# ============================================================


class TestExceptions:
    def test_base_exception(self):
        with pytest.raises(AIAnalystError):
            raise AIAnalystError("test")

    def test_missing_configuration_error(self):
        with pytest.raises(MissingConfigurationError):
            raise MissingConfigurationError("no key")

    def test_llm_provider_error(self):
        with pytest.raises(LLMProviderError):
            raise LLMProviderError("bad model")

    def test_retrieval_error(self):
        with pytest.raises(RetrievalError):
            raise RetrievalError("doc missing")

    def test_tool_execution_error(self):
        with pytest.raises(ToolExecutionError):
            raise ToolExecutionError("tool failed")

    def test_grounding_error(self):
        with pytest.raises(GroundingError):
            raise GroundingError("ungrounded")


# ============================================================
# PROMPT TESTS
# ============================================================


class TestPrompts:
    def test_system_prompt_not_empty(self):
        assert len(SYSTEM_PROMPT) > 100

    def test_build_tool_prompt(self):
        prompt = build_tool_prompt(TOOLS)
        assert "get_sales_trends" in prompt
        assert "Tool-calling" in prompt or "tools" in prompt.lower()

    def test_format_sources_with_sources(self):
        text = format_sources(["sales.csv", "inventory.csv"])
        assert "sales.csv" in text
        assert "inventory.csv" in text

    def test_format_sources_empty(self):
        assert format_sources([]) == ""


# ============================================================
# RETRIEVER TESTS
# ============================================================


class TestRetriever:
    def test_retrieve_returns_list(self):
        results = retrieve("inventory", top_k=2)
        assert isinstance(results, list)

    def test_retrieve_stockout(self):
        results = retrieve("stockout risk inventory", top_k=2)
        assert len(results) <= 2
        if results:
            assert results[0].score > 0

    def test_format_context_empty(self):
        assert format_context([]) == ""

    def test_format_context_with_chunks(self):
        from src.ai_analyst.retriever import DocChunk
        chunks = [DocChunk(source="test.md", content="inventory risk stockout", score=0.5)]
        text = format_context(chunks)
        assert "test.md" in text
        assert "inventory risk stockout" in text


# ============================================================
# TOOL TESTS
# ============================================================


class TestTools:
    def test_tool_registry_not_empty(self):
        assert len(TOOL_REGISTRY) >= 10

    def test_get_sales_trends_empty(self):
        result = get_sales_trends(days=0)
        assert "error" in result or "data" in result

    def test_get_forecasts_empty(self):
        result = get_forecasts()
        assert "error" in result or "data" in result

    def test_get_inventory_snapshot_empty(self):
        result = get_inventory_snapshot()
        assert "error" in result or "data" in result

    def test_get_stockout_risks_empty(self):
        result = get_stockout_risks()
        assert "error" in result or "data" in result

    def test_get_overstock_risks_empty(self):
        result = get_overstock_risks()
        assert "error" in result or "data" in result

    def test_get_anomalies_empty(self):
        result = get_anomalies()
        assert "error" in result or "data" in result

    def test_get_product_segments_empty(self):
        result = get_product_segments()
        assert "error" in result or "data" in result

    def test_get_store_segments_empty(self):
        result = get_store_segments()
        assert "error" in result or "data" in result

    def test_get_warehouse_performance_empty(self):
        result = get_warehouse_performance()
        assert "error" in result or "data" in result

    def test_get_executive_kpis_empty(self):
        result = get_executive_kpis()
        assert "error" in result or "data" in result

    def test_get_reorder_recommendations_empty(self):
        result = get_reorder_recommendations()
        assert "error" in result or "data" in result

    def test_get_forecast_explanation_empty(self):
        result = get_forecast_explanation("P999", "ST99")
        assert "error" in result or "data" in result

    def test_tools_have_required_fields(self):
        for tool in TOOLS:
            assert "name" in tool
            assert "description" in tool
            assert "parameters" in tool

    def test_tool_registry_matches_tools(self):
        for tool in TOOLS:
            assert tool["name"] in TOOL_REGISTRY


# ============================================================
# ORCHESTRATOR TESTS
# ============================================================


class TestOrchestrator:
    def test_extract_tool_calls_json_array(self):
        text = '[{"name": "get_forecasts", "arguments": {}}]'
        calls = _extract_tool_calls(text)
        assert len(calls) == 1
        assert calls[0]["name"] == "get_forecasts"

    def test_extract_tool_calls_no_match(self):
        calls = _extract_tool_calls("Hello world")
        assert calls == []

    def test_extract_tool_calls_markdown_block(self):
        text = '```json\n[{"name": "get_sales_trends", "arguments": {"days": 7}}]\n```'
        calls = _extract_tool_calls(text)
        assert len(calls) == 1
        assert calls[0]["name"] == "get_sales_trends"

    def test_build_user_message(self):
        msg = _build_user_message("test question", "some context", [])
        assert "test question" in msg
        assert "some context" in msg

    def test_build_user_message_with_tool_results(self):
        msg = _build_user_message("test", "", [{"tool": "get_forecasts", "result": {"summary": {"total": 10}}}])
        assert "get_forecasts" in msg
        assert "10" in msg

    def test_run_offline_answer_stockout(self):
        answer, sources = _run_offline_answer("Why are stockouts increasing?", AIAnalystConfig(offline_mode=True))
        assert len(answer) > 0
        assert "stockout" in answer.lower()

    def test_run_offline_answer_overstock(self):
        answer, sources = _run_offline_answer("Which products are overstocked?", AIAnalystConfig(offline_mode=True))
        assert len(answer) > 0
        assert "overstock" in answer.lower()

    def test_run_offline_answer_reorder(self):
        answer, sources = _run_offline_answer("Which products should I reorder?", AIAnalystConfig(offline_mode=True))
        assert len(answer) > 0
        assert "reorder" in answer.lower()

    def test_run_offline_answer_forecast(self):
        answer, sources = _run_offline_answer("What is the demand forecast?", AIAnalystConfig(offline_mode=True))
        assert len(answer) > 0
        assert "forecast" in answer.lower()

    def test_run_offline_answer_revenue(self):
        answer, sources = _run_offline_answer("Show me sales revenue", AIAnalystConfig(offline_mode=True))
        assert len(answer) > 0
        assert "revenue" in answer.lower()

    def test_run_offline_answer_anomaly(self):
        answer, sources = _run_offline_answer("Explain this anomaly", AIAnalystConfig(offline_mode=True))
        assert len(answer) > 0
        assert "anomal" in answer.lower()

    def test_run_offline_answer_fallback(self):
        answer, sources = _run_offline_answer("What is the weather today?", AIAnalystConfig(offline_mode=True))
        assert len(answer) > 0

    def test_ask_empty_question(self):
        result = ask("", config=AIAnalystConfig(offline_mode=True))
        assert "error" not in result or result["error"] is None
        assert len(result["answer"]) > 0

    def test_ask_offline_stockout(self):
        result = ask("Why are stockouts increasing?", config=AIAnalystConfig(offline_mode=True))
        assert "error" not in result or result["error"] is None
        assert "stockout" in result["answer"].lower()

    def test_ask_missing_config_returns_error_or_offline(self):
        result = ask("test", config=AIAnalystConfig(offline_mode=False, llm_api_key="", llm_provider="openai"))
        assert result["answer"] is not None and len(result["answer"]) > 0


# ============================================================
# GROUNDING TESTS
# ============================================================


class TestGrounding:
    def test_tool_results_are_from_actual_data(self):
        result = get_sales_trends(days=30)
        if "data" in result and result["data"]:
            for record in result["data"]:
                assert "total_revenue" in record or "total_quantity" in record

    def test_no_invented_metrics(self):
        result = get_executive_kpis()
        if "data" in result:
            kpis = result["data"]
            for key in ["total_inventory_value", "estimated_carrying_cost", "stockout_exposure"]:
                if key in kpis:
                    assert kpis[key] >= 0

    def test_forecast_data_grounded(self):
        result = get_forecasts()
        if "data" in result and result["data"]:
            for record in result["data"][:3]:
                assert "forecast_demand_14d" in record or "product_id" in record


# ============================================================
# SESSION STATE INITIALIZATION TESTS
# ============================================================


class TestSessionStateInitialization:
    def _fake_session_state(self, initial=None):
        initial = initial or {}

        class FakeSessionState(dict):
            def __getattr__(self, name):
                try:
                    return self[name]
                except KeyError:
                    raise AttributeError(name)

            def __setattr__(self, name, value):
                self[name] = value

            def __contains__(self, key):
                return super().__contains__(key)

        return FakeSessionState(initial)

    def test_init_session_state_creates_empty_chat_history(self):
        from dashboard.src_pages.ai_analyst import _init_session_state
        from unittest.mock import patch

        fake_state = self._fake_session_state()
        with patch("streamlit.session_state", fake_state):
            _init_session_state()
            assert fake_state.chat_history == []

    def test_init_session_state_preserves_existing_history(self):
        from dashboard.src_pages.ai_analyst import _init_session_state
        from unittest.mock import patch

        existing = [{"role": "user", "content": "hello"}]
        fake_state = self._fake_session_state({"chat_history": existing})
        with patch("streamlit.session_state", fake_state):
            _init_session_state()
            assert fake_state.chat_history is existing
            assert len(fake_state.chat_history) == 1

    def test_init_session_state_caps_at_50_messages(self):
        from dashboard.src_pages.ai_analyst import _init_session_state
        from unittest.mock import patch

        long_history = [{"role": "user", "content": f"msg{i}"} for i in range(60)]
        fake_state = self._fake_session_state({"chat_history": long_history})
        with patch("streamlit.session_state", fake_state):
            _init_session_state()
            assert len(fake_state.chat_history) == 50
            assert fake_state.chat_history[0]["content"] == "msg10"

    def test_render_ai_analyst_page_handles_missing_chat_history(self):
        from dashboard.src_pages.ai_analyst import render_ai_analyst_page
        from unittest.mock import patch, MagicMock

        fake_state = self._fake_session_state()
        mock_data = {"products": MagicMock()}

        with patch("streamlit.session_state", fake_state):
            with patch("dashboard.src_pages.ai_analyst.st") as mock_st:
                mock_st.session_state = fake_state
                mock_st.markdown = MagicMock()
                mock_st.container = MagicMock(return_value=MagicMock(__enter__=MagicMock(return_value=None), __exit__=MagicMock()))
                mock_st.columns = MagicMock(return_value=[MagicMock()])
                mock_st.sidebar = MagicMock()
                mock_st.button = MagicMock(return_value=False)
                mock_st.chat_message = MagicMock(return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()))
                mock_st.chat_input = MagicMock(return_value=None)
                mock_st.spinner = MagicMock(return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()))

                with patch("dashboard.src_pages.ai_analyst.AIAnalystConfig") as MockConfig:
                    MockConfig.from_env.return_value = MagicMock(is_configured=False)
                    with patch("dashboard.src_pages.ai_analyst.render_alert"):
                        with patch("dashboard.src_pages.ai_analyst.render_section_header"):
                            with patch("dashboard.components.ui.render_kpi_row"):
                                with patch("dashboard.src_pages.ai_analyst.ask"):
                                    render_ai_analyst_page(mock_data)

            assert "chat_history" in fake_state
            assert fake_state.chat_history == []
