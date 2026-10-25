"""Operational safeguards tests: query input bounds, structured JSON logging, and LLM timeouts."""

import asyncio
import json
import logging
from unittest.mock import MagicMock, patch

import pytest

from app.models.schemas import QueryRequest
from app.services.retrieval_service import RetrievalService
from app.utils.logging_config import JSONFormatter, setup_logging


class TestOperationalSafeguards:
    """Test suite verifying input bounds, structured logging, and LLM retry/timeout safeguards."""

    def test_query_empty_or_whitespace_question_rejected(self) -> None:
        """Verify queries with only spaces, tabs, or empty text fail Pydantic validation."""
        with pytest.raises(ValueError, match="at least 2 non-whitespace characters"):
            QueryRequest(question="   \t\n  ", top_k=3)

        with pytest.raises(ValueError, match="at least 2 non-whitespace characters"):
            QueryRequest(question="a", top_k=3)

    def test_query_max_length_exceeded_rejected(self) -> None:
        """Verify questions exceeding 1000 characters fail validation."""
        huge_question = "Explain this chart " * 70  # ~1330 characters
        with pytest.raises(ValueError, match="cannot exceed 1000 characters"):
            QueryRequest(question=huge_question, top_k=3)

    def test_query_top_k_bounds(self) -> None:
        """Verify top_k bounds must be within 1 and 50."""
        with pytest.raises(ValueError):
            QueryRequest(question="Valid question", top_k=0)

        with pytest.raises(ValueError):
            QueryRequest(question="Valid question", top_k=51)

        # Valid top_k
        q = QueryRequest(question="Valid question", top_k=25)
        assert q.top_k == 25

    def test_structured_json_logging_format(self) -> None:
        """Verify JSONFormatter outputs valid structured JSON lines with required telemetry fields."""
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname="app/services/test.py",
            lineno=42,
            msg="User %s ingested document %s",
            args=("alice", "financial.pdf"),
            exc_info=None,
        )

        formatted_line = formatter.format(record)
        parsed = json.loads(formatted_line)

        assert parsed["level"] == "INFO"
        assert parsed["logger"] == "test_logger"
        assert parsed["line"] == 42
        assert parsed["message"] == "User alice ingested document financial.pdf"
        assert "timestamp" in parsed

    def test_setup_logging_configures_json_stream_handler(self) -> None:
        """Verify setup_logging attaches JSONFormatter when log_format='json'."""
        setup_logging(log_level="DEBUG", log_format="json")
        root = logging.getLogger()
        assert len(root.handlers) > 0
        assert isinstance(root.handlers[0].formatter, JSONFormatter)

        # Reset back to text format for rest of tests
        setup_logging(log_level="INFO", log_format="text")

    def test_retrieval_llm_timeout_triggers_grounded_fallback(self) -> None:
        """Verify that when an external LLM call hangs beyond timeout, retrieval falls back gracefully."""
        from app.config import Settings

        settings = Settings(
            vlm_provider="gemini",
            gemini_api_key="valid-mock-key",
            vlm_timeout_seconds=0.1,  # Fast timeout for test
        )
        service = RetrievalService(settings=settings)

        prompt = "Synthesize an answer based on: [Page 1 text]"
        context = ["[Source: test.pdf | Page 1] Revenue was $10M."]

        # Simulate hanging generate_content call
        with patch("google.generativeai.configure"), patch("google.generativeai.GenerativeModel") as mock_model_cls:
            mock_model = MagicMock()

            def _hang(*args, **kwargs):
                import time
                time.sleep(1.0)
                return MagicMock(text="Hanging answer")

            mock_model.generate_content.side_effect = _hang
            mock_model_cls.return_value = mock_model

            answer, model_name = asyncio.run(service._generate_answer(prompt, context, "What was revenue?"))
            # Must fall back to local grounded synthesizer
            assert model_name == "local-grounded-synthesizer"
            assert "Revenue was $10M" in answer
