"""Unit and mock integration tests for VLMService."""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from PIL import Image

from app.config import Settings
from app.core.exceptions import VLMServiceError
from app.models.enums import VLMProvider
from app.models.schemas import ExtractedImage
from app.services.vlm_service import VLM_SYSTEM_PROMPT, VLMService


@pytest.fixture
def test_image_path(tmp_path: Path) -> Path:
    """Creates a temporary test PNG image."""
    img = Image.new("RGB", (120, 120), color=(50, 150, 200))
    p = tmp_path / "chart_sample.png"
    img.save(p, format="PNG")
    return p


class TestVLMService:
    """Test suite for VLMService configuration, fallbacks, and provider calls."""

    def test_missing_file_raises_error(self, tmp_path: Path) -> None:
        """Verify passing a non-existent image triggers VLMServiceError."""
        service = VLMService(settings=Settings(gemini_api_key=None))
        with pytest.raises(VLMServiceError, match="Image file not found"):
            asyncio.run(service.summarize_image(tmp_path / "non_existent.png"))

    def test_synthetic_fallback_when_unconfigured(self, test_image_path: Path) -> None:
        """Verify that when no API keys are provided, a descriptive synthetic summary is returned."""
        service = VLMService(settings=Settings(gemini_api_key=None, openai_api_key=None))
        assert not service.is_live_configured

        summary = asyncio.run(
            service.summarize_image(
                image_path=test_image_path,
                source_pdf="financial_q3.pdf",
                page_number=4,
            )
        )
        assert "financial_q3.pdf" in summary
        assert "page 4" in summary
        assert "chart_sample.png" in summary

    def test_batch_summarization(self, test_image_path: Path) -> None:
        """Verify batch summarization processes multiple ExtractedImage objects concurrently."""
        service = VLMService(settings=Settings(gemini_api_key=None))
        images = [
            ExtractedImage(
                image_id=f"doc_p{i}_img0",
                source_pdf="doc.pdf",
                page_number=i,
                image_index=0,
                local_path=str(test_image_path.as_posix()),
                width=120,
                height=120,
            )
            for i in range(1, 4)
        ]

        results = asyncio.run(service.summarize_extracted_images(images))
        assert len(results) == 3
        for img in results:
            assert img.vlm_summary is not None
            assert f"page {img.page_number}" in img.vlm_summary

    def test_mocked_gemini_vision_call(self, test_image_path: Path) -> None:
        """Verify Gemini API integration flow when configured."""
        settings = Settings(
            vlm_provider="gemini",
            gemini_api_key="valid-mock-api-key",
            vlm_model_name="gemini-1.5-flash",
        )

        with patch("google.generativeai.configure") as mock_conf, patch(
            "google.generativeai.GenerativeModel"
        ) as mock_model_cls:
            mock_model_instance = MagicMock()
            mock_response = MagicMock()
            mock_response.text = (
                "A vertical bar chart titled 'Quarterly Latency' showing latency reduction from 120ms to 45ms across Q1-Q4."
            )
            mock_model_instance.generate_content.return_value = mock_response
            mock_model_cls.return_value = mock_model_instance

            service = VLMService(settings=settings)
            assert service.is_live_configured

            summary = asyncio.run(
                service.summarize_image(
                    image_path=test_image_path,
                    source_pdf="report.pdf",
                    page_number=2,
                )
            )

            assert "Quarterly Latency" in summary
            assert mock_model_instance.generate_content.called
            call_args = mock_model_instance.generate_content.call_args[0][0]
            assert call_args[0] == VLM_SYSTEM_PROMPT

    def test_mocked_openai_vision_call(self, test_image_path: Path) -> None:
        """Verify OpenAI GPT-4o-mini Vision integration flow when configured."""
        settings = Settings(
            vlm_provider="openai",
            openai_api_key="valid-mock-openai-key",
            vlm_model_name="gpt-4o-mini",
        )

        with patch("openai.AsyncOpenAI") as mock_openai_cls:
            mock_client = MagicMock()
            mock_completion = MagicMock()
            mock_choice = MagicMock()
            mock_choice.message.content = (
                "System architecture flowchart depicting data ingestion into vector DB and query synthesis."
            )
            mock_completion.choices = [mock_choice]
            mock_client.chat.completions.create = AsyncMock(return_value=mock_completion)
            mock_openai_cls.return_value = mock_client

            service = VLMService(settings=settings)
            assert service.is_live_configured

            summary = asyncio.run(
                service.summarize_image(
                    image_path=test_image_path,
                    source_pdf="arch.pdf",
                    page_number=1,
                )
            )

            assert "System architecture flowchart" in summary
            assert mock_client.chat.completions.create.called

    def test_graceful_degradation_on_vlm_failure(self, test_image_path: Path) -> None:
        """Verify that when the VLM client repeatedly fails, the pipeline returns a fallback description without crashing."""
        settings = Settings(
            vlm_provider="gemini",
            gemini_api_key="valid-mock-api-key",
            vlm_max_retries=2,
        )

        with patch("google.generativeai.configure"), patch("google.generativeai.GenerativeModel") as mock_model_cls:
            mock_model_instance = MagicMock()
            mock_model_instance.generate_content.side_effect = ConnectionError("API connection timeout")
            mock_model_cls.return_value = mock_model_instance

            service = VLMService(settings=settings)
            summary = asyncio.run(
                service.summarize_image(
                    image_path=test_image_path,
                    source_pdf="corrupt_api.pdf",
                    page_number=5,
                )
            )

            assert "summary unavailable" in summary
            assert "corrupt_api.pdf" in summary
            assert "page 5" in summary
