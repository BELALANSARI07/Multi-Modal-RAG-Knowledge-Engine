"""Vision-Language Model (VLM) service for generating dense semantic descriptions of visuals."""

import asyncio
import base64
import logging
import os
import time
from pathlib import Path
from typing import List, Optional, Union

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.config import Settings, get_settings
from app.core.exceptions import VLMServiceError
from app.models.enums import VLMProvider
from app.models.schemas import ExtractedImage

logger = logging.getLogger(__name__)

VLM_SYSTEM_PROMPT = (
    "You are an expert technical document analysis assistant specializing in data visuals.\n"
    "Describe the visual content of this image in rich, precise, factual detail.\n"
    "Structure your description to include:\n"
    "1. The exact TYPE of visual (e.g. bar chart, line chart, scatter plot, flowchart, architecture diagram, tabular graphic, heatmap).\n"
    "2. ALL readable text: titles, axis labels, units of measurement, legend entries, callouts, and data points.\n"
    "3. Key trends, quantitative comparisons, anomalies, or relationships conveyed.\n"
    "4. Structural flow or component hierarchy if this is a diagram or schema.\n\n"
    "Output ONLY a dense, information-rich descriptive paragraph suitable for semantic indexing and vector retrieval. "
    "Do NOT add conversational preambles, assumptions, or subjective interpretations."
)


# Specific transient exception types eligible for retry (never retry logic/programming errors)
TRANSIENT_EXCEPTIONS = (
    httpx.ConnectError,
    httpx.ConnectTimeout,
    httpx.ReadTimeout,
    httpx.WriteTimeout,
    ConnectionError,
    TimeoutError,
    asyncio.TimeoutError,
)


class VLMService:
    """Orchestrates image summarization requests against Gemini or OpenAI Vision APIs."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()
        self.provider = VLMProvider(self.settings.vlm_provider.lower())
        self.semaphore = asyncio.Semaphore(self.settings.vlm_max_concurrent)
        self.timeout_seconds = self.settings.vlm_timeout_seconds
        self._gemini_configured = False
        self._openai_client = None

        self._initialize_provider()

    def _initialize_provider(self) -> None:
        """Configure client libraries based on the selected provider."""
        if self.provider == VLMProvider.GEMINI:
            key = self.settings.gemini_api_key or os.getenv("GEMINI_API_KEY")
            if key and not key.startswith("your_"):
                try:
                    import google.generativeai as genai

                    genai.configure(api_key=key)
                    self._gemini_configured = True
                    logger.info("Configured Gemini VLM with model: %s", self.settings.vlm_model_name)
                except Exception as e:
                    logger.warning("Failed to configure google.generativeai: %s", e)
            else:
                logger.warning("Gemini API key is not configured or is a placeholder.")

        elif self.provider == VLMProvider.OPENAI:
            key = self.settings.openai_api_key or os.getenv("OPENAI_API_KEY")
            if key and not key.startswith("your_"):
                try:
                    from openai import AsyncOpenAI

                    self._openai_client = AsyncOpenAI(api_key=key, timeout=self.timeout_seconds)
                    logger.info("Configured OpenAI Vision client with model: %s", self.settings.vlm_model_name)
                except Exception as e:
                    logger.warning("Failed to configure AsyncOpenAI: %s", e)
            else:
                logger.warning("OpenAI API key is not configured or is a placeholder.")

    @property
    def is_live_configured(self) -> bool:
        """Check if a live API client is ready to receive requests."""
        if self.provider == VLMProvider.GEMINI:
            return self._gemini_configured
        elif self.provider == VLMProvider.OPENAI:
            return self._openai_client is not None
        return False

    async def summarize_image(
        self,
        image_path: Union[str, Path],
        source_pdf: str = "",
        page_number: int = 1,
    ) -> str:
        """Generate a dense semantic summary of a single extracted visual image."""
        path = Path(image_path)
        if not path.is_file():
            raise VLMServiceError(f"Image file not found: {path}")

        # If live API credentials are not set, fall back to mock descriptive summary
        if not self.is_live_configured:
            logger.info("VLM API credentials not detected. Generating synthetic visual summary for %s", path.name)
            return self._generate_synthetic_summary(path, source_pdf, page_number)

        async with self.semaphore:
            start_time = time.perf_counter()
            try:
                # Wrap API call with timeout protection
                summary = await asyncio.wait_for(
                    self._call_vlm_with_retry(path),
                    timeout=self.timeout_seconds,
                )
                elapsed = time.perf_counter() - start_time
                logger.debug(
                    "Generated VLM summary for %s in %.2fs (%d chars)",
                    path.name,
                    elapsed,
                    len(summary),
                )
                return summary.strip()
            except Exception as e:
                logger.error(
                    "VLM summarization failed for %s on page %d of %s: %s",
                    path.name,
                    page_number,
                    source_pdf,
                    str(e),
                )
                # Graceful degradation: return a factual fallback description
                return f"[Visual Content from page {page_number} of {source_pdf}: image {path.name} (summary unavailable: {str(e)})]"

    async def summarize_extracted_images(
        self,
        images: List[ExtractedImage],
    ) -> List[ExtractedImage]:
        """Process a list of ExtractedImage instances concurrently and populate vlm_summary."""
        if not images:
            return []

        async def _process_single(img: ExtractedImage) -> ExtractedImage:
            summary = await self.summarize_image(
                image_path=img.local_path,
                source_pdf=img.source_pdf,
                page_number=img.page_number,
            )
            img.vlm_summary = summary
            return img

        tasks = [_process_single(img) for img in images]
        return await asyncio.gather(*tasks)

    async def _call_vlm_with_retry(self, image_path: Path) -> str:
        """Execute the VLM API call with retry restricted strictly to transient network exceptions."""

        @retry(
            reraise=True,
            stop=stop_after_attempt(self.settings.vlm_max_retries),
            wait=wait_exponential(multiplier=1, min=2, max=10),
            retry=retry_if_exception_type(TRANSIENT_EXCEPTIONS),
        )
        async def _attempt_call() -> str:
            if self.provider == VLMProvider.GEMINI:
                return await self._call_gemini(image_path)
            elif self.provider == VLMProvider.OPENAI:
                return await self._call_openai(image_path)
            else:
                raise VLMServiceError(f"Unsupported VLM provider: {self.provider}")

        return await _attempt_call()

    async def _call_gemini(self, image_path: Path) -> str:
        """Call Google Gemini Vision API."""
        import google.generativeai as genai
        from PIL import Image

        model = genai.GenerativeModel(self.settings.vlm_model_name)

        def _sync_generate() -> str:
            with Image.open(image_path) as pil_img:
                response = model.generate_content([VLM_SYSTEM_PROMPT, pil_img])
                if not response.text:
                    raise VLMServiceError("Empty response text from Gemini Vision API")
                return response.text

        return await asyncio.to_thread(_sync_generate)

    async def _call_openai(self, image_path: Path) -> str:
        """Call OpenAI GPT-4o-mini Vision API."""
        if not self._openai_client:
            raise VLMServiceError("OpenAI client is not initialized")

        with open(image_path, "rb") as f:
            encoded_bytes = base64.b64encode(f.read()).decode("utf-8")

        ext = image_path.suffix.lstrip(".").lower()
        mime_type = f"image/{ext}" if ext in ("png", "jpeg", "jpg", "webp") else "image/png"
        data_url = f"data:{mime_type};base64,{encoded_bytes}"

        response = await self._openai_client.chat.completions.create(
            model=self.settings.vlm_model_name,
            messages=[
                {"role": "system", "content": VLM_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe the data and content of this image in detail."},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                },
            ],
            max_tokens=500,
            temperature=0.2,
        )

        content = response.choices[0].message.content
        if not content:
            raise VLMServiceError("Empty content returned from OpenAI Vision API")
        return content

    def _generate_synthetic_summary(self, path: Path, source_pdf: str, page_number: int) -> str:
        """Generate a deterministic synthetic visual summary for local development & unit testing."""
        return (
            f"Visual graphic extracted from {source_pdf or 'document'} on page {page_number}. "
            f"File: {path.name}. Displays structured visual metrics, diagrams, or comparative chart representations "
            f"illustrating domain trends and technical data relevant to this document section."
        )
