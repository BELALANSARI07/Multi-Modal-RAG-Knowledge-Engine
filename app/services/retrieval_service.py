"""Multi-modal retrieval and generation service (RAG orchestrator) with guardrails."""

import asyncio
import logging
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional

from app.config import Settings, get_settings
from app.models.schemas import QueryMetadata, QueryRequest, QueryResponse, SourceNode
from app.services.embedding_service import EmbeddingService
from app.services.vectorstore_service import VectorStoreService

logger = logging.getLogger(__name__)

RAG_SYSTEM_PROMPT_TEMPLATE = """You are a highly knowledgeable technical AI assistant answering questions based on multi-modal document context.
Use ONLY the provided context to answer the question. If the context is insufficient, clearly state that the information is not present. Do NOT speculate or fabricate facts.

Some context items are dense descriptions of charts, graphs, flowcharts, or architecture diagrams.
When citing visual data, explicitly note that the insight comes from a figure, diagram, or chart from the source document.

--- CONTEXT ---
{context_blocks}
--- END CONTEXT ---

Question: {question}

Provide an accurate, well-structured answer citing specific metrics, page numbers, and figures from the context."""


class RetrievalService:
    """Orchestrates query vectorization, top-k ChromaDB retrieval, context augmentation, and LLM synthesis."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        embedding_service: Optional[EmbeddingService] = None,
        vectorstore_service: Optional[VectorStoreService] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.embedder = embedding_service or EmbeddingService(self.settings)
        self.vectorstore = vectorstore_service or VectorStoreService(self.settings)
        self.min_relevance = self.settings.min_relevance_score
        self.max_context_chars = self.settings.max_context_chars

    async def query(self, request: QueryRequest) -> QueryResponse:
        """Execute end-to-end multi-modal RAG retrieval and answer generation with guardrails."""
        # 1. Measure Retrieval
        t_retrieval_start = time.perf_counter()

        # Generate query vector
        query_vector = await self.embedder.embed_single_async(request.question)

        # Build metadata filter if filter_source is specified
        where_filter: Optional[Dict[str, Any]] = None
        if request.filter_source:
            where_filter = {"source_pdf": request.filter_source}

        # Query ChromaDB collection
        search_results = await self.vectorstore.query_async(
            query_embeddings=[query_vector],
            n_results=request.top_k,
            where=where_filter,
        )

        retrieval_elapsed_ms = (time.perf_counter() - t_retrieval_start) * 1000.0

        # Parse retrieved matches
        sources: List[SourceNode] = []
        raw_documents = search_results.get("documents", [[]])[0]
        raw_metadatas = search_results.get("metadatas", [[]])[0]
        raw_distances = search_results.get("distances", [[]])[0]

        context_segments: List[str] = []
        accumulated_chars = 0

        for idx, (doc_text, meta, dist) in enumerate(zip(raw_documents, raw_metadatas, raw_distances)):
            source_pdf = meta.get("source_pdf", "Unknown")
            page_num = meta.get("page_number", 1)
            chunk_type = meta.get("chunk_type", "text")
            image_path = meta.get("image_path") or None

            # Calculate cosine similarity score from distance
            relevance_score = max(0.0, min(1.0, 1.0 - float(dist)))

            # Guardrail 1: Relevance Score Threshold
            if relevance_score < self.min_relevance:
                logger.debug(
                    "Filtering out chunk from %s (score %.2f < threshold %.2f)",
                    source_pdf,
                    relevance_score,
                    self.min_relevance,
                )
                continue

            # Build web-accessible URL for static image rendering
            image_url = None
            if image_path:
                norm_path = Path(image_path).as_posix()
                if "extracted_images" in norm_path:
                    sub = norm_path.split("extracted_images/")[-1]
                    image_url = f"/static/images/{sub}"

            source_node = SourceNode(
                content_preview=doc_text[:200] + ("..." if len(doc_text) > 200 else ""),
                source_pdf=source_pdf,
                page_number=page_num,
                chunk_type=chunk_type,
                relevance_score=round(relevance_score, 4),
                image_path=image_path,
                image_url=image_url,
            )
            sources.append(source_node)

            type_label = "Visual Graphic / Chart" if chunk_type == "image_summary" else "Text"
            segment = f"[Source: {source_pdf} | Page {page_num} | Content: {type_label}]\n{doc_text}"

            # Guardrail 2: Context Window Truncation Protection
            if accumulated_chars + len(segment) <= self.max_context_chars:
                context_segments.append(segment)
                accumulated_chars += len(segment)
            else:
                remaining_space = self.max_context_chars - accumulated_chars
                if remaining_space > 100:
                    context_segments.append(segment[:remaining_space] + "... [context truncated]")
                break

        # Handle empty context scenario
        if not context_segments:
            return QueryResponse(
                question=request.question,
                answer="No relevant documents or visual data above the minimum confidence threshold were found to answer your question.",
                sources=[],
                metadata=QueryMetadata(
                    model_used="none",
                    retrieval_time_ms=round(retrieval_elapsed_ms, 2),
                    generation_time_ms=0.0,
                    total_sources=0,
                ),
            )

        # 2. Measure Generation
        t_gen_start = time.perf_counter()
        context_block_str = "\n\n".join(context_segments)
        prompt = RAG_SYSTEM_PROMPT_TEMPLATE.format(
            context_blocks=context_block_str,
            question=request.question,
        )

        answer_text, model_name = await self._generate_answer(prompt, context_segments, request.question)
        generation_elapsed_ms = (time.perf_counter() - t_gen_start) * 1000.0

        return QueryResponse(
            question=request.question,
            answer=answer_text,
            sources=sources if request.include_sources else [],
            metadata=QueryMetadata(
                model_used=model_name,
                retrieval_time_ms=round(retrieval_elapsed_ms, 2),
                generation_time_ms=round(generation_elapsed_ms, 2),
                total_sources=len(sources),
            ),
        )

    async def _generate_answer(
        self,
        prompt: str,
        context_segments: List[str],
        question: str,
    ) -> tuple[str, str]:
        """Synthesize answer using configured LLM or intelligent grounded fallback."""
        provider = self.settings.vlm_provider.lower()

        # 1. Try Gemini LLM
        if provider == "gemini":
            key = self.settings.gemini_api_key or os.getenv("GEMINI_API_KEY")
            if key and not key.startswith("your_"):
                try:
                    import google.generativeai as genai

                    genai.configure(api_key=key)
                    model = genai.GenerativeModel(self.settings.vlm_model_name)

                    def _sync_call():
                        resp = model.generate_content(prompt)
                        return resp.text

                    res = await asyncio.to_thread(_sync_call)
                    return res.strip(), f"gemini/{self.settings.vlm_model_name}"
                except Exception as e:
                    logger.warning("Gemini synthesis call failed (%s), falling back to grounded synthesizer.", e)

        # 2. Try OpenAI LLM
        elif provider == "openai":
            key = self.settings.openai_api_key or os.getenv("OPENAI_API_KEY")
            if key and not key.startswith("your_"):
                try:
                    from openai import AsyncOpenAI

                    client = AsyncOpenAI(api_key=key)
                    resp = await client.chat.completions.create(
                        model=self.settings.vlm_model_name,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.2,
                    )
                    return resp.choices[0].message.content.strip(), f"openai/{self.settings.vlm_model_name}"
                except Exception as e:
                    logger.warning("OpenAI synthesis call failed (%s), falling back to grounded synthesizer.", e)

        # 3. Grounded Fallback Synthesizer (Zero hallucination for local testing / unconfigured credentials)
        logger.info("Using grounded local synthesis engine.")
        summarized_answer = (
            f"Based on the retrieved multi-modal context:\n\n"
            f"{context_segments[0]}\n\n"
            f"(Synthesized from {len(context_segments)} retrieved document and visual sources)."
        )
        return summarized_answer, "local-grounded-synthesizer"
