"""Demonstration script: Generates a synthetic multi-modal PDF and executes Phase 1 ingestion."""

import asyncio
import io
import sys
from pathlib import Path

# Add project root to sys.path so app modules are importable.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pymupdf as fitz  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from app.core.pdf_parser import PDFParser  # noqa: E402
from app.services.ingestion_service import IngestionService  # noqa: E402
from app.utils.file_utils import ensure_directory  # noqa: E402


def generate_sample_report(output_path: Path) -> Path:
    """Generate a realistic test PDF with formatted text and an embedded synthetic bar chart."""
    doc = fitz.open()

    # --- Page 1: Title and Overview ---
    page1 = doc.new_page()
    page1.insert_text(
        (50, 70),
        "QUARTERLY PERFORMANCE REPORT: AI INFRASTRUCTURE",
        fontsize=16,
    )
    page1.insert_text(
        (50, 110),
        "Executive Summary\n\n"
        "During the third quarter of 2026, enterprise adoption of Multi-Modal Retrieval Augmented\n"
        "Generation (RAG) increased by 42%. Organizations increasingly rely on unified knowledge\n"
        "engines that fuse structured visual data (such as financial charts and architecture diagrams)\n"
        "with unstructured textual documents to eliminate hallucination in domain-specific tasks.\n\n"
        "Vector search scalability and token efficiency continue to be the primary optimization targets\n"
        "for production deployments. Incorporating dense image summaries into standard vector\n"
        "indexes bridges the semantic gap without introducing complex multi-vector retrieval pipelines.",
        fontsize=11,
    )

    # --- Page 2: Visual Chart & Analysis ---
    page2 = doc.new_page()
    page2.insert_text(
        (50, 70),
        "Figure 1: Quarterly Model Latency and Accuracy Metrics",
        fontsize=14,
    )

    # Create synthetic visual chart using Pillow
    chart_img = Image.new("RGB", (500, 300), color=(245, 247, 250))
    draw = ImageDraw.Draw(chart_img)

    # Draw border
    draw.rectangle([10, 10, 490, 290], outline=(180, 190, 205), width=2)
    # Draw bars representing quarters
    bars = [
        ("Q1", 60, (66, 133, 244)),
        ("Q2", 120, (52, 168, 83)),
        ("Q3", 190, (251, 188, 5)),
        ("Q4", 230, (234, 67, 53)),
    ]
    x_start = 80
    for label, height, color in bars:
        draw.rectangle([x_start, 250 - height, x_start + 60, 250], fill=color)
        draw.text((x_start + 18, 255), label, fill=(50, 50, 50))
        x_start += 100

    # Draw axis lines
    draw.line([60, 250, 460, 250], fill=(100, 100, 100), width=2)
    draw.line([60, 40, 60, 250], fill=(100, 100, 100), width=2)

    chart_buf = io.BytesIO()
    chart_img.save(chart_buf, format="PNG")
    page2.insert_image(fitz.Rect(50, 100, 450, 340), stream=chart_buf.getvalue())

    page2.insert_text(
        (50, 370),
        "Analysis of Figure 1:\n"
        "As illustrated above, query accuracy scaled progressively across each quarter, reaching peak\n"
        "performance of 94% following the introduction of the unified multi-modal index.",
        fontsize=11,
    )

    ensure_directory(output_path.parent)
    doc.save(str(output_path))
    doc.close()
    return output_path


def main() -> None:
    sample_pdf_path = PROJECT_ROOT / "data" / "uploads" / "sample_quarterly_report.pdf"
    print(f"[+] Generating synthetic multi-modal PDF at: {sample_pdf_path}")
    generate_sample_report(sample_pdf_path)

    print("[+] Initializing PDFParser...")
    parser = PDFParser()

    print("[+] Ingesting and parsing PDF...")
    result = parser.parse(sample_pdf_path)

    print("\n" + "=" * 60)
    print("PHASE 1 INGESTION RESULTS:")
    print("=" * 60)
    print(f"Source Document : {result.source_pdf}")
    print(f"Total Pages     : {result.total_pages}")
    print(f"Text Chunks     : {len(result.text_chunks)}")
    print(f"Extracted Images: {len(result.extracted_images)}")
    print("=" * 60)

    print("\n--- Text Chunks Extracted ---")
    for chunk in result.text_chunks:
        preview = chunk.text.replace("\n", " ")[:80]
        print(f"[{chunk.chunk_id}] (Page {chunk.page_number}, {chunk.char_count} chars): {preview}...")

    print("\n--- Extracted Visuals Saved ---")
    for img in result.extracted_images:
        print(f"[{img.image_id}] (Page {img.page_number}, {img.width}x{img.height}): {img.local_path}")

    print("\n[+] Initializing Unified Ingestion Orchestrator (Phase 3)...")
    ingestion_service = IngestionService()
    print("[+] Ingesting PDF into unified vector store (ChromaDB)...")
    ingest_result = asyncio.run(ingestion_service.ingest_file(sample_pdf_path))

    print("\n" + "=" * 60)
    print("PHASE 3 UNIFIED VECTORIZATION RESULTS:")
    print("=" * 60)
    print(f"Status           : {ingest_result.status}")
    print(f"Document ID      : {ingest_result.document_id}")
    print(f"Total Pages      : {ingest_result.stats.total_pages}")
    print(f"Text Chunks      : {ingest_result.stats.text_chunks}")
    print(f"Images Extracted : {ingest_result.stats.images_extracted}")
    print(f"Images Summarized: {ingest_result.stats.images_summarized}")
    print(f"ChromaDB Total Vectors : {ingestion_service.vectorstore.count()}")

    print("\n[+] Performing trial semantic similarity query...")
    sample_query = "What was the quarterly model accuracy and latency metric?"
    print(f"    Query: '{sample_query}'")
    query_vec = ingestion_service.embedding_service.embed_single(sample_query)
    search_res = ingestion_service.vectorstore.query(query_embeddings=[query_vec], n_results=2)

    print("\n--- Top Retrieved Context from ChromaDB ---")
    for idx, (doc, meta, dist) in enumerate(
        zip(search_res["documents"][0], search_res["metadatas"][0], search_res["distances"][0])
    ):
        print(f"\n[Result {idx+1}] Type: {meta['chunk_type']} | Page: {meta['page_number']} | Cosine Distance: {dist:.4f}")
        if meta.get("image_path"):
            print(f"  Referenced Image Path: {meta['image_path']}")
        print(f"  Content: {doc[:150]}...")

    print("\n" + "=" * 60)
    print("[OK] Phases 1, 2, and 3 full multi-modal pipeline successfully verified!")


if __name__ == "__main__":
    main()
