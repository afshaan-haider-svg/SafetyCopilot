"""
SafetyCopilot — Document Chunker

Splits page-level HSE document text into smaller overlapping chunks
while preserving source metadata for retrieval and citations.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List, Dict, Any

from src.ingestion.pdf_loader import DocumentData


@dataclass
class ChunkData:
    """Structured representation of a single text chunk."""

    text: str
    chunk_id: str
    chunk_index: int
    filename: str
    source_path: str
    category: str
    page_number: int
    total_pages: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _split_text_recursive(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[str]:
    """
    Split text into overlapping chunks while preferring natural
    paragraph and sentence boundaries.
    """

    if not text or not text.strip():
        return []

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0.")

    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative.")

    if chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be smaller than chunk_size."
        )

    text = text.strip()

    # Short text does not need splitting.
    if len(text) <= chunk_size:
        return [text]

    chunks: List[str] = []
    start = 0
    text_length = len(text)

    while start < text_length:
        target_end = min(start + chunk_size, text_length)
        end = target_end

        # If this is not the final chunk, try to finish at a
        # natural boundary instead of cutting text abruptly.
        if target_end < text_length:
            search_start = start + (chunk_size // 2)

            candidate_boundaries = []

            for separator in ("\n\n", "\n", ". ", "? ", "! "):
                position = text.rfind(
                    separator,
                    search_start,
                    target_end,
                )

                if position != -1:
                    # Include sentence punctuation where appropriate.
                    if separator in (". ", "? ", "! "):
                        position += 1

                    candidate_boundaries.append(position)

            if candidate_boundaries:
                end = max(candidate_boundaries)

        chunk_text = text[start:end].strip()

        if chunk_text:
            chunks.append(chunk_text)

        # Finished document/page.
        if end >= text_length:
            break

        # Create overlap with previous chunk.
        next_start = max(0, end - chunk_overlap)

        # Safety check to guarantee forward progress.
        if next_start <= start:
            next_start = end

        start = next_start

    return chunks


def chunk_document(
    document: DocumentData,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[ChunkData]:
    """
    Convert a parsed PDF document into metadata-rich chunks.

    Chunking is performed page-by-page so page-level citation
    information remains accurate.
    """

    chunks: List[ChunkData] = []
    global_chunk_index = 0

    for page in document.pages:
        page_chunks = _split_text_recursive(
            text=page.text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        for page_chunk_index, chunk_text in enumerate(page_chunks):
            chunk_id = (
                f"{document.filename}"
                f"::page_{page.page_number}"
                f"::chunk_{page_chunk_index}"
            )

            chunks.append(
                ChunkData(
                    text=chunk_text,
                    chunk_id=chunk_id,
                    chunk_index=global_chunk_index,
                    filename=document.filename,
                    source_path=document.source_path,
                    category=document.category,
                    page_number=page.page_number,
                    total_pages=document.total_pages,
                )
            )

            global_chunk_index += 1

    return chunks


def chunk_documents(
    documents: List[DocumentData],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[ChunkData]:
    """Chunk multiple parsed PDF documents."""

    all_chunks: List[ChunkData] = []

    for document in documents:
        document_chunks = chunk_document(
            document=document,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        all_chunks.extend(document_chunks)

    return all_chunks