"""
SafetyCopilot — BM25 Retriever

Provides lexical/keyword-based retrieval over SafetyCopilot
HSE chunks using BM25.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

from rank_bm25 import BM25Okapi

from src.ingestion.pdf_loader import DocumentData
from src.ingestion.chunker import ChunkData, chunk_documents


def _tokenize(text: str) -> List[str]:
    """
    Convert text into lowercase searchable tokens.

    Keeps letters, numbers and common HSE-style terms while
    removing punctuation and unnecessary whitespace.
    """

    if not text:
        return []

    return re.findall(
        r"[a-zA-Z0-9]+(?:[-'][a-zA-Z0-9]+)*",
        text.lower(),
    )


class BM25Retriever:
    """BM25 keyword retriever for HSE document chunks."""

    def __init__(
        self,
        chunks: List[ChunkData],
    ) -> None:

        if not chunks:
            raise ValueError(
                "BM25Retriever requires at least one chunk."
            )

        self.chunks = chunks

        self.tokenized_corpus = [
            _tokenize(chunk.text)
            for chunk in chunks
        ]

        self.bm25 = BM25Okapi(
            self.tokenized_corpus
        )

    @classmethod
    def from_documents(
        cls,
        documents: List[DocumentData],
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ) -> "BM25Retriever":
        """Create BM25 retriever directly from parsed documents."""

        chunks = chunk_documents(
            documents=documents,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        return cls(chunks)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Retrieve highest-scoring chunks for a keyword query."""

        if not query or not query.strip():
            raise ValueError(
                "Query cannot be empty."
            )

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than 0."
            )

        query_tokens = _tokenize(query)

        if not query_tokens:
            return []

        scores = self.bm25.get_scores(
            query_tokens
        )

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True,
        )[:top_k]

        results: List[Dict[str, Any]] = []

        for rank, index in enumerate(
            ranked_indices,
            start=1,
        ):

            chunk = self.chunks[index]

            results.append(
                {
                    "rank": rank,
                    "score": float(scores[index]),
                    "text": chunk.text,
                    "chunk_id": chunk.chunk_id,
                    "filename": chunk.filename,
                    "category": chunk.category,
                    "page_number": chunk.page_number,
                    "total_pages": chunk.total_pages,
                }
            )

        return results