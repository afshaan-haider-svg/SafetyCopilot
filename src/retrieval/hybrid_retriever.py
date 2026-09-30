"""
SafetyCopilot — Hybrid Retriever

Combines:
1. Dense semantic retrieval from Qdrant
2. BM25 lexical retrieval
3. Reciprocal Rank Fusion (RRF)

No raw-score mixing is performed because dense cosine
scores and BM25 scores use different scales.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from src.retrieval.retriever import SafetyRetriever
from src.retrieval.bm25_retriever import BM25Retriever


class HybridRetriever:
    """Combine dense and BM25 retrieval using RRF."""

    def __init__(
        self,
        dense_retriever: SafetyRetriever,
        bm25_retriever: BM25Retriever,
        rrf_k: int = 60,
    ) -> None:

        if rrf_k <= 0:
            raise ValueError("rrf_k must be greater than 0.")

        self.dense_retriever = dense_retriever
        self.bm25_retriever = bm25_retriever
        self.rrf_k = rrf_k

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        candidate_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve and fuse dense + BM25 results.

        Parameters
        ----------
        query:
            User question.

        top_k:
            Number of final hybrid results.

        candidate_k:
            Number of candidates retrieved independently
            from dense and BM25 retrieval.
        """

        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        if top_k <= 0:
            raise ValueError("top_k must be greater than 0.")

        if candidate_k < top_k:
            candidate_k = top_k

        # -------------------------------------------------
        # Dense retrieval
        # -------------------------------------------------

        dense_results = self.dense_retriever.retrieve(
            query=query,
            top_k=candidate_k,
        )

        # -------------------------------------------------
        # BM25 retrieval
        # -------------------------------------------------

        bm25_results = self.bm25_retriever.retrieve(
            query=query,
            top_k=candidate_k,
        )

        # -------------------------------------------------
        # Reciprocal Rank Fusion
        # -------------------------------------------------

        fused: Dict[str, Dict[str, Any]] = {}

        for result in dense_results:

            chunk_id = result["chunk_id"]

            if not chunk_id:
                continue

            if chunk_id not in fused:
                fused[chunk_id] = {
                    **result,
                    "dense_rank": None,
                    "dense_score": None,
                    "bm25_rank": None,
                    "bm25_score": None,
                    "rrf_score": 0.0,
                }

            fused[chunk_id]["dense_rank"] = result["rank"]
            fused[chunk_id]["dense_score"] = result["score"]

            fused[chunk_id]["rrf_score"] += (
                1.0 / (self.rrf_k + result["rank"])
            )

        for result in bm25_results:

            chunk_id = result["chunk_id"]

            if not chunk_id:
                continue

            if chunk_id not in fused:
                fused[chunk_id] = {
                    **result,
                    "dense_rank": None,
                    "dense_score": None,
                    "bm25_rank": None,
                    "bm25_score": None,
                    "rrf_score": 0.0,
                }

            fused[chunk_id]["bm25_rank"] = result["rank"]
            fused[chunk_id]["bm25_score"] = result["score"]

            fused[chunk_id]["rrf_score"] += (
                1.0 / (self.rrf_k + result["rank"])
            )

        # -------------------------------------------------
        # Sort by fused RRF score
        # -------------------------------------------------

        ranked_results = sorted(
            fused.values(),
            key=lambda item: item["rrf_score"],
            reverse=True,
        )

        final_results = ranked_results[:top_k]

        for final_rank, result in enumerate(
            final_results,
            start=1,
        ):
            result["rank"] = final_rank

        return final_results

    def close(self) -> None:
        """Close dense vector-store connection."""

        self.dense_retriever.close()