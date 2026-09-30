"""
SafetyCopilot — Cross-Encoder Reranker

Reranks retrieval candidates by jointly evaluating
the user query and each candidate chunk.
"""

from __future__ import annotations

from typing import Any, Dict, List

from sentence_transformers import CrossEncoder


DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class SafetyReranker:
    """Cross-encoder reranker for SafetyCopilot."""

    def __init__(
        self,
        model_name: str = DEFAULT_RERANKER_MODEL,
        device: str = "cpu",
    ) -> None:

        self.model_name = model_name
        self.device = device

        print(f"Loading reranker model: {model_name}")

        self.model = CrossEncoder(
            model_name,
            device=device,
        )

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:

        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        if not candidates:
            return []

        if top_k <= 0:
            raise ValueError("top_k must be greater than 0.")

        pairs = [
            [query, candidate["text"]]
            for candidate in candidates
        ]

        scores = self.model.predict(
            pairs,
            show_progress_bar=False,
        )

        reranked = []

        for candidate, score in zip(
            candidates,
            scores,
        ):
            item = candidate.copy()

            # Preserve the position produced by hybrid retrieval.
            item["hybrid_rank"] = candidate.get("rank")
            item["reranker_score"] = float(score)

            reranked.append(item)

        reranked.sort(
            key=lambda item: item["reranker_score"],
            reverse=True,
        )

        reranked = reranked[:top_k]

        for rank, item in enumerate(
            reranked,
            start=1,
        ):
            item["rank"] = rank

        return reranked