"""
SafetyCopilot — Embedding Model

Converts text into dense numerical vectors for semantic retrieval.
Uses Sentence Transformers with normalized embeddings.
"""

from __future__ import annotations

from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer


DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class EmbeddingModel:
    """Wrapper around a Sentence Transformer embedding model."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        device: str = "cpu",
    ) -> None:

        self.model_name = model_name
        self.device = device

        print(f"Loading embedding model: {model_name}")

        self.model = SentenceTransformer(
            model_name,
            device=device,
        )

        self.dimension = self.model.get_embedding_dimension()

    def embed_text(self, text: str) -> np.ndarray:
        """Generate one normalized embedding vector."""

        if not text or not text.strip():
            raise ValueError("Cannot embed empty text.")

        embedding = self.model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        return embedding

    def embed_texts(
        self,
        texts: List[str],
        batch_size: int = 32,
    ) -> np.ndarray:
        """Generate normalized embeddings for multiple texts."""

        if not texts:
            raise ValueError("Text list cannot be empty.")

        if any(not text or not text.strip() for text in texts):
            raise ValueError("Text list contains empty text.")

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

        return embeddings