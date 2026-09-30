"""
SafetyCopilot — Embedding Model Test

Tests:
1. Model loading
2. Embedding dimensions
3. Vector normalization
4. Semantic similarity
"""

from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.embeddings.embedding_model import EmbeddingModel


def cosine_similarity(vector_a, vector_b) -> float:
    """Calculate cosine similarity between two vectors."""

    return float(
        np.dot(vector_a, vector_b)
        / (
            np.linalg.norm(vector_a)
            * np.linalg.norm(vector_b)
        )
    )


def main():

    print("=" * 90)
    print("  SAFETYCOPILOT — EMBEDDING MODEL TEST")
    print("=" * 90)

    model = EmbeddingModel()

    print(f"\nModel      : {model.model_name}")
    print(f"Dimension  : {model.dimension}")
    print(f"Device     : {model.device}")

    # -----------------------------------------------------
    # Semantic test sentences
    # -----------------------------------------------------

    text_a = "What PPE should workers use?"

    text_b = (
        "What personal protective equipment "
        "is required for workers?"
    )

    text_c = (
        "What are the dangers of working "
        "inside confined spaces?"
    )

    texts = [
        text_a,
        text_b,
        text_c,
    ]

    print("\nGenerating embeddings...")

    embeddings = model.embed_texts(texts)

    vector_a = embeddings[0]
    vector_b = embeddings[1]
    vector_c = embeddings[2]

    # -----------------------------------------------------
    # Basic validation
    # -----------------------------------------------------

    print("\n" + "-" * 90)
    print("VECTOR VALIDATION")
    print("-" * 90)

    print(f"\nEmbedding shape : {embeddings.shape}")

    print(
        f"Vector A norm   : "
        f"{np.linalg.norm(vector_a):.6f}"
    )

    print(
        f"Vector B norm   : "
        f"{np.linalg.norm(vector_b):.6f}"
    )

    print(
        f"Vector C norm   : "
        f"{np.linalg.norm(vector_c):.6f}"
    )

    # -----------------------------------------------------
    # Semantic similarity
    # -----------------------------------------------------

    similarity_ab = cosine_similarity(
        vector_a,
        vector_b,
    )

    similarity_ac = cosine_similarity(
        vector_a,
        vector_c,
    )

    print("\n" + "-" * 90)
    print("SEMANTIC SIMILARITY")
    print("-" * 90)

    print(f"\nA: {text_a}")
    print(f"B: {text_b}")
    print(f"C: {text_c}")

    print(
        f"\nSimilarity A ↔ B : "
        f"{similarity_ab:.4f}"
    )

    print(
        f"Similarity A ↔ C : "
        f"{similarity_ac:.4f}"
    )

    # -----------------------------------------------------
    # Final result
    # -----------------------------------------------------

    print("\n" + "=" * 90)
    print("FINAL RESULT")
    print("=" * 90)

    if (
        embeddings.shape[1] == model.dimension
        and similarity_ab > similarity_ac
    ):

        print(
            "\n[PASS] Embedding model is working correctly."
        )

        print(
            "Semantically related PPE queries received "
            "a higher similarity score."
        )

    else:

        print(
            "\n[WARNING] Embedding behavior requires review."
        )

    print("\n" + "=" * 90)


if __name__ == "__main__":
    main()