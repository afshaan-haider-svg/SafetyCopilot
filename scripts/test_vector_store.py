"""
SafetyCopilot — Qdrant Vector Store Test

Tests:
1. Embedding generation
2. Local Qdrant collection creation
3. Vector insertion
4. Semantic retrieval
5. Payload/metadata retrieval
"""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.embeddings.embedding_model import EmbeddingModel
from src.vectorstore.qdrant_store import QdrantVectorStore


def main():

    print("=" * 90)
    print("  SAFETYCOPILOT — QDRANT VECTOR STORE TEST")
    print("=" * 90)

    # -----------------------------------------------------
    # Test documents
    # -----------------------------------------------------

    texts = [
        "Workers must wear suitable personal protective equipment.",
        "Confined spaces may contain dangerous gases or low oxygen levels.",
        "Electrical equipment should be isolated before maintenance work.",
    ]

    categories = [
        "ppe",
        "confined_space",
        "electrical_safety",
    ]

    # -----------------------------------------------------
    # Embeddings
    # -----------------------------------------------------

    print("\nLoading embedding model...")

    embedding_model = EmbeddingModel()

    embeddings = embedding_model.embed_texts(texts)

    print(
        f"Generated {len(embeddings)} embeddings "
        f"with dimension {embeddings.shape[1]}."
    )

    # -----------------------------------------------------
    # Vector store
    # -----------------------------------------------------

    store = QdrantVectorStore(
        collection_name="safetycopilot_test",
        vector_size=embedding_model.dimension,
        storage_path=PROJECT_ROOT / "data" / "qdrant_test",
    )

    try:

        print("\nCreating local Qdrant collection...")

        store.create_collection(
            recreate=True
        )

        payloads = []

        for index, text in enumerate(texts):

            payloads.append(
                {
                    "text": text,
                    "category": categories[index],
                    "test_id": index + 1,
                }
            )

        print("Adding 3 test vectors...")

        store.add_points(
            ids=[1, 2, 3],
            vectors=embeddings.tolist(),
            payloads=payloads,
        )

        print(
            f"Points stored: {store.count()}"
        )

        # -------------------------------------------------
        # Search
        # -------------------------------------------------

        query = (
            "What protective equipment should "
            "a worker wear?"
        )

        print(f"\nQuery: {query}")

        query_vector = embedding_model.embed_text(
            query
        )

        results = store.search(
            query_vector=query_vector.tolist(),
            limit=3,
        )

        print("\nSEARCH RESULTS")
        print("-" * 90)

        for rank, result in enumerate(
            results,
            start=1,
        ):

            print(
                f"\nRank     : {rank}"
            )

            print(
                f"Score    : {result.score:.4f}"
            )

            print(
                f"Category : "
                f"{result.payload.get('category')}"
            )

            print(
                f"Text     : "
                f"{result.payload.get('text')}"
            )

        # -------------------------------------------------
        # Validation
        # -------------------------------------------------

        print("\n" + "=" * 90)
        print("FINAL RESULT")
        print("=" * 90)

        top_category = (
            results[0]
            .payload
            .get("category")
        )

        if (
            store.count() == 3
            and top_category == "ppe"
        ):

            print(
                "\n[PASS] Qdrant vector store "
                "is working correctly."
            )

            print(
                "The PPE query retrieved the "
                "PPE vector as the top result."
            )

        else:

            print(
                "\n[WARNING] Vector retrieval "
                "requires review."
            )

    finally:

        store.close()

    print("\n" + "=" * 90)


if __name__ == "__main__":
    main()