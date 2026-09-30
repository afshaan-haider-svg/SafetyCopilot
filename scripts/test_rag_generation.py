"""
SafetyCopilot — End-to-End RAG Generation Test

Pipeline:
Question
→ Dense Retrieval
→ BM25
→ RRF Hybrid Fusion
→ Cross-Encoder Reranking
→ Gemini
→ Grounded Answer + Sources
"""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.ingestion.pdf_loader import (
    load_pdfs_from_directory,
)

from src.ingestion.chunker import (
    chunk_documents,
)

from src.retrieval.retriever import (
    SafetyRetriever,
)

from src.retrieval.bm25_retriever import (
    BM25Retriever,
)

from src.retrieval.hybrid_retriever import (
    HybridRetriever,
)

from src.retrieval.reranker import (
    SafetyReranker,
)

from src.generation.rag_generator import (
    RAGGenerator,
)


RAW_DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
)

QDRANT_PATH = (
    PROJECT_ROOT
    / "data"
    / "qdrant"
)


QUESTION = (
    "What are the main dangers of working "
    "inside a confined space?"
)


def main():

    print("=" * 100)
    print(
        "  SAFETYCOPILOT — END-TO-END RAG TEST"
    )
    print("=" * 100)

    # -----------------------------------------------------
    # 1. Documents
    # -----------------------------------------------------

    print("\n[1/6] Loading HSE documents...")

    documents = load_pdfs_from_directory(
        RAW_DATA_DIR,
        recursive=True,
    )

    chunks = chunk_documents(
        documents=documents,
        chunk_size=1000,
        chunk_overlap=200,
    )

    print(
        f"      Documents : {len(documents)}"
    )

    print(
        f"      Chunks    : {len(chunks)}"
    )

    # -----------------------------------------------------
    # 2. BM25
    # -----------------------------------------------------

    print(
        "\n[2/6] Building BM25 retriever..."
    )

    bm25_retriever = BM25Retriever(
        chunks=chunks
    )

    # -----------------------------------------------------
    # 3. Dense
    # -----------------------------------------------------

    print(
        "\n[3/6] Loading dense retriever..."
    )

    dense_retriever = SafetyRetriever(
        collection_name="safetycopilot",
        storage_path=QDRANT_PATH,
        top_k=15,
    )

    hybrid_retriever = HybridRetriever(
        dense_retriever=dense_retriever,
        bm25_retriever=bm25_retriever,
        rrf_k=60,
    )

    # -----------------------------------------------------
    # 4. Reranker
    # -----------------------------------------------------

    print(
        "\n[4/6] Loading cross-encoder..."
    )

    reranker = SafetyReranker()

    # -----------------------------------------------------
    # 5. Retrieve
    # -----------------------------------------------------

    try:

        print(
            "\n[5/6] Retrieving HSE evidence..."
        )

        print(
            f"\nQuestion: {QUESTION}"
        )

        hybrid_candidates = (
            hybrid_retriever.retrieve(
                query=QUESTION,
                top_k=10,
                candidate_k=15,
            )
        )

        evidence = reranker.rerank(
            query=QUESTION,
            candidates=hybrid_candidates,
            top_k=3,
        )

        print(
            f"\n      Evidence chunks: "
            f"{len(evidence)}"
        )

        for number, item in enumerate(
            evidence,
            start=1,
        ):

            print(
                f"\n      [{number}] "
                f"{item['filename']}"
            )

            print(
                f"          Page     : "
                f"{item['page_number']}"
            )

            print(
                f"          Category : "
                f"{item['category']}"
            )

            print(
                f"          Reranker : "
                f"{item['reranker_score']:.4f}"
            )

        # -------------------------------------------------
        # 6. Generation
        # -------------------------------------------------

        print(
            "\n[6/6] Generating grounded answer..."
        )

        generator = RAGGenerator(
    gemini_api_key="",
    gemini_retries=1,
    retry_delay=1,
)
         
        result = generator.generate(
            question=QUESTION,
            evidence=evidence,
        )

        print(
            "\n" + "=" * 100
        )

        print("ANSWER")

        print("=" * 100)

        print(
            "\n" + result["answer"]
        )

        print(
            "\n" + "=" * 100
        )

        print("VERIFIED SOURCES")

        print("=" * 100)

        for source in result["sources"]:

            print(
                f"\n[{source['citation']}] "
                f"{source['filename']}"
            )

            print(
                f"    Page     : "
                f"{source['page_number']}"
            )

            print(
                f"    Category : "
                f"{source['category']}"
            )

        print(
            f"Provider : {result.get('provider')}"
        )

        print(
            f"Model    : {result.get('model')}"
        )

        print(
            f"Grounded : {result['grounded']}"
        )

        if "error" in result:

            print(
                "\nLLM STATUS:"
            )

            print(
                result["error"]
            )

        print(
            "\n" + "=" * 100
        )

    finally:

        hybrid_retriever.close()


if __name__ == "__main__":
    main()