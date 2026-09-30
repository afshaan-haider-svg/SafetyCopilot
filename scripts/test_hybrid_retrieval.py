"""
SafetyCopilot — Hybrid Retrieval Test

Tests:
Dense Retrieval + BM25 + Reciprocal Rank Fusion
against the real HSE knowledge base.
"""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.ingestion.pdf_loader import load_pdfs_from_directory
from src.ingestion.chunker import chunk_documents
from src.retrieval.retriever import SafetyRetriever
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.hybrid_retriever import HybridRetriever


RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
QDRANT_PATH = PROJECT_ROOT / "data" / "qdrant"


TEST_QUERIES = [
    "What are the dangers of oxygen deficiency in a confined space?",
    "What personal protective equipment should workers use?",
    "What precautions are needed around fragile surfaces when working at height?",
    "How should electrical equipment risks be controlled?",
    "What should an employer do when carrying out a workplace risk assessment?",
]


def main():

    print("=" * 100)
    print("  SAFETYCOPILOT — HYBRID RETRIEVAL TEST")
    print("=" * 100)

    # -----------------------------------------------------
    # Build BM25 corpus from same chunks used by Qdrant
    # -----------------------------------------------------

    print("\nLoading HSE documents...")

    documents = load_pdfs_from_directory(
        RAW_DATA_DIR,
        recursive=True,
    )

    chunks = chunk_documents(
        documents=documents,
        chunk_size=1000,
        chunk_overlap=200,
    )

    print(f"Documents : {len(documents)}")
    print(f"Chunks    : {len(chunks)}")

    print("\nBuilding BM25 retriever...")

    bm25_retriever = BM25Retriever(
        chunks=chunks
    )

    print("Loading dense retriever...")

    dense_retriever = SafetyRetriever(
        collection_name="safetycopilot",
        storage_path=QDRANT_PATH,
        top_k=10,
    )

    hybrid_retriever = HybridRetriever(
        dense_retriever=dense_retriever,
        bm25_retriever=bm25_retriever,
        rrf_k=60,
    )

    try:

        for query_number, query in enumerate(
            TEST_QUERIES,
            start=1,
        ):

            print("\n" + "=" * 100)
            print(f"QUERY {query_number}")
            print("=" * 100)

            print(f"\n{query}")

            results = hybrid_retriever.retrieve(
                query=query,
                top_k=3,
                candidate_k=10,
            )

            for result in results:

                print("\n" + "-" * 100)

                print(
                    f"Hybrid Rank : {result['rank']}"
                )

                print(
                    f"RRF Score   : "
                    f"{result['rrf_score']:.6f}"
                )

                print(
                    f"Dense Rank  : "
                    f"{result['dense_rank']}"
                )

                dense_score = result["dense_score"]

                if dense_score is not None:
                    print(
                        f"Dense Score : "
                        f"{dense_score:.4f}"
                    )
                else:
                    print("Dense Score : N/A")

                print(
                    f"BM25 Rank   : "
                    f"{result['bm25_rank']}"
                )

                bm25_score = result["bm25_score"]

                if bm25_score is not None:
                    print(
                        f"BM25 Score  : "
                        f"{bm25_score:.4f}"
                    )
                else:
                    print("BM25 Score  : N/A")

                print(
                    f"Category    : "
                    f"{result['category']}"
                )

                print(
                    f"Source      : "
                    f"{result['filename']}"
                )

                print(
                    f"Page        : "
                    f"{result['page_number']}/"
                    f"{result['total_pages']}"
                )

                preview = (
                    result["text"][:400]
                    .replace("\n", " ")
                )

                print(
                    f"Text        : {preview}"
                )

                if len(result["text"]) > 400:
                    print("              ...")

    finally:

        hybrid_retriever.close()

    print("\n" + "=" * 100)
    print("  HYBRID RETRIEVAL TEST COMPLETE")
    print("=" * 100)


if __name__ == "__main__":
    main()