"""
SafetyCopilot — BM25 Retrieval Test
"""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.ingestion.pdf_loader import load_pdfs_from_directory
from src.ingestion.chunker import chunk_documents
from src.retrieval.bm25_retriever import BM25Retriever


RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


TEST_QUERIES = [
    "confined space oxygen",
    "personal protective equipment",
    "working at height fragile surfaces",
    "electrical equipment risk",
    "risk assessment workplace",
]


def main():

    print("=" * 100)
    print("  SAFETYCOPILOT — BM25 RETRIEVAL TEST")
    print("=" * 100)

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

    print("\nBuilding BM25 index...")

    retriever = BM25Retriever(
        chunks=chunks
    )

    for query_number, query in enumerate(
        TEST_QUERIES,
        start=1,
    ):

        print("\n" + "=" * 100)
        print(f"QUERY {query_number}: {query}")
        print("=" * 100)

        results = retriever.retrieve(
            query=query,
            top_k=3,
        )

        for result in results:

            print("\n" + "-" * 100)

            print(
                f"Rank     : {result['rank']}"
            )

            print(
                f"BM25     : {result['score']:.4f}"
            )

            print(
                f"Category : {result['category']}"
            )

            print(
                f"Source   : {result['filename']}"
            )

            print(
                f"Page     : "
                f"{result['page_number']}/"
                f"{result['total_pages']}"
            )

            preview = (
                result["text"][:350]
                .replace("\n", " ")
            )

            print(
                f"Text     : {preview}"
            )

            if len(result["text"]) > 350:
                print("           ...")

    print("\n" + "=" * 100)
    print("  BM25 TEST COMPLETE")
    print("=" * 100)


if __name__ == "__main__":
    main()