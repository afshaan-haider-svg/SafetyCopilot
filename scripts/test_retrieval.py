"""
SafetyCopilot — Real HSE Semantic Retrieval Test
"""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.retrieval.retriever import SafetyRetriever


TEST_QUERIES = [
    "What are the dangers of working in a confined space?",
    "What personal protective equipment should workers use?",
    "What precautions should be taken when working at height?",
    "How can electrical accidents be prevented?",
    "How should workplace risks be managed?",
]


def main():

    print("=" * 100)
    print("  SAFETYCOPILOT — REAL HSE RETRIEVAL TEST")
    print("=" * 100)

    retriever = SafetyRetriever(
        collection_name="safetycopilot",
        storage_path=PROJECT_ROOT / "data" / "qdrant",
        top_k=3,
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
                    f"Score    : {result['score']:.4f}"
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

                preview = result["text"][:450].replace(
                    "\n",
                    " ",
                )

                print(
                    f"Text     : {preview}"
                )

                if len(result["text"]) > 450:
                    print("           ...")

    finally:

        retriever.close()

    print("\n" + "=" * 100)
    print("  RETRIEVAL TEST COMPLETE")
    print("=" * 100)


if __name__ == "__main__":
    main()