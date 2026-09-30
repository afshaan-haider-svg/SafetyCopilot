"""
SafetyCopilot — Chunker Test

Tests the document chunking pipeline against all HSE PDFs.

Checks:
- Total documents
- Total pages
- Total chunks
- Chunk size statistics
- Metadata preservation
- Empty chunks
- Oversized chunks
- Sample chunk quality
"""

from pathlib import Path
import sys
import statistics


# ---------------------------------------------------------
# Make project root importable
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------
# Imports
# ---------------------------------------------------------

from src.ingestion.pdf_loader import load_pdfs_from_directory
from src.ingestion.chunker import chunk_documents


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


# ---------------------------------------------------------
# Main Test
# ---------------------------------------------------------

def main():

    print("=" * 100)
    print("  SAFETYCOPILOT — DOCUMENT CHUNKER TEST")
    print("=" * 100)

    print(f"\n  Scanning: {RAW_DATA_DIR}")

    # -----------------------------------------------------
    # Load PDFs
    # -----------------------------------------------------

    documents = load_pdfs_from_directory(
        RAW_DATA_DIR,
        recursive=True,
    )

    if not documents:
        print("\n  [ERROR] No documents were loaded.")
        return

    total_pages = sum(
        document.total_pages
        for document in documents
    )

    print(f"\n  Documents loaded : {len(documents)}")
    print(f"  Total pages      : {total_pages}")

    # -----------------------------------------------------
    # Chunk Documents
    # -----------------------------------------------------

    chunks = chunk_documents(
        documents=documents,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    if not chunks:
        print("\n  [ERROR] No chunks were generated.")
        return

    # -----------------------------------------------------
    # Statistics
    # -----------------------------------------------------

    chunk_lengths = [
        len(chunk.text)
        for chunk in chunks
    ]

    minimum_size = min(chunk_lengths)
    maximum_size = max(chunk_lengths)
    average_size = statistics.mean(chunk_lengths)
    median_size = statistics.median(chunk_lengths)

    empty_chunks = [
        chunk
        for chunk in chunks
        if not chunk.text.strip()
    ]

    oversized_chunks = [
        chunk
        for chunk in chunks
        if len(chunk.text) > CHUNK_SIZE
    ]

    # -----------------------------------------------------
    # Summary
    # -----------------------------------------------------

    print("\n" + "-" * 100)

    print(f"  Chunk size target : {CHUNK_SIZE}")
    print(f"  Chunk overlap     : {CHUNK_OVERLAP}")

    print(f"\n  Total chunks      : {len(chunks)}")
    print(f"  Minimum size      : {minimum_size} characters")
    print(f"  Average size      : {average_size:.2f} characters")
    print(f"  Median size       : {median_size:.2f} characters")
    print(f"  Maximum size      : {maximum_size} characters")

    print(f"\n  Empty chunks      : {len(empty_chunks)}")
    print(f"  Oversized chunks  : {len(oversized_chunks)}")

    # -----------------------------------------------------
    # Per-document statistics
    # -----------------------------------------------------

    print("\n" + "=" * 100)
    print("  CHUNKS PER DOCUMENT")
    print("=" * 100)

    print(
        f"\n  {'DOCUMENT':55} "
        f"{'CATEGORY':22} "
        f"{'CHUNKS':>8}"
    )

    print("  " + "-" * 90)

    for document in documents:

        document_chunks = [
            chunk
            for chunk in chunks
            if chunk.filename == document.filename
        ]

        print(
            f"  {document.filename[:53]:55} "
            f"{document.category[:20]:22} "
            f"{len(document_chunks):>8}"
        )

    # -----------------------------------------------------
    # Metadata validation
    # -----------------------------------------------------

    metadata_errors = []

    for chunk in chunks:

        if not chunk.filename:
            metadata_errors.append(
                f"{chunk.chunk_id}: missing filename"
            )

        if not chunk.category:
            metadata_errors.append(
                f"{chunk.chunk_id}: missing category"
            )

        if chunk.page_number < 1:
            metadata_errors.append(
                f"{chunk.chunk_id}: invalid page number"
            )

        if chunk.page_number > chunk.total_pages:
            metadata_errors.append(
                f"{chunk.chunk_id}: page exceeds total pages"
            )

        if not chunk.chunk_id:
            metadata_errors.append(
                "Chunk missing chunk_id"
            )

    print("\n" + "=" * 100)
    print("  METADATA VALIDATION")
    print("=" * 100)

    if metadata_errors:

        print(
            f"\n  [FAILED] {len(metadata_errors)} metadata issue(s) found."
        )

        for error in metadata_errors[:10]:
            print(f"  - {error}")

    else:

        print(
            "\n  [PASS] All chunks contain valid source metadata."
        )

    # -----------------------------------------------------
    # Sample chunks
    # -----------------------------------------------------

    print("\n" + "=" * 100)
    print("  SAMPLE CHUNKS")
    print("=" * 100)

    sample_count = min(3, len(chunks))

    for index in range(sample_count):

        chunk = chunks[index]

        print(f"\n  SAMPLE {index + 1}")

        print(f"  Chunk ID    : {chunk.chunk_id}")
        print(f"  File        : {chunk.filename}")
        print(f"  Category    : {chunk.category}")
        print(
            f"  Page        : "
            f"{chunk.page_number}/{chunk.total_pages}"
        )

        print(
            f"  Characters  : {len(chunk.text)}"
        )

        print("\n  Text Preview:")
        print("  " + "-" * 80)

        preview = chunk.text[:600]

        for line in preview.splitlines():
            print(f"  {line}")

        if len(chunk.text) > 600:
            print("\n  ... [preview truncated]")

        print("  " + "-" * 80)

    # -----------------------------------------------------
    # Final Result
    # -----------------------------------------------------

    print("\n" + "=" * 100)
    print("  FINAL RESULT")
    print("=" * 100)

    if (
        len(empty_chunks) == 0
        and len(oversized_chunks) == 0
        and len(metadata_errors) == 0
    ):

        print(
            "\n  [PASS] Chunking pipeline validation successful."
        )

    else:

        print(
            "\n  [WARNING] Chunking pipeline requires review."
        )

    print("\n" + "=" * 100)


if __name__ == "__main__":
    main()