"""
SafetyCopilot — PDF Loader Test Script

Scans data/raw/ for all PDFs, loads each with the pdf_loader module,
and prints a concise summary table plus a text preview from the first document.

Usage:
    python scripts/test_pdf_loader.py
"""

import os
import sys
import textwrap

# Ensure the project root is on sys.path so we can import src.*
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.ingestion.pdf_loader import load_pdfs_from_directory

RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
PREVIEW_MAX_CHARS = 1000  # Characters to show in the text preview


def main():
    print()
    print("=" * 100)
    print("  SAFETYCOPILOT — PDF LOADER TEST")
    print("=" * 100)
    print()
    print(f"  Scanning: {RAW_DIR}")
    print()

    # Load all PDFs
    documents = load_pdfs_from_directory(RAW_DIR, recursive=True)

    if not documents:
        print("  No PDF files found in data/raw/.")
        print()
        return

    # --- Summary Table ---
    header = (
        f"  {'#':<4}"
        f"{'FILENAME':<52}"
        f"{'CATEGORY':<22}"
        f"{'PAGES':>6}"
        f"{'WITH TEXT':>10}"
        f"{'CHARACTERS':>12}"
    )
    print(header)
    print("  " + "-" * 96)

    total_pages = 0
    total_text_pages = 0
    total_chars = 0

    for i, doc in enumerate(documents, 1):
        row = (
            f"  {i:<4}"
            f"{doc.filename:<52}"
            f"{doc.category:<22}"
            f"{doc.total_pages:>6}"
            f"{doc.pages_with_text:>10}"
            f"{doc.total_characters:>12,}"
        )
        print(row)
        total_pages += doc.total_pages
        total_text_pages += doc.pages_with_text
        total_chars += doc.total_characters

    print("  " + "-" * 96)
    print(
        f"  {'':4}"
        f"{'TOTALS':<52}"
        f"{'':22}"
        f"{total_pages:>6}"
        f"{total_text_pages:>10}"
        f"{total_chars:>12,}"
    )
    print()

    # --- Extraction Quality ---
    empty_pages = []
    for doc in documents:
        for page in doc.pages:
            if not page.text:
                empty_pages.append((doc.filename, page.page_number))

    if empty_pages:
        print("  EMPTY PAGES DETECTED:")
        for fname, pnum in empty_pages:
            print(f"    - {fname}, page {pnum}")
        print()
    else:
        print("  All pages contain extracted text.")
        print()

    # --- Text Preview (first document only) ---
    first_doc = documents[0]
    print("=" * 100)
    print(f"  TEXT PREVIEW — {first_doc.filename} (page 1)")
    print("=" * 100)
    print()

    if first_doc.pages and first_doc.pages[0].text:
        preview = first_doc.pages[0].text[:PREVIEW_MAX_CHARS]
        # Indent each line for readability
        for line in preview.splitlines():
            print(f"    {line}")

        if len(first_doc.pages[0].text) > PREVIEW_MAX_CHARS:
            remaining = len(first_doc.pages[0].text) - PREVIEW_MAX_CHARS
            print()
            print(f"    ... [{remaining:,} more characters on this page]")
    else:
        print("    (no text on page 1)")

    print()
    print("=" * 100)
    print("  TEST COMPLETE")
    print("=" * 100)
    print()


if __name__ == "__main__":
    main()
