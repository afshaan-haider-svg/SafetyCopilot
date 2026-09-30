"""
SafetyCopilot — Document Inventory & Validation Script
Scans data/raw/ for PDF files and reports readability + page counts.
Uses only Python standard library (no external dependencies).
"""

import os
import re

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")


def human_size(size_bytes: int) -> str:
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.2f} MB"


def validate_pdf(fpath: str) -> dict:
    """Check if a PDF is readable and estimate page count using stdlib only."""
    readable = False
    page_count = "N/A"
    error_msg = ""

    try:
        with open(fpath, "rb") as fh:
            header = fh.read(8)

            if not header.startswith(b"%PDF"):
                error_msg = "Missing PDF header (%PDF magic bytes)"
            else:
                readable = True

                # Extract PDF version
                fh.seek(0)
                content = fh.read()

                # Count /Type /Page (exclude /Type /Pages which is the page tree)
                pages = re.findall(rb"/Type\s*/Page(?!s)", content)
                if pages:
                    page_count = len(pages)
                else:
                    page_count = "Could not detect"

    except PermissionError:
        error_msg = "Permission denied"
    except Exception as e:
        error_msg = str(e)

    return {
        "readable": readable,
        "pages": page_count,
        "error": error_msg,
    }


def scan_raw_directory():
    results = []

    for root, dirs, files in os.walk(RAW_DIR):
        dirs.sort()
        for f in sorted(files):
            if not f.lower().endswith(".pdf"):
                continue

            fpath = os.path.join(root, f)
            category = os.path.basename(root)
            size_bytes = os.path.getsize(fpath)
            validation = validate_pdf(fpath)

            results.append({
                "file": f,
                "category": category,
                "size_str": human_size(size_bytes),
                "size_bytes": size_bytes,
                **validation,
            })

    return results


def print_report(results):
    # Header
    print()
    print("=" * 110)
    print("  SAFETYCOPILOT — DOCUMENT INVENTORY & VALIDATION REPORT")
    print("=" * 110)
    print()

    # Table header
    hdr = (
        f"  {'#':<4}"
        f"{'FILENAME':<50}"
        f"{'CATEGORY':<24}"
        f"{'SIZE':>10}"
        f"{'VALID':>8}  "
        f"{'PAGES':>16}"
    )
    print(hdr)
    print("  " + "-" * 104)

    # Rows
    corrupted = []
    for i, r in enumerate(results, 1):
        status = "YES" if r["readable"] else "NO"
        pages = str(r["pages"])
        row = (
            f"  {i:<4}"
            f"{r['file']:<50}"
            f"{r['category']:<24}"
            f"{r['size_str']:>10}"
            f"{status:>8}  "
            f"{pages:>16}"
        )
        print(row)
        if not r["readable"]:
            corrupted.append(r)

    # Summary
    print()
    print("  " + "=" * 104)
    total_bytes = sum(r["size_bytes"] for r in results)
    total_pages = sum(r["pages"] for r in results if isinstance(r["pages"], int))
    readable_count = sum(1 for r in results if r["readable"])

    print(f"  Total files found:   {len(results)}")
    print(f"  Readable:            {readable_count}")
    print(f"  Corrupted/Unreadable:{len(corrupted)}")
    print(f"  Total size:          {human_size(total_bytes)}")
    print(f"  Total pages:         {total_pages}")
    print()

    # Corrupted files detail
    if corrupted:
        print("  " + "!" * 60)
        print("  CORRUPTED / UNREADABLE FILES:")
        print("  " + "!" * 60)
        for r in corrupted:
            print(f"    - {r['file']}  ({r['category']})")
            print(f"      Error: {r['error']}")
        print()
    else:
        print("  All files passed validation. No corrupted files detected.")
        print()

    # Category breakdown
    categories = {}
    for r in results:
        cat = r["category"]
        if cat not in categories:
            categories[cat] = {"count": 0, "pages": 0, "bytes": 0}
        categories[cat]["count"] += 1
        categories[cat]["bytes"] += r["size_bytes"]
        if isinstance(r["pages"], int):
            categories[cat]["pages"] += r["pages"]

    print("  COVERAGE BY CATEGORY:")
    print("  " + "-" * 60)
    print(f"  {'CATEGORY':<28}{'FILES':>8}{'PAGES':>8}{'SIZE':>12}")
    print("  " + "-" * 60)
    for cat in sorted(categories):
        c = categories[cat]
        print(f"  {cat:<28}{c['count']:>8}{c['pages']:>8}{human_size(c['bytes']):>12}")

    # Empty categories
    all_subdirs = set()
    for entry in os.listdir(RAW_DIR):
        full = os.path.join(RAW_DIR, entry)
        if os.path.isdir(full) and entry != "__pycache__":
            all_subdirs.add(entry)
    populated = set(categories.keys())
    empty = sorted(all_subdirs - populated)

    if empty:
        print()
        print("  EMPTY CATEGORIES (no PDFs found):")
        for e in empty:
            print(f"    - {e}/")

    print()
    print("=" * 110)


if __name__ == "__main__":
    results = scan_raw_directory()
    print_report(results)
