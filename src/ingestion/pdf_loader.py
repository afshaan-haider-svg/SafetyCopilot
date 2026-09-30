"""
SafetyCopilot — PDF Document Loader

Extracts text page-by-page from PDF files using PyMuPDF.
Returns structured page-level data with metadata.

Usage:
    from src.ingestion.pdf_loader import load_pdf

    pages = load_pdf(
        "data/raw/confined_space/HSE_Confined_Spaces_IND258.pdf"
    )

    for page in pages.pages:
        print(page.page_number, len(page.text))
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Dict, Any

import pymupdf


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class PageData:
    """Structured representation of a single extracted PDF page."""

    text: str
    page_number: int
    total_pages: int
    filename: str
    source_path: str
    category: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert page data to a dictionary."""
        return asdict(self)


@dataclass
class DocumentData:
    """Structured representation of an entire parsed PDF document."""

    filename: str
    source_path: str
    category: str
    total_pages: int
    pages: List[PageData] = field(default_factory=list)

    @property
    def pages_with_text(self) -> int:
        """Return the number of pages containing non-empty text."""
        return sum(1 for page in self.pages if page.text)

    @property
    def total_characters(self) -> int:
        """Return total extracted character count."""
        return sum(len(page.text) for page in self.pages)

    def to_dict(self) -> Dict[str, Any]:
        """Convert document data to a dictionary."""
        return {
            "filename": self.filename,
            "source_path": self.source_path,
            "category": self.category,
            "total_pages": self.total_pages,
            "pages_with_text": self.pages_with_text,
            "total_characters": self.total_characters,
            "pages": [page.to_dict() for page in self.pages],
        }


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

def _derive_category(file_path: Path) -> str:
    """
    Derive the document category from the PDF's parent folder.

    Expected structure:

        data/raw/<category>/document.pdf

    Example:

        data/raw/confined_space/file.pdf

    becomes:

        confined_space
    """

    parent = file_path.parent.name

    if parent.lower() == "raw":
        return "unknown"

    return parent


# ---------------------------------------------------------------------------
# PDF Loader
# ---------------------------------------------------------------------------

def load_pdf(file_path: str | Path) -> DocumentData:
    """
    Load a PDF document and extract text page-by-page.

    Parameters
    ----------
    file_path : str | Path
        Path to the PDF document.

    Returns
    -------
    DocumentData
        Structured document containing page text and metadata.

    Raises
    ------
    FileNotFoundError
        If the PDF does not exist.

    ValueError
        If the supplied path is invalid, is not a PDF,
        or the PDF cannot be opened.
    """

    path = Path(file_path).resolve()

    # ------------------------------------------------------------------
    # Validate input
    # ------------------------------------------------------------------

    if not path.exists():
        raise FileNotFoundError(
            f"PDF file not found: {path}"
        )

    if not path.is_file():
        raise ValueError(
            f"Path is not a file: {path}"
        )

    if path.suffix.lower() != ".pdf":
        raise ValueError(
            f"Not a PDF file (extension is '{path.suffix}'): {path}"
        )

    # ------------------------------------------------------------------
    # Open PDF
    # ------------------------------------------------------------------

    try:
        doc = pymupdf.open(str(path))

    except Exception as exc:
        raise ValueError(
            f"Failed to open PDF '{path.name}': {exc}"
        ) from exc

    try:

        total_pages = len(doc)

        filename = path.name
        source_path = str(path)
        category = _derive_category(path)

        pages: List[PageData] = []

        # ------------------------------------------------------------------
        # Extract text page-by-page
        # ------------------------------------------------------------------

        for page_index in range(total_pages):

            page = doc[page_index]

            try:
                raw_text = page.get_text("text") or ""

            except Exception:
                # Do not crash the entire document if one page
                # cannot provide extractable text.
                raw_text = ""

            cleaned_text = raw_text.strip()

            page_data = PageData(
                text=cleaned_text,

                # Human-friendly page numbering starts at 1
                page_number=page_index + 1,

                total_pages=total_pages,
                filename=filename,
                source_path=source_path,
                category=category,
            )

            pages.append(page_data)

        # ------------------------------------------------------------------
        # Build structured document
        # ------------------------------------------------------------------

        return DocumentData(
            filename=filename,
            source_path=source_path,
            category=category,
            total_pages=total_pages,
            pages=pages,
        )

    finally:

        # Always close the PDF even if extraction fails.
        doc.close()


# ---------------------------------------------------------------------------
# Batch PDF Loader
# ---------------------------------------------------------------------------

def load_pdfs_from_directory(
    directory: str | Path,
    recursive: bool = True,
) -> List[DocumentData]:
    """
    Load all PDF files from a directory.

    Parameters
    ----------
    directory : str | Path
        Directory containing PDF documents.

    recursive : bool
        If True, also search subdirectories.

    Returns
    -------
    List[DocumentData]
        Successfully loaded PDF documents.

    Notes
    -----
    A PDF that fails to load is skipped and a warning is printed.
    """

    dir_path = Path(directory).resolve()

    if not dir_path.exists():
        raise FileNotFoundError(
            f"Directory not found: {dir_path}"
        )

    if not dir_path.is_dir():
        raise NotADirectoryError(
            f"Path is not a directory: {dir_path}"
        )

    pattern = "**/*.pdf" if recursive else "*.pdf"

    pdf_files = sorted(
        dir_path.glob(pattern)
    )

    results: List[DocumentData] = []

    for pdf_path in pdf_files:

        try:

            document = load_pdf(pdf_path)

            results.append(document)

        except (FileNotFoundError, ValueError) as exc:

            print(
                f"[WARNING] Skipped '{pdf_path.name}': {exc}"
            )

    return results