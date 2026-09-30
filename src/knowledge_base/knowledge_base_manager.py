"""
SafetyCopilot — Dynamic Knowledge Base Manager

Keeps the complete RAG knowledge base synchronized when
HSE PDF documents are uploaded or deleted.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Dict, List

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


class KnowledgeBaseManager:
    """
    Manage SafetyCopilot's dynamic HSE knowledge base.
    """

    def __init__(
        self,
        raw_data_dir: str | Path,
        qdrant_path: str | Path,
        collection_name: str = "safetycopilot",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        dense_top_k: int = 15,
        rrf_k: int = 60,
        embedding_batch_size: int = 32,
    ) -> None:

        self.raw_data_dir = Path(
            raw_data_dir
        )

        self.qdrant_path = Path(
            qdrant_path
        )

        self.collection_name = (
            collection_name
        )

        self.chunk_size = chunk_size

        self.chunk_overlap = (
            chunk_overlap
        )

        self.dense_top_k = (
            dense_top_k
        )

        self.rrf_k = rrf_k

        self.embedding_batch_size = (
            embedding_batch_size
        )

        self.raw_data_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.qdrant_path.mkdir(
            parents=True,
            exist_ok=True,
        )

        self._lock = threading.RLock()

        self.documents: List[Any] = []
        self.chunks: List[Any] = []

        self.dense_retriever: (
            SafetyRetriever | None
        ) = None

        self.bm25_retriever: (
            BM25Retriever | None
        ) = None

        self.hybrid_retriever: (
            HybridRetriever | None
        ) = None

        # Create dense retriever once.
        self.dense_retriever = (
            SafetyRetriever(
                collection_name=(
                    self.collection_name
                ),
                storage_path=(
                    self.qdrant_path
                ),
                top_k=(
                    self.dense_top_k
                ),
            )
        )

    # =========================================================
    # LOAD DOCUMENTS
    # =========================================================

    def load_documents(
        self,
    ) -> List[Any]:
        """
        Load all PDFs recursively.
        """

        return load_pdfs_from_directory(
            self.raw_data_dir,
            recursive=True,
        )

    # =========================================================
    # CREATE CHUNKS
    # =========================================================

    def create_chunks(
        self,
        documents: List[Any],
    ) -> List[Any]:
        """
        Convert loaded documents into retrieval chunks.
        """

        if not documents:
            return []

        return chunk_documents(
            documents=documents,
            chunk_size=self.chunk_size,
            chunk_overlap=(
                self.chunk_overlap
            ),
        )

    # =========================================================
    # FULL REBUILD
    # =========================================================

    def rebuild(
        self,
    ) -> Dict[str, int]:
        """
        Completely rebuild dense, BM25 and hybrid retrieval.
        """

        with self._lock:

            print(
                "\n"
                + "=" * 70
            )

            print(
                "Rebuilding SafetyCopilot "
                "knowledge base..."
            )

            print(
                "=" * 70
            )

            new_documents = (
                self.load_documents()
            )

            new_chunks = (
                self.create_chunks(
                    new_documents
                )
            )

            print(
                f"Documents loaded: "
                f"{len(new_documents)}"
            )

            print(
                f"Chunks generated: "
                f"{len(new_chunks)}"
            )

            if self.dense_retriever is None:
                raise RuntimeError(
                    "Dense retriever is "
                    "not initialized."
                )

            vector_count = (
                self.dense_retriever
                .rebuild_index(
                    chunks=new_chunks,
                    batch_size=(
                        self.embedding_batch_size
                    ),
                )
            )

            new_bm25_retriever = (
                BM25Retriever(
                    chunks=new_chunks,
                )
            )

            new_hybrid_retriever = (
                HybridRetriever(
                    dense_retriever=(
                        self.dense_retriever
                    ),
                    bm25_retriever=(
                        new_bm25_retriever
                    ),
                    rrf_k=self.rrf_k,
                )
            )

            old_hybrid = (
                self.hybrid_retriever
            )

            self.documents = (
                new_documents
            )

            self.chunks = (
                new_chunks
            )

            self.bm25_retriever = (
                new_bm25_retriever
            )

            self.hybrid_retriever = (
                new_hybrid_retriever
            )

            if (
                old_hybrid is not None
                and old_hybrid
                is not new_hybrid_retriever
            ):
                self._close_old_hybrid(
                    old_hybrid
                )

            stats = {
                "documents": len(
                    self.documents
                ),
                "chunks": len(
                    self.chunks
                ),
                "vectors": vector_count,
            }

            print(
                "Knowledge base rebuilt "
                "successfully."
            )

            print(
                f"Documents : "
                f"{stats['documents']}"
            )

            print(
                f"Chunks    : "
                f"{stats['chunks']}"
            )

            print(
                f"Vectors   : "
                f"{stats['vectors']}"
            )

            print(
                "=" * 70
                + "\n"
            )

            return stats

    # =========================================================
    # INITIALIZE
    # =========================================================

    def initialize(
        self,
        rebuild_vectors: bool = False,
    ) -> Dict[str, int]:
        """
        Initialize runtime retrieval state.

        Existing Qdrant vectors are reused unless a rebuild
        is explicitly requested or Qdrant is empty.
        """

        with self._lock:

            if rebuild_vectors:
                return self.rebuild()

            new_documents = (
                self.load_documents()
            )

            new_chunks = (
                self.create_chunks(
                    new_documents
                )
            )

            if self.dense_retriever is None:
                raise RuntimeError(
                    "Dense retriever is "
                    "not initialized."
                )

            vector_count = (
                self.dense_retriever
                .vector_count()
            )

            # PDFs exist but Qdrant is empty.
            if (
                new_chunks
                and vector_count == 0
            ):
                vector_count = (
                    self.dense_retriever
                    .rebuild_index(
                        chunks=new_chunks,
                        batch_size=(
                            self.embedding_batch_size
                        ),
                    )
                )

            # No PDFs but stale vectors exist.
            elif (
                not new_chunks
                and vector_count > 0
            ):
                vector_count = (
                    self.dense_retriever
                    .rebuild_index(
                        chunks=[],
                        batch_size=(
                            self.embedding_batch_size
                        ),
                    )
                )

            new_bm25_retriever = (
                BM25Retriever(
                    chunks=new_chunks,
                )
            )

            new_hybrid_retriever = (
                HybridRetriever(
                    dense_retriever=(
                        self.dense_retriever
                    ),
                    bm25_retriever=(
                        new_bm25_retriever
                    ),
                    rrf_k=self.rrf_k,
                )
            )

            old_hybrid = (
                self.hybrid_retriever
            )

            self.documents = (
                new_documents
            )

            self.chunks = (
                new_chunks
            )

            self.bm25_retriever = (
                new_bm25_retriever
            )

            self.hybrid_retriever = (
                new_hybrid_retriever
            )

            if (
                old_hybrid is not None
                and old_hybrid
                is not new_hybrid_retriever
            ):
                self._close_old_hybrid(
                    old_hybrid
                )

            return {
                "documents": len(
                    self.documents
                ),
                "chunks": len(
                    self.chunks
                ),
                "vectors": vector_count,
            }

    # =========================================================
    # RETRIEVE
    # =========================================================

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        candidate_k: int = 15,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve evidence using hybrid retrieval.
        """

        if not query or not query.strip():
            raise ValueError(
                "Query cannot be empty."
            )

        with self._lock:

            if self.hybrid_retriever is None:
                raise RuntimeError(
                    "Knowledge base has "
                    "not been initialized."
                )

            if not self.chunks:
                return []

            return (
                self.hybrid_retriever
                .retrieve(
                    query=query,
                    top_k=top_k,
                    candidate_k=(
                        candidate_k
                    ),
                )
            )

    # =========================================================
    # INCREMENTAL DOCUMENT ADDITION
    # =========================================================

    def add_document(
        self,
        filename: str,
    ) -> Dict[str, Any]:
        """
        Add one uploaded PDF incrementally.

        Only vectors belonging to the new document are added.
        Existing Qdrant vectors are preserved.
        """

        safe_filename = Path(
            filename
        ).name

        if not safe_filename:
            raise ValueError(
                "Filename cannot be empty."
            )

        with self._lock:

            if self.dense_retriever is None:
                raise RuntimeError(
                    "Dense retriever is "
                    "not initialized."
                )

            # -------------------------------------------------
            # Locate uploaded PDF
            # -------------------------------------------------

            matches = [
                path
                for path
                in self.raw_data_dir.rglob(
                    "*.pdf"
                )
                if path.name
                == safe_filename
            ]

            if not matches:
                raise FileNotFoundError(
                    "Document not found: "
                    f"{safe_filename}"
                )

            if len(matches) > 1:
                raise RuntimeError(
                    "Multiple PDFs with the "
                    "same filename were found: "
                    f"{safe_filename}"
                )

            # -------------------------------------------------
            # Ensure document is not already indexed
            # -------------------------------------------------

            existing_vectors = (
                self.dense_retriever
                .document_vector_count(
                    safe_filename
                )
            )

            if existing_vectors > 0:
                raise RuntimeError(
                    "The document is already "
                    "indexed in Qdrant: "
                    f"{safe_filename}"
                )

            vectors_before = (
                self.dense_retriever
                .vector_count()
            )

            # -------------------------------------------------
            # Reload current PDFs
            # -------------------------------------------------

            updated_documents = (
                self.load_documents()
            )

            updated_chunks = (
                self.create_chunks(
                    updated_documents
                )
            )

            # -------------------------------------------------
            # Select only chunks from new PDF
            # -------------------------------------------------

            new_chunks = [
                chunk
                for chunk
                in updated_chunks
                if Path(
                    getattr(
                        chunk,
                        "filename",
                        "",
                    )
                ).name
                == safe_filename
            ]

            if not new_chunks:
                raise RuntimeError(
                    "The uploaded PDF did "
                    "not produce any chunks."
                )

            existing_chunks = [
                chunk
                for chunk
                in updated_chunks
                if Path(
                    getattr(
                        chunk,
                        "filename",
                        "",
                    )
                ).name
                != safe_filename
            ]

            # -------------------------------------------------
            # Validate existing state
            # -------------------------------------------------

            if (
                len(existing_chunks)
                != vectors_before
            ):
                raise RuntimeError(
                    "Pre-upload consistency "
                    "check failed. "
                    f"Existing chunks: "
                    f"{len(existing_chunks)}, "
                    f"existing vectors: "
                    f"{vectors_before}."
                )

            # -------------------------------------------------
            # Add ONLY new vectors
            # -------------------------------------------------

            vectors_after = (
                self.dense_retriever
                .add_document_chunks(
                    chunks=new_chunks,
                    batch_size=(
                        self.embedding_batch_size
                    ),
                )
            )

            expected_vectors = (
                vectors_before
                + len(new_chunks)
            )

            if (
                vectors_after
                != expected_vectors
            ):
                raise RuntimeError(
                    "Post-upload vector count "
                    "validation failed. "
                    f"Expected "
                    f"{expected_vectors}, "
                    f"stored "
                    f"{vectors_after}."
                )

            # -------------------------------------------------
            # Rebuild BM25 from updated chunks
            # -------------------------------------------------

            new_bm25_retriever = (
                BM25Retriever(
                    chunks=updated_chunks,
                )
            )

            # -------------------------------------------------
            # Refresh Hybrid Retriever
            # -------------------------------------------------

            new_hybrid_retriever = (
                HybridRetriever(
                    dense_retriever=(
                        self.dense_retriever
                    ),
                    bm25_retriever=(
                        new_bm25_retriever
                    ),
                    rrf_k=self.rrf_k,
                )
            )

            old_hybrid = (
                self.hybrid_retriever
            )

            self.documents = (
                updated_documents
            )

            self.chunks = (
                updated_chunks
            )

            self.bm25_retriever = (
                new_bm25_retriever
            )

            self.hybrid_retriever = (
                new_hybrid_retriever
            )

            if (
                old_hybrid is not None
                and old_hybrid
                is not new_hybrid_retriever
            ):
                self._close_old_hybrid(
                    old_hybrid
                )

            # -------------------------------------------------
            # Final validation
            # -------------------------------------------------

            final_vectors = (
                self.dense_retriever
                .vector_count()
            )

            if (
                final_vectors
                != len(self.chunks)
            ):
                raise RuntimeError(
                    "Knowledge base is out "
                    "of sync after document "
                    "upload. "
                    f"Chunks: "
                    f"{len(self.chunks)}, "
                    f"vectors: "
                    f"{final_vectors}."
                )

            document_vectors = (
                self.dense_retriever
                .document_vector_count(
                    safe_filename
                )
            )

            if (
                document_vectors
                != len(new_chunks)
            ):
                raise RuntimeError(
                    "Uploaded document vector "
                    "validation failed. "
                    f"Expected "
                    f"{len(new_chunks)}, "
                    f"stored "
                    f"{document_vectors}."
                )

            result: Dict[str, Any] = {
                "filename": safe_filename,
                "added_vectors": (
                    document_vectors
                ),
                "documents": len(
                    self.documents
                ),
                "chunks": len(
                    self.chunks
                ),
                "vectors": (
                    final_vectors
                ),
            }

            print(
                "\nDocument indexed "
                "successfully."
            )

            print(
                f"Filename  : "
                f"{safe_filename}"
            )

            print(
                f"New vectors: "
                f"{document_vectors}"
            )

            print(
                f"Documents : "
                f"{result['documents']}"
            )

            print(
                f"Chunks    : "
                f"{result['chunks']}"
            )

            print(
                f"Vectors   : "
                f"{result['vectors']}"
            )

            return result

    # =========================================================
    # DOCUMENT-SPECIFIC DELETION
    # =========================================================

    def delete_document(
        self,
        filename: str,
    ) -> Dict[str, Any]:
        """
        Delete one PDF and only its vectors.

        The complete Qdrant collection is not rebuilt.
        """

        safe_filename = Path(
            filename
        ).name

        if not safe_filename:
            raise ValueError(
                "Filename cannot be empty."
            )

        with self._lock:

            if self.dense_retriever is None:
                raise RuntimeError(
                    "Dense retriever is "
                    "not initialized."
                )

            # -------------------------------------------------
            # Locate PDF
            # -------------------------------------------------

            matches = [
                path
                for path
                in self.raw_data_dir.rglob(
                    "*.pdf"
                )
                if path.name
                == safe_filename
            ]

            if not matches:
                raise FileNotFoundError(
                    "Document not found: "
                    f"{safe_filename}"
                )

            if len(matches) > 1:
                raise RuntimeError(
                    "Multiple PDFs with the "
                    "same filename were found: "
                    f"{safe_filename}"
                )

            target_path = matches[0]

            # -------------------------------------------------
            # Current vector state
            # -------------------------------------------------

            vectors_before = (
                self.dense_retriever
                .vector_count()
            )

            document_vectors = (
                self.dense_retriever
                .document_vector_count(
                    safe_filename
                )
            )

            if document_vectors <= 0:
                raise RuntimeError(
                    "No Qdrant vectors were "
                    "found for "
                    f"{safe_filename}. "
                    "The knowledge base may "
                    "already be out of sync."
                )

            # -------------------------------------------------
            # Pre-delete consistency validation
            # -------------------------------------------------

            current_documents = (
                self.load_documents()
            )

            current_chunks = (
                self.create_chunks(
                    current_documents
                )
            )

            remaining_chunks = [
                chunk
                for chunk
                in current_chunks
                if Path(
                    getattr(
                        chunk,
                        "filename",
                        "",
                    )
                ).name
                != safe_filename
            ]

            expected_vectors = (
                vectors_before
                - document_vectors
            )

            if (
                expected_vectors
                != len(remaining_chunks)
            ):
                raise RuntimeError(
                    "Pre-delete consistency "
                    "check failed. "
                    f"Expected "
                    f"{len(remaining_chunks)} "
                    "remaining chunks but "
                    "Qdrant would contain "
                    f"{expected_vectors} "
                    "vectors."
                )

            print(
                f"\nDeleting document: "
                f"{safe_filename}"
            )

            print(
                f"Vectors to remove: "
                f"{document_vectors}"
            )

            # -------------------------------------------------
            # Delete ONLY this document's vectors
            # -------------------------------------------------

            removed_vectors = (
                self.dense_retriever
                .delete_document(
                    safe_filename
                )
            )

            # -------------------------------------------------
            # Delete physical PDF
            # -------------------------------------------------

            try:
                target_path.unlink()

            except Exception as exc:

                # Best effort recovery:
                # physical PDF still exists, so restore
                # its vectors if possible.
                try:
                    restored_documents = (
                        self.load_documents()
                    )

                    restored_chunks = (
                        self.create_chunks(
                            restored_documents
                        )
                    )

                    restore_chunks = [
                        chunk
                        for chunk
                        in restored_chunks
                        if Path(
                            getattr(
                                chunk,
                                "filename",
                                "",
                            )
                        ).name
                        == safe_filename
                    ]

                    if restore_chunks:
                        (
                            self.dense_retriever
                            .add_document_chunks(
                                chunks=(
                                    restore_chunks
                                ),
                                batch_size=(
                                    self.embedding_batch_size
                                ),
                            )
                        )

                except Exception:
                    pass

                raise RuntimeError(
                    "PDF deletion failed after "
                    "its vectors were removed."
                ) from exc

            # -------------------------------------------------
            # Remove empty category directories
            # -------------------------------------------------

            parent = target_path.parent

            while (
                parent
                != self.raw_data_dir
                and parent.exists()
            ):
                try:
                    if any(
                        parent.iterdir()
                    ):
                        break

                    parent.rmdir()
                    parent = (
                        parent.parent
                    )

                except Exception:
                    break

            # -------------------------------------------------
            # Reload remaining documents/chunks
            # -------------------------------------------------

            new_documents = (
                self.load_documents()
            )

            new_chunks = (
                self.create_chunks(
                    new_documents
                )
            )

            # -------------------------------------------------
            # Validate Qdrant
            # -------------------------------------------------

            stored_vectors = (
                self.dense_retriever
                .vector_count()
            )

            remaining_deleted_vectors = (
                self.dense_retriever
                .document_vector_count(
                    safe_filename
                )
            )

            if (
                remaining_deleted_vectors
                != 0
            ):
                raise RuntimeError(
                    "Document vector deletion "
                    "was incomplete. "
                    f"{remaining_deleted_vectors} "
                    "vectors remain for "
                    f"{safe_filename}."
                )

            if (
                stored_vectors
                != len(new_chunks)
            ):
                raise RuntimeError(
                    "Qdrant vector count does "
                    "not match remaining "
                    "chunk count. "
                    f"Expected "
                    f"{len(new_chunks)}, "
                    f"stored "
                    f"{stored_vectors}."
                )

            # -------------------------------------------------
            # Rebuild BM25 only
            # -------------------------------------------------

            new_bm25_retriever = (
                BM25Retriever(
                    chunks=new_chunks,
                )
            )

            # -------------------------------------------------
            # Refresh hybrid retriever
            # -------------------------------------------------

            new_hybrid_retriever = (
                HybridRetriever(
                    dense_retriever=(
                        self.dense_retriever
                    ),
                    bm25_retriever=(
                        new_bm25_retriever
                    ),
                    rrf_k=self.rrf_k,
                )
            )

            old_hybrid = (
                self.hybrid_retriever
            )

            self.documents = (
                new_documents
            )

            self.chunks = (
                new_chunks
            )

            self.bm25_retriever = (
                new_bm25_retriever
            )

            self.hybrid_retriever = (
                new_hybrid_retriever
            )

            if (
                old_hybrid is not None
                and old_hybrid
                is not new_hybrid_retriever
            ):
                self._close_old_hybrid(
                    old_hybrid
                )

            result: Dict[str, Any] = {
                "filename": (
                    safe_filename
                ),
                "removed_vectors": (
                    removed_vectors
                ),
                "documents": len(
                    self.documents
                ),
                "chunks": len(
                    self.chunks
                ),
                "vectors": (
                    stored_vectors
                ),
            }

            print(
                "Document deleted "
                "successfully."
            )

            print(
                f"Documents : "
                f"{result['documents']}"
            )

            print(
                f"Chunks    : "
                f"{result['chunks']}"
            )

            print(
                f"Vectors   : "
                f"{result['vectors']}"
            )

            return result

    # =========================================================
    # STATISTICS
    # =========================================================

    @property
    def document_count(
        self,
    ) -> int:

        with self._lock:
            return len(
                self.documents
            )

    @property
    def chunk_count(
        self,
    ) -> int:

        with self._lock:
            return len(
                self.chunks
            )

    @property
    def vector_count(
        self,
    ) -> int:

        with self._lock:

            if self.dense_retriever is None:
                return 0

            return (
                self.dense_retriever
                .vector_count()
            )

    def stats(
        self,
    ) -> Dict[str, int]:

        return {
            "documents": (
                self.document_count
            ),
            "chunks": (
                self.chunk_count
            ),
            "vectors": (
                self.vector_count
            ),
        }

    # =========================================================
    # INTERNAL CLEANUP
    # =========================================================

    def _close_old_hybrid(
        self,
        hybrid: HybridRetriever,
    ) -> None:
        """
        Release old hybrid object without closing the
        shared dense retriever.
        """

        del hybrid

    # =========================================================
    # CLOSE
    # =========================================================

    def close(
        self,
    ) -> None:

        with self._lock:

            self.hybrid_retriever = None
            self.bm25_retriever = None

            if (
                self.dense_retriever
                is not None
            ):
                self.dense_retriever.close()

            self.dense_retriever = None