"""
SafetyCopilot — Semantic Retriever

Converts user queries into embeddings and retrieves
the most semantically relevant HSE chunks from Qdrant.

Also provides vector-index management helpers for
SafetyCopilot's dynamic HSE knowledge base.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List

from src.embeddings.embedding_model import EmbeddingModel
from src.vectorstore.qdrant_store import QdrantVectorStore
from uuid import uuid4


class SafetyRetriever:
    """
    Semantic retriever for the SafetyCopilot knowledge base.

    Responsibilities:
    - Embed user queries
    - Retrieve relevant chunks from Qdrant
    - Rebuild the complete vector index
    - Delete vectors belonging to a document
    - Report vector-store statistics
    """

    def __init__(
        self,
        collection_name: str = "safetycopilot",
        storage_path: str | Path = "data/qdrant",
        top_k: int = 5,
    ) -> None:

        self.top_k = top_k
        self.collection_name = collection_name
        self.storage_path = Path(storage_path)

        # Load embedding model once.
        self.embedding_model = EmbeddingModel()

        # Persistent local Qdrant store.
        self.store = QdrantVectorStore(
            collection_name=self.collection_name,
            vector_size=self.embedding_model.dimension,
            storage_path=self.storage_path,
        )

    # ---------------------------------------------------------
    # Semantic Retrieval
    # ---------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve the most semantically relevant HSE chunks.
        """

        if not query or not query.strip():
            raise ValueError(
                "Query cannot be empty."
            )

        limit = top_k or self.top_k

        if limit <= 0:
            raise ValueError(
                "top_k must be greater than zero."
            )

        if self.store.count() == 0:
            return []

        query_vector = self.embedding_model.embed_text(
            query.strip()
        )

        results = self.store.search(
            query_vector=query_vector.tolist(),
            limit=limit,
        )

        retrieved_chunks: List[
            Dict[str, Any]
        ] = []

        for rank, result in enumerate(
            results,
            start=1,
        ):
            payload = result.payload or {}

            retrieved_chunks.append(
                {
                    "rank": rank,
                    "score": float(
                        result.score
                    ),
                    "text": payload.get(
                        "text",
                        "",
                    ),
                    "chunk_id": payload.get(
                        "chunk_id",
                        "",
                    ),
                    "chunk_index": payload.get(
                        "chunk_index"
                    ),
                    "filename": payload.get(
                        "filename",
                        "",
                    ),
                    "source_path": payload.get(
                        "source_path",
                        "",
                    ),
                    "category": payload.get(
                        "category",
                        "",
                    ),
                    "page_number": payload.get(
                        "page_number"
                    ),
                    "total_pages": payload.get(
                        "total_pages"
                    ),
                }
            )

        return retrieved_chunks

    # ---------------------------------------------------------
    # Internal Collection Reset
    # ---------------------------------------------------------

    def _reset_collection(self) -> None:
        """
        Force a clean Qdrant collection reset.

        The collection is explicitly deleted, its removal is
        verified, then a fresh empty collection is created.

        This prevents stale vectors from surviving when the
        knowledge base becomes smaller after deleting a PDF.
        """

        print(
            "Resetting SafetyCopilot "
            "Qdrant collection..."
        )

        # -------------------------------------------------
        # 1. Delete existing collection
        # -------------------------------------------------

        if self.store.collection_exists():
            self.store.delete_collection()

        # -------------------------------------------------
        # 2. Verify deletion
        # -------------------------------------------------

        max_attempts = 20

        for attempt in range(
            1,
            max_attempts + 1,
        ):
            if not self.store.collection_exists():
                break

            if attempt == max_attempts:
                raise RuntimeError(
                    "Qdrant collection could not be "
                    "deleted during index rebuild."
                )

            time.sleep(0.05)

        # -------------------------------------------------
        # 3. Create fresh collection
        # -------------------------------------------------

        self.store.create_collection(
            recreate=False
        )

        if not self.store.collection_exists():
            raise RuntimeError(
                "Qdrant collection could not be "
                "created during index rebuild."
            )

        # -------------------------------------------------
        # 4. Verify fresh collection is empty
        # -------------------------------------------------

        fresh_count = self.store.count()

        if fresh_count != 0:
            raise RuntimeError(
                "Fresh Qdrant collection is not empty. "
                f"Expected 0 vectors, found "
                f"{fresh_count}."
            )

        print(
            "Qdrant collection reset "
            "successfully: 0 vectors."
        )

    # ---------------------------------------------------------
    # Complete Knowledge Base Rebuild
    # ---------------------------------------------------------

    def rebuild_index(
        self,
        chunks: List[Any],
        batch_size: int = 32,
    ) -> int:
        """
        Rebuild the complete Qdrant vector index from chunks.

        A strict clean rebuild is used:

        1. Generate embeddings
        2. Delete old Qdrant collection
        3. Verify old collection is gone
        4. Create fresh empty collection
        5. Verify fresh collection contains zero vectors
        6. Insert only the current chunks
        7. Verify final vector count

        Returns:
            Number of vectors stored in Qdrant.
        """

        if batch_size <= 0:
            raise ValueError(
                "batch_size must be greater than zero."
            )

        print(
            "\nRebuilding SafetyCopilot "
            "vector index..."
        )

        # -------------------------------------------------
        # Empty knowledge base
        # -------------------------------------------------

        if not chunks:
            self._reset_collection()

            final_count = self.store.count()

            if final_count != 0:
                raise RuntimeError(
                    "Empty knowledge base rebuild "
                    "failed. "
                    f"Expected 0 vectors, stored "
                    f"{final_count}."
                )

            print(
                "Vector index rebuilt "
                "successfully: 0 chunks."
            )

            return 0

        # -------------------------------------------------
        # Prepare chunk texts
        # -------------------------------------------------

        texts = [
            chunk.text
            for chunk in chunks
        ]

        # -------------------------------------------------
        # Generate embeddings
        # -------------------------------------------------

        embeddings = (
            self.embedding_model.embed_texts(
                texts=texts,
                batch_size=batch_size,
            )
        )

        if len(embeddings) != len(chunks):
            raise RuntimeError(
                "Embedding count does not match "
                "chunk count."
            )

        # -------------------------------------------------
        # Prepare IDs
        # -------------------------------------------------

        ids = list(
            range(
                1,
                len(chunks) + 1,
            )
        )

        # -------------------------------------------------
        # Prepare payloads
        # -------------------------------------------------

        payloads: List[
            Dict[str, Any]
        ] = []

        for chunk in chunks:
            payloads.append(
                {
                    "text": chunk.text,
                    "chunk_id": (
                        chunk.chunk_id
                    ),
                    "chunk_index": (
                        chunk.chunk_index
                    ),
                    "filename": (
                        chunk.filename
                    ),
                    "source_path": (
                        chunk.source_path
                    ),
                    "category": (
                        chunk.category
                    ),
                    "page_number": (
                        chunk.page_number
                    ),
                    "total_pages": (
                        chunk.total_pages
                    ),
                }
            )

        # -------------------------------------------------
        # STRICT collection reset
        # -------------------------------------------------

        self._reset_collection()

        before_insert_count = (
            self.store.count()
        )

        if before_insert_count != 0:
            raise RuntimeError(
                "Qdrant collection contains vectors "
                "before rebuild insertion. "
                f"Expected 0, stored "
                f"{before_insert_count}."
            )

        # -------------------------------------------------
        # Insert current vectors
        # -------------------------------------------------

        self.store.add_points(
            ids=ids,
            vectors=embeddings.tolist(),
            payloads=payloads,
        )

        # -------------------------------------------------
        # Final validation
        # -------------------------------------------------

        stored_count = self.store.count()

        if stored_count != len(chunks):
            raise RuntimeError(
                "Qdrant vector count does not "
                "match generated chunk count. "
                f"Expected {len(chunks)}, "
                f"stored {stored_count}."
            )

        print(
            "Vector index rebuilt successfully: "
            f"{stored_count} chunks."
        )

        return stored_count


    # ---------------------------------------------------------
    # Incremental Document Vector Addition
    # ---------------------------------------------------------

    def add_document_chunks(
        self,
        chunks: List[Any],
        batch_size: int = 32,
    ) -> int:
        """
        Add chunks for one new document without rebuilding
        the existing Qdrant collection.

        Returns the total number of vectors stored after
        the new document has been indexed.
        """

        if batch_size <= 0:
            raise ValueError(
                "batch_size must be greater than zero."
            )

        if not chunks:
            raise ValueError(
                "No document chunks were supplied."
            )

        filenames = {
            Path(
                getattr(
                    chunk,
                    "filename",
                    "",
                )
            ).name
            for chunk in chunks
            if getattr(
                chunk,
                "filename",
                "",
            )
        }

        if len(filenames) != 1:
            raise ValueError(
                "Incremental indexing expects chunks "
                "from exactly one document."
            )

        filename = next(iter(filenames))

        # Do not accidentally duplicate an already
        # indexed document.
        existing_document_vectors = (
            self.store.count_by_filename(
                filename
            )
        )

        if existing_document_vectors > 0:
            raise ValueError(
                f"Vectors already exist for {filename}."
            )

        vectors_before = self.store.count()

        texts = [
            chunk.text
            for chunk in chunks
        ]

        embeddings = (
            self.embedding_model.embed_texts(
                texts=texts,
                batch_size=batch_size,
            )
        )

        if len(embeddings) != len(chunks):
            raise RuntimeError(
                "Embedding count does not match "
                "chunk count."
            )

        # Existing rebuild IDs start from 1 and are
        # sequential. New IDs therefore continue after
        # the current vector count.
        ids = [
    str(uuid4())
    for _ in chunks
]

        payloads: List[
            Dict[str, Any]
        ] = []

        for chunk in chunks:
            payloads.append(
                {
                    "text": chunk.text,
                    "chunk_id": (
                        chunk.chunk_id
                    ),
                    "chunk_index": (
                        chunk.chunk_index
                    ),
                    "filename": (
                        chunk.filename
                    ),
                    "source_path": (
                        chunk.source_path
                    ),
                    "category": (
                        chunk.category
                    ),
                    "page_number": (
                        chunk.page_number
                    ),
                    "total_pages": (
                        chunk.total_pages
                    ),
                }
            )

        self.store.add_points(
            ids=ids,
            vectors=embeddings.tolist(),
            payloads=payloads,
        )

        vectors_after = self.store.count()

        expected_vectors = (
            vectors_before
            + len(chunks)
        )

        if vectors_after != expected_vectors:
            # Best-effort rollback: remove only vectors
            # belonging to the newly added document.
            try:
                self.store.delete_by_filename(
                    filename
                )
            except Exception:
                pass

            raise RuntimeError(
                "Incremental vector indexing failed. "
                f"Expected {expected_vectors} vectors, "
                f"stored {vectors_after}."
            )

        document_vectors = (
            self.store.count_by_filename(
                filename
            )
        )

        if document_vectors != len(chunks):
            try:
                self.store.delete_by_filename(
                    filename
                )
            except Exception:
                pass

            raise RuntimeError(
                "New document vector count does not "
                "match its generated chunk count. "
                f"Expected {len(chunks)}, "
                f"stored {document_vectors}."
            )

        print(
            f"Document indexed incrementally: "
            f"{filename}"
        )

        print(
            f"New vectors : {len(chunks)}"
        )

        print(
            f"Total vectors: {vectors_after}"
        )

        return vectors_after
    # ---------------------------------------------------------
    # Individual Document Vector Deletion
    # ---------------------------------------------------------

    def delete_document(
        self,
        filename: str,
    ) -> int:
        """
        Delete all vectors belonging to a PDF.

        Returns the number of vectors removed.
        """

        safe_filename = Path(
            filename
        ).name

        if not safe_filename:
            raise ValueError(
                "Filename cannot be empty."
            )

        before = (
            self.store.count_by_filename(
                safe_filename
            )
        )

        if before == 0:
            return 0

        self.store.delete_by_filename(
            safe_filename
        )

        after = (
            self.store.count_by_filename(
                safe_filename
            )
        )

        if after != 0:
            raise RuntimeError(
                "Qdrant document vector deletion "
                "was incomplete. "
                f"{after} vectors remain for "
                f"{safe_filename}."
            )

        removed = max(
            before - after,
            0,
        )

        return removed

    # ---------------------------------------------------------
    # Statistics
    # ---------------------------------------------------------

    def vector_count(self) -> int:
        """
        Return total vectors currently stored.
        """

        return self.store.count()

    def document_vector_count(
        self,
        filename: str,
    ) -> int:
        """
        Return vector count for one document.
        """

        safe_filename = Path(
            filename
        ).name

        if not safe_filename:
            return 0

        return (
            self.store.count_by_filename(
                safe_filename
            )
        )

    # ---------------------------------------------------------
    # Cleanup
    # ---------------------------------------------------------

    def close(self) -> None:
        """
        Close the Qdrant connection.
        """

        self.store.close()