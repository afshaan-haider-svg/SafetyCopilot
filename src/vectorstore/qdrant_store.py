"""
SafetyCopilot — Qdrant Vector Store

Provides a lightweight wrapper around Qdrant for storing,
retrieving, deleting, and rebuilding SafetyCopilot embeddings.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)


class QdrantVectorStore:
    """
    Persistent local Qdrant vector store used by SafetyCopilot.

    Supports:
    - collection creation
    - safe collection recreation
    - vector upsert
    - semantic search
    - document-specific deletion
    - collection counting
    - client refresh
    - safe cleanup
    """

    def __init__(
        self,
        collection_name: str = "safetycopilot",
        vector_size: int = 384,
        storage_path: str | Path = "data/qdrant",
    ) -> None:

        self.collection_name = collection_name
        self.vector_size = vector_size

        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.client = self._create_client()

    # ---------------------------------------------------------
    # Client Management
    # ---------------------------------------------------------

    def _create_client(self) -> QdrantClient:
        """
        Create a fresh local Qdrant client.
        """

        return QdrantClient(
            path=str(self.storage_path)
        )

    def refresh_client(self) -> None:
        """
        Close the current local Qdrant client and open
        a fresh client against the same storage directory.

        This is important after destructive collection
        operations so stale local state is not reused.
        """

        try:
            self.client.close()
        except Exception:
            pass

        self.client = self._create_client()

    # ---------------------------------------------------------
    # Collection Management
    # ---------------------------------------------------------

    def collection_exists(self) -> bool:
        """
        Return True when the configured collection exists.
        """

        return self.client.collection_exists(
            self.collection_name
        )

    def create_collection(
        self,
        recreate: bool = False,
    ) -> None:
        """
        Create the configured collection.

        If recreate=True, perform a strict clean recreation.
        """

        if recreate:
            self.recreate_collection()
            return

        if self.collection_exists():
            return

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=self.vector_size,
                distance=Distance.COSINE,
            ),
        )

    def delete_collection(self) -> None:
        """
        Delete the configured collection if it exists.

        The client is refreshed afterwards so subsequent
        operations use the latest local-storage state.
        """

        if self.collection_exists():
            self.client.delete_collection(
                collection_name=self.collection_name
            )

        self.refresh_client()

    def recreate_collection(self) -> None:
        """
        Completely recreate the configured collection.

        Steps:
        1. Delete old collection
        2. Refresh local Qdrant client
        3. Verify old collection is gone
        4. Create fresh collection
        5. Refresh client again
        6. Verify fresh collection is empty
        """

        # ---------------------------------------------
        # Delete old collection
        # ---------------------------------------------

        if self.collection_exists():
            self.client.delete_collection(
                collection_name=self.collection_name
            )

        # Drop any stale local client state.
        self.refresh_client()

        # ---------------------------------------------
        # Verify deletion
        # ---------------------------------------------

        if self.collection_exists():
            raise RuntimeError(
                "Qdrant collection still exists "
                "after deletion."
            )

        # ---------------------------------------------
        # Create fresh collection
        # ---------------------------------------------

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=self.vector_size,
                distance=Distance.COSINE,
            ),
        )

        # Reopen once more after creation.
        self.refresh_client()

        # ---------------------------------------------
        # Verify new collection
        # ---------------------------------------------

        if not self.collection_exists():
            raise RuntimeError(
                "Qdrant collection could not be "
                "recreated."
            )

        fresh_count = self.count()

        if fresh_count != 0:
            raise RuntimeError(
                "Fresh Qdrant collection is not empty. "
                f"Expected 0 vectors, found "
                f"{fresh_count}."
            )

    # ---------------------------------------------------------
    # Vector Storage
    # ---------------------------------------------------------

    def add_points(
        self,
        ids: List[int],
        vectors: List[List[float]],
        payloads: List[Dict[str, Any]],
    ) -> None:
        """
        Add or update vector points in Qdrant.
        """

        if not (
            len(ids)
            == len(vectors)
            == len(payloads)
        ):
            raise ValueError(
                "ids, vectors and payloads must have "
                "the same length."
            )

        if not ids:
            return

        self.create_collection()

        points: List[PointStruct] = []

        for point_id, vector, payload in zip(
            ids,
            vectors,
            payloads,
        ):
            points.append(
                PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=payload,
                )
            )

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
            wait=True,
        )

    # ---------------------------------------------------------
    # Search
    # ---------------------------------------------------------

    def search(
        self,
        query_vector: List[float],
        limit: int = 3,
    ):
        """
        Search the collection for nearest vectors.
        """

        if limit <= 0:
            raise ValueError(
                "Search limit must be greater than zero."
            )

        if not self.collection_exists():
            return []

        result = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=limit,
            with_payload=True,
        )

        return result.points

    # ---------------------------------------------------------
    # Document Deletion
    # ---------------------------------------------------------

    def delete_by_filename(
        self,
        filename: str,
    ) -> None:
        """
        Delete all vectors belonging to one PDF.
        """

        safe_filename = Path(filename).name

        if not safe_filename:
            raise ValueError(
                "Filename cannot be empty."
            )

        if not self.collection_exists():
            return

        document_filter = Filter(
            must=[
                FieldCondition(
                    key="filename",
                    match=MatchValue(
                        value=safe_filename
                    ),
                )
            ]
        )

        self.client.delete(
            collection_name=self.collection_name,
            points_selector=document_filter,
            wait=True,
        )

    # ---------------------------------------------------------
    # Statistics
    # ---------------------------------------------------------

    def count(self) -> int:
        """
        Return total stored vector count.
        """

        if not self.collection_exists():
            return 0

        result = self.client.count(
            collection_name=self.collection_name,
            exact=True,
        )

        return int(result.count)

    def count_by_filename(
        self,
        filename: str,
    ) -> int:
        """
        Return vector count belonging to one PDF.
        """

        safe_filename = Path(filename).name

        if not safe_filename:
            return 0

        if not self.collection_exists():
            return 0

        document_filter = Filter(
            must=[
                FieldCondition(
                    key="filename",
                    match=MatchValue(
                        value=safe_filename
                    ),
                )
            ]
        )

        result = self.client.count(
            collection_name=self.collection_name,
            count_filter=document_filter,
            exact=True,
        )

        return int(result.count)

    # ---------------------------------------------------------
    # Cleanup
    # ---------------------------------------------------------

    def close(self) -> None:
        """
        Safely close the local Qdrant client.
        """

        try:
            self.client.close()
        except Exception:
            pass