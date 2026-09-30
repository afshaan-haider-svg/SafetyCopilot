"""
SafetyCopilot — Qdrant Vector Store

Provides a wrapper around Qdrant for storing, retrieving,
deleting, and rebuilding SafetyCopilot embeddings.

Supports:
- Local persistent Qdrant for development
- Qdrant Cloud for deployment
"""

from __future__ import annotations

import os
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
    Qdrant vector store used by SafetyCopilot.

    Automatically selects the Qdrant backend:

    Local development:
        data/qdrant

    Cloud deployment:
        QDRANT_URL
        QDRANT_API_KEY
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

        # -----------------------------------------------------
        # Cloud configuration
        # -----------------------------------------------------

        self.qdrant_url = os.getenv(
            "QDRANT_URL",
            "",
        ).strip()

        self.qdrant_api_key = os.getenv(
            "QDRANT_API_KEY",
            "",
        ).strip()

        # QDRANT_URL present -> Cloud
        # QDRANT_URL absent  -> Local
        self.use_cloud = bool(self.qdrant_url)

        if not self.use_cloud:
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
        Create either a Qdrant Cloud client or a local
        persistent Qdrant client.
        """

        if self.use_cloud:

            if not self.qdrant_api_key:
                raise RuntimeError(
                    "QDRANT_URL is configured but "
                    "QDRANT_API_KEY is missing."
                )

            return QdrantClient(
                url=self.qdrant_url,
                api_key=self.qdrant_api_key,
                timeout=60,
            )

        return QdrantClient(
            path=str(self.storage_path)
        )

    def refresh_client(self) -> None:
        """
        Refresh the Qdrant client.

        Local embedded Qdrant benefits from reopening the
        storage after destructive collection operations.

        Qdrant Cloud does not require this refresh, so the
        existing remote client is retained.
        """

        if self.use_cloud:
            return

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

        If recreate=True, delete any existing collection
        and create a fresh empty collection.
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
        """

        if self.collection_exists():
            self.client.delete_collection(
                collection_name=self.collection_name
            )

        # Only local embedded Qdrant requires client refresh.
        if not self.use_cloud:
            self.refresh_client()

    def recreate_collection(self) -> None:
        """
        Completely recreate the configured collection.

        Cloud mode:
        - Delete existing remote collection if present
        - Create a new remote collection
        - Keep the same remote client

        Local mode:
        - Delete existing collection
        - Refresh embedded client
        - Create a new collection
        - Refresh embedded client again
        - Verify the collection is empty
        """

        # =====================================================
        # QDRANT CLOUD
        # =====================================================

        if self.use_cloud:

            if self.collection_exists():
                self.client.delete_collection(
                    collection_name=self.collection_name
                )

            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE,
                ),
            )

            if not self.collection_exists():
                raise RuntimeError(
                    "Qdrant Cloud collection could not "
                    "be recreated."
                )

            # Important:
            # Do NOT refresh the remote client here.
            # Do NOT perform the local stale-state check.
            return

        # =====================================================
        # LOCAL QDRANT
        # =====================================================

        if self.collection_exists():
            self.client.delete_collection(
                collection_name=self.collection_name
            )

        # Embedded local storage may keep stale state,
        # therefore reopen the client.
        self.refresh_client()

        if self.collection_exists():
            raise RuntimeError(
                "Qdrant collection still exists "
                "after deletion."
            )

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=self.vector_size,
                distance=Distance.COSINE,
            ),
        )

        self.refresh_client()

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
        Safely close the Qdrant client.
        """

        try:
            self.client.close()
        except Exception:
            pass