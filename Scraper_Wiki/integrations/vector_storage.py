"""Vector storage drivers for Milvus and Weaviate."""

from __future__ import annotations

from typing import List, Any


class BaseVectorStore:
    """Base class for vector databases."""

    def __init__(self, client: Any):
        self.client = client

    def save_embeddings(self, data: List[dict]) -> None:
        """Insert embeddings into the underlying store."""
        if not data:
            return
        for row in data:
            self._insert(row)

    def _insert(self, row: dict) -> None:
        try:
            self.client.insert(row)
        except Exception:  # pragma: no cover - depends on external DB
            pass


class MilvusVectorStore(BaseVectorStore):
    """Milvus based vector store."""

    def __init__(
        self, uri: str = "http://localhost:19530", collection: str = "embeddings"
    ):
        try:
            from pymilvus import connections, Collection
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ImportError("pymilvus is required for MilvusVectorStore") from exc

        connections.connect(uri=uri)
        client = Collection(collection)
        super().__init__(client)


class WeaviateVectorStore(BaseVectorStore):
    """Weaviate based vector store."""

    def __init__(self, uri: str = "http://localhost:8080"):
        try:
            import weaviate
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "weaviate-client is required for WeaviateVectorStore"
            ) from exc

        client = weaviate.Client(uri)
        super().__init__(client)
