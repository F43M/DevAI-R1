from typing import List


class VectorStorage:
    """Simple interface for vector databases like Milvus or Weaviate."""

    def __init__(self, client):
        self.client = client

    def save_embeddings(self, data: List[dict]) -> None:
        if not data:
            return
        for row in data:
            self._insert(row)

    def _insert(self, row: dict) -> None:
        try:
            self.client.insert(row)
        except Exception:  # pragma: no cover - integration depends on client
            pass
