"""Search utilities backed by Elasticsearch."""

from .indexer import bulk_index, query_index

__all__ = ["bulk_index", "query_index"]
