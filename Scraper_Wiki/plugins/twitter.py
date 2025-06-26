"""Twitter scraping plugin using the v2 API."""

from __future__ import annotations

from typing import Dict, List

import requests

from .base import BasePlugin


class TwitterPlugin(BasePlugin):
    """Fetch tweets matching a search query."""

    def __init__(self, bearer_token: str | None = None) -> None:
        super().__init__()
        self.bearer_token = bearer_token or ""
        self.api_url = "https://api.twitter.com/2/tweets/search/recent"

    def fetch_items(self, lang: str, category: str) -> List[Dict]:
        """Return recent tweets for the given query."""
        headers = {"Authorization": f"Bearer {self.bearer_token}"}
        params = {"query": category, "max_results": 10}
        resp = self.request(self.api_url, headers=headers, params=params)
        data = resp.json()
        items = data.get("data", [])
        for it in items:
            it["lang"] = lang
            it["category"] = category
        return self.deduplicate(items)

    def parse_item(self, item: Dict) -> Dict:
        """Convert a tweet to a dataset record."""
        record = {
            "id": item.get("id"),
            "language": item.get("lang", "en"),
            "category": item.get("category", ""),
            "content": item.get("text", ""),
            "author_id": item.get("author_id"),
            "created_at": item.get("created_at"),
        }
        record.setdefault("raw_code", "")
        record.setdefault("context", "")
        record.setdefault("problems", [])
        record.setdefault("fixed_version", "")
        record.setdefault("lessons", "")
        record.setdefault("origin_metrics", {})
        record.setdefault("challenge", "")
        return record


Plugin = TwitterPlugin
