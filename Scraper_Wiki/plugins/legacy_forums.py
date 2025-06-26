"""Scrape legacy discussion forums like Yahoo Answers and Delphi/PHP boards."""

from __future__ import annotations

from typing import List, Dict

import requests

from scraper_wiki import Config
from .base import Plugin


class LegacyForumsPlugin(Plugin):
    """Fetch question/answer pairs from multiple legacy forums."""

    def __init__(
        self,
        yahoo_url: str | None = None,
        delphi_url: str | None = None,
        php_url: str | None = None,
    ) -> None:
        self.yahoo_url = yahoo_url or "https://answers.yahoo.com/api"
        self.delphi_url = delphi_url or "https://forum.delphi.com/api"
        self.php_url = php_url or "https://forum.phpdeveloper.com/api"

    def fetch_items(self, lang: str, category: str) -> List[Dict]:
        """Return basic query descriptors for all sources."""
        return [
            {"source": "yahoo", "query": category, "lang": lang},
            {"source": "delphi", "query": category, "lang": lang},
            {"source": "php", "query": category, "lang": lang},
        ]

    def _fetch(self, base: str, query: str) -> Dict:
        resp = requests.get(f"{base}?q={query}", timeout=Config.TIMEOUT)
        resp.raise_for_status()
        return resp.json()

    def parse_item(self, item: Dict) -> Dict:
        """Return a unified question/answer pair."""
        source = item.get("source")
        query = item.get("query", "")
        if source == "yahoo":
            data = self._fetch(self.yahoo_url, query)
        elif source == "delphi":
            data = self._fetch(self.delphi_url, query)
        elif source == "php":
            data = self._fetch(self.php_url, query)
        else:
            return {}
        question = data.get("question") or data.get("title", "")
        answer = data.get("answer") or data.get("reply") or data.get("content", "")
        record = {
            "question": question,
            "answer": answer,
            "source": source,
            "language": item.get("lang", "en"),
        }
        record.setdefault("raw_code", "")
        record.setdefault("context", "")
        record.setdefault("problems", [])
        record.setdefault("fixed_version", "")
        record.setdefault("lessons", "")
        record.setdefault("origin_metrics", {})
        record.setdefault("challenge", "")
        return record


# Registry alias
Plugin = LegacyForumsPlugin
