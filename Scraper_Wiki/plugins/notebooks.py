"""Public notebook scraping plugin."""

from __future__ import annotations

from typing import Dict, List

import requests

from scraper_wiki import Config
from .base import Plugin


class NotebooksPlugin(Plugin):  # type: ignore[misc]
    """Fetch raw notebook files from GitHub or Kaggle."""

    def fetch_items(self, lang: str, category: str) -> List[Dict]:
        """Return descriptor for a notebook URL."""
        return [{"url": category, "lang": lang, "category": category}]

    def _fetch_notebook(self, url: str) -> str:
        resp = requests.get(url, timeout=Config.TIMEOUT)
        resp.raise_for_status()
        return resp.text

    def parse_item(self, item: Dict) -> Dict:
        """Download the notebook content."""
        url = item.get("url")
        if not url:
            return {}
        content = self._fetch_notebook(url)
        record = {
            "url": url,
            "language": item.get("lang", "en"),
            "category": item.get("category", ""),
            "notebook": content,
        }
        record.setdefault("raw_code", "")
        record.setdefault("context", "")
        record.setdefault("problems", [])
        record.setdefault("fixed_version", "")
        record.setdefault("lessons", "")
        record.setdefault("origin_metrics", {})
        record.setdefault("challenge", "")
        return record


Plugin = NotebooksPlugin
