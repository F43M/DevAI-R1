"""Plugin for scraping DevDocs documentation."""

from __future__ import annotations

from typing import Dict, List

import html2text
import requests

from scraper_wiki import Config
from .base import Plugin


class DevDocsPlugin(Plugin):
    """Fetch pages from DevDocs."""

    def __init__(self, base_url: str | None = None) -> None:
        self.base_url = base_url or "https://devdocs.io"

    def fetch_items(self, lang: str, category: str) -> List[Dict]:
        """Return a descriptor for a DevDocs page."""
        url = f"{self.base_url}/{category}"
        return [{"url": url, "lang": lang, "category": category}]

    def parse_item(self, item: Dict) -> Dict:
        """Download and parse a DevDocs page."""
        url = item.get("url")
        if not url:
            return {}
        resp = requests.get(url, timeout=Config.TIMEOUT)
        resp.raise_for_status()
        html = resp.text
        text = html2text.html2text(html) if hasattr(html2text, "html2text") else html
        record = {
            "url": url,
            "language": item.get("lang", "en"),
            "category": item.get("category", ""),
            "content": text,
        }
        record.setdefault("raw_code", "")
        record.setdefault("context", "")
        record.setdefault("problems", [])
        record.setdefault("fixed_version", "")
        record.setdefault("lessons", "")
        record.setdefault("origin_metrics", {})
        record.setdefault("challenge", "")
        return record


Plugin = DevDocsPlugin
