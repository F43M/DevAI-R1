"""Plugin for scraping Python official documentation."""

from __future__ import annotations

from typing import Dict, List

import html2text

from .base import BasePlugin


class PythonDocsPlugin(BasePlugin):
    """Fetch pages from docs.python.org."""

    def __init__(self, base_url: str | None = None) -> None:
        super().__init__()
        self.base_url = base_url or "https://docs.python.org/3"

    def fetch_items(self, lang: str, category: str) -> List[Dict]:
        """Return descriptor for a documentation page."""
        url = f"{self.base_url}/{category}"
        return [{"url": url, "lang": lang, "category": category}]

    def parse_item(self, item: Dict) -> Dict:
        """Download and parse a documentation page."""
        url = item.get("url")
        if not url:
            return {}
        resp = self.request(url)
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


Plugin = PythonDocsPlugin
