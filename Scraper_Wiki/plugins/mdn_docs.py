"""Plugin for scraping MDN documentation."""

from __future__ import annotations

from typing import Dict, List

import html2text
import requests

from scraper_wiki import Config
from .base import Plugin


class MDNDocsPlugin(Plugin):
    """Fetch pages from MDN documentation."""

    def fetch_items(self, lang: str, category: str) -> List[Dict]:
        """Return basic descriptors for an MDN page."""
        url = f"https://developer.mozilla.org/{lang}/docs/{category}"
        return [{"url": url, "lang": lang, "category": category}]

    def parse_item(self, item: Dict) -> Dict:
        """Download and parse an MDN page."""
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


Plugin = MDNDocsPlugin
