"""Plugin for fetching RFC documents."""

from __future__ import annotations

from typing import Dict, List

import requests

from scraper_wiki import Config
from .base import Plugin


class RFCScraper(Plugin):
    """Retrieve RFC text documents."""

    def __init__(self, base_url: str | None = None) -> None:
        self.base_url = base_url or "https://www.rfc-editor.org/rfc"

    def fetch_items(self, lang: str, category: str) -> List[Dict]:
        """Return a descriptor for an RFC file."""
        if category.startswith("http"):
            url = category
        else:
            num = category.lstrip("rfc").strip()
            url = f"{self.base_url}/rfc{num}.txt"
        return [{"url": url, "lang": lang, "category": category}]

    def parse_item(self, item: Dict) -> Dict:
        """Download the RFC file."""
        url = item.get("url")
        if not url:
            return {}
        resp = requests.get(url, timeout=Config.TIMEOUT)
        resp.raise_for_status()
        text = resp.text
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


Plugin = RFCScraper
