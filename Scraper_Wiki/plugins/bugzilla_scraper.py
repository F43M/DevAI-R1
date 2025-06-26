"""Plugin for scraping Bugzilla bug data."""

from __future__ import annotations

from typing import Dict, List

import requests

from scraper_wiki import Config
from .base import Plugin


class BugzillaScraper(Plugin):
    """Retrieve bug information from Bugzilla."""

    def __init__(self, base_url: str | None = None) -> None:
        self.base_url = base_url or "https://bugzilla.mozilla.org/rest/bug"

    def fetch_items(self, lang: str, category: str) -> List[Dict]:
        """Return descriptor for a bug."""
        url = f"{self.base_url}/{category}"
        return [{"bug": category, "url": url, "lang": lang, "category": category}]

    def parse_item(self, item: Dict) -> Dict:
        """Download and parse bug metadata."""
        url = item.get("url")
        if not url:
            return {}
        resp = requests.get(url, timeout=Config.TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        bug = (data.get("bugs") or data.get("bug") or [{}])[0]
        record = {
            "bug": item.get("bug", ""),
            "language": item.get("lang", "en"),
            "category": item.get("category", ""),
            "title": bug.get("summary", ""),
            "description": bug.get("description", ""),
        }
        record.setdefault("raw_code", "")
        record.setdefault("context", "")
        record.setdefault("problems", [])
        record.setdefault("fixed_version", "")
        record.setdefault("lessons", "")
        record.setdefault("origin_metrics", {})
        record.setdefault("challenge", "")
        return record


Plugin = BugzillaScraper
