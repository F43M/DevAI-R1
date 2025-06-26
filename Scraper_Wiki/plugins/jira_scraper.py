"""Plugin for scraping JIRA issue data."""

from __future__ import annotations

from typing import Dict, List

import requests

from scraper_wiki import Config
from .base import Plugin


class JiraScraper(Plugin):
    """Retrieve issue information from a JIRA instance."""

    def __init__(self, base_url: str | None = None) -> None:
        self.base_url = base_url or "https://jira.example.com/rest/api/2/issue"

    def fetch_items(self, lang: str, category: str) -> List[Dict]:
        """Return descriptor for an issue."""
        url = f"{self.base_url}/{category}"
        return [{"issue": category, "url": url, "lang": lang, "category": category}]

    def parse_item(self, item: Dict) -> Dict:
        """Download and parse issue metadata."""
        url = item.get("url")
        if not url:
            return {}
        resp = requests.get(url, timeout=Config.TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        fields = data.get("fields", {})
        record = {
            "issue": item.get("issue", ""),
            "language": item.get("lang", "en"),
            "category": item.get("category", ""),
            "title": fields.get("summary", ""),
            "description": fields.get("description", ""),
        }
        record.setdefault("raw_code", "")
        record.setdefault("context", "")
        record.setdefault("problems", [])
        record.setdefault("fixed_version", "")
        record.setdefault("lessons", "")
        record.setdefault("origin_metrics", {})
        record.setdefault("challenge", "")
        return record


Plugin = JiraScraper
