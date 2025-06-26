"""SourceForge repository scraping utilities."""

from __future__ import annotations

from typing import Dict, List

import requests

from scraper_wiki import Config
from .base import Plugin


class SourceForgeScraper(Plugin):  # type: ignore[misc]
    """Fetch project information from SourceForge."""

    def __init__(self, api_url: str | None = None) -> None:
        self.api_url = api_url or "https://sourceforge.net/rest/p"

    def fetch_items(self, lang: str, category: str) -> List[Dict]:
        """Return descriptor for a SourceForge project."""
        return [{"name": category, "lang": lang, "category": category}]

    def _fetch_project(self, name: str) -> Dict:
        resp = requests.get(f"{self.api_url}/{name}", timeout=Config.TIMEOUT)
        resp.raise_for_status()
        return resp.json()

    def parse_item(self, item: Dict) -> Dict:
        """Return project metadata."""
        name = item.get("name")
        if not name:
            return {}
        data = self._fetch_project(name)
        record = {
            "name": data.get("name", ""),
            "language": item.get("lang", "en"),
            "category": item.get("category", ""),
            "summary": data.get("summary", ""),
            "description": data.get("description", ""),
        }
        record.setdefault("raw_code", "")
        record.setdefault("context", "")
        record.setdefault("problems", [])
        record.setdefault("fixed_version", "")
        record.setdefault("lessons", "")
        record.setdefault("origin_metrics", {})
        record.setdefault("challenge", "")
        return record


Plugin = SourceForgeScraper
