"""Plugin for fetching package info from CRAN."""

from __future__ import annotations

from typing import Dict, List

import requests

from scraper_wiki import Config
from .base import Plugin


class CRANPackagesPlugin(Plugin):
    """Retrieve package metadata from CRAN."""

    def __init__(self, base_url: str | None = None) -> None:
        self.base_url = base_url or "https://crandb.r-pkg.org"

    def fetch_items(self, lang: str, category: str) -> List[Dict]:
        """Return descriptor for a package."""
        url = f"{self.base_url}/{category}"
        return [{"name": category, "url": url, "lang": lang, "category": category}]

    def parse_item(self, item: Dict) -> Dict:
        """Download and parse CRAN metadata."""
        url = item.get("url")
        if not url:
            return {}
        resp = requests.get(url, timeout=Config.TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        record = {
            "name": item.get("name", ""),
            "language": item.get("lang", "en"),
            "category": item.get("category", ""),
            "title": data.get("Title", ""),
            "description": data.get("Description", ""),
        }
        record.setdefault("raw_code", "")
        record.setdefault("context", "")
        record.setdefault("problems", [])
        record.setdefault("fixed_version", "")
        record.setdefault("lessons", "")
        record.setdefault("origin_metrics", {})
        record.setdefault("challenge", "")
        return record


Plugin = CRANPackagesPlugin
