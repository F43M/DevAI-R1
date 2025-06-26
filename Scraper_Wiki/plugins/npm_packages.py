"""Plugin for fetching information from the npm registry."""

from __future__ import annotations

from typing import Dict, List

import requests

from scraper_wiki import Config
from .base import Plugin


class NPMPackagesPlugin(Plugin):
    """Retrieve package metadata from npm."""

    def __init__(self, registry_url: str | None = None) -> None:
        self.registry_url = registry_url or "https://registry.npmjs.org"

    def fetch_items(self, lang: str, category: str) -> List[Dict]:
        """Return descriptor for a package."""
        url = f"{self.registry_url}/{category}"
        return [{"name": category, "url": url, "lang": lang, "category": category}]

    def parse_item(self, item: Dict) -> Dict:
        """Download and parse npm package metadata."""
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
            "description": data.get("description", ""),
            "readme": data.get("readme", ""),
        }
        record.setdefault("raw_code", "")
        record.setdefault("context", "")
        record.setdefault("problems", [])
        record.setdefault("fixed_version", "")
        record.setdefault("lessons", "")
        record.setdefault("origin_metrics", {})
        record.setdefault("challenge", "")
        return record


Plugin = NPMPackagesPlugin
