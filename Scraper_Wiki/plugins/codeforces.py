"""Codeforces scraping plugin."""

from __future__ import annotations

from typing import Dict, List

import requests

from scraper_wiki import Config
from .base import Plugin


class CodeforcesPlugin(Plugin):  # type: ignore[misc]
    """Fetch problems from Codeforces."""

    def __init__(self, api_url: str | None = None) -> None:
        self.api_url = api_url or "https://codeforces.com/api"

    def fetch_items(self, lang: str, category: str) -> List[Dict]:
        """Return descriptor for a Codeforces problem."""
        return [{"slug": category, "lang": lang, "category": category}]

    def _fetch_problem(self, slug: str) -> Dict:
        resp = requests.get(
            f"{self.api_url}/problemset/problem/{slug}", timeout=Config.TIMEOUT
        )
        resp.raise_for_status()
        return resp.json()

    def _fetch_discussion(self, slug: str) -> Dict:
        resp = requests.get(
            f"{self.api_url}/problemset/discussion/{slug}", timeout=Config.TIMEOUT
        )
        resp.raise_for_status()
        return resp.json()

    def parse_item(self, item: Dict) -> Dict:
        """Return problem statement, solution and discussion."""
        slug = item.get("slug")
        if not slug:
            return {}
        problem = self._fetch_problem(slug)
        discussion = self._fetch_discussion(slug)
        record = {
            "title": problem.get("name", ""),
            "language": item.get("lang", "en"),
            "category": item.get("category", ""),
            "problem": problem.get("statement", ""),
            "solution": problem.get("solution", ""),
            "discussion": discussion.get("content", ""),
        }
        record.setdefault("raw_code", "")
        record.setdefault("context", "")
        record.setdefault("problems", [])
        record.setdefault("fixed_version", "")
        record.setdefault("lessons", "")
        record.setdefault("origin_metrics", {})
        record.setdefault("challenge", "")
        return record


Plugin = CodeforcesPlugin
