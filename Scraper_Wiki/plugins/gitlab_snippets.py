"""GitLab snippets scraping utilities."""

from __future__ import annotations

from typing import Dict, List

import requests

from .base import Plugin


class GitLabSnippets(Plugin):  # type: ignore[misc]
    """Fetch snippets from GitLab."""

    def __init__(self, token: str | None = None, api_url: str | None = None) -> None:
        self.api_url = api_url or "https://gitlab.com/api/v4"
        self.token = token

    def _headers(self) -> Dict[str, str]:
        headers: Dict[str, str] = {}
        if self.token:
            headers["PRIVATE-TOKEN"] = self.token
        return headers

    def fetch_items(self, lang: str, category: str) -> List[Dict]:
        """Return snippets matching the search query."""
        resp = requests.get(
            f"{self.api_url}/snippets",
            headers=self._headers(),
            params={"search": category, "per_page": 20},
        )
        resp.raise_for_status()
        items = resp.json()
        for it in items:
            it["lang"] = lang
            it["category"] = category
        return items

    def parse_item(self, item: Dict) -> Dict:
        """Return snippet title and raw code."""
        snippet_id = item.get("id")
        if snippet_id is None:
            return {}
        resp = requests.get(
            f"{self.api_url}/snippets/{snippet_id}/raw",
            headers=self._headers(),
        )
        resp.raise_for_status()
        record = {
            "title": item.get("title", ""),
            "language": item.get("lang", "en"),
            "category": item.get("category", ""),
            "code": resp.text,
        }
        record.setdefault("raw_code", "")
        record.setdefault("context", "")
        record.setdefault("problems", [])
        record.setdefault("fixed_version", "")
        record.setdefault("lessons", "")
        record.setdefault("origin_metrics", {})
        record.setdefault("challenge", "")
        return record


Plugin = GitLabSnippets
