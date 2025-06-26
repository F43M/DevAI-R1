"""Fetch commit histories from notable security bug repositories."""

from __future__ import annotations

from typing import Dict, List, Tuple

import requests

from .base import Plugin


class BugHistoryScraper(Plugin):  # type: ignore[misc]
    """Retrieve commits from public Git repositories."""

    def __init__(self, token: str | None = None, api_url: str | None = None) -> None:
        self.api_url = api_url or "https://api.github.com"
        self.token = token

    def _headers(self) -> Dict[str, str]:
        headers = {"Accept": "application/vnd.github.v3+json"}
        if self.token:
            headers["Authorization"] = f"token {self.token}"
        return headers

    def _parse_repo(self, repo_url: str) -> Tuple[str, str]:
        parts = repo_url.rstrip("/").split("/")[-2:]
        if len(parts) != 2:
            raise ValueError("Invalid GitHub repository URL")
        return parts[0], parts[1]

    def fetch_items(
        self, lang: str, category: str, since: str | None = None
    ) -> List[Dict]:
        """Return commit metadata for the repository URL provided in ``category``."""
        owner, repo = self._parse_repo(category)
        url = f"{self.api_url}/repos/{owner}/{repo}/commits"
        params = {"since": since} if since else None
        resp = requests.get(url, headers=self._headers(), params=params)
        resp.raise_for_status()
        commits = resp.json()
        for c in commits:
            c["lang"] = lang
            c["category"] = category
        return commits

    def parse_item(self, item: Dict) -> Dict:
        """Return key commit details."""
        record = {
            "commit": item.get("sha", ""),
            "language": item.get("lang", "en"),
            "category": item.get("category", ""),
            "message": item.get("commit", {}).get("message", ""),
        }
        record.setdefault("raw_code", "")
        record.setdefault("context", "")
        record.setdefault("problems", [])
        record.setdefault("fixed_version", "")
        record.setdefault("lessons", "")
        record.setdefault("origin_metrics", {})
        record.setdefault("challenge", "")
        return record


# Registry alias
Plugin = BugHistoryScraper
