"""Simple GitHub Gist scraping utilities."""

from __future__ import annotations

from typing import Dict, List

import requests


class GistScraper:
    """Fetch public gists from GitHub."""

    def __init__(self, token: str | None = None, api_url: str | None = None) -> None:
        self.api_url = api_url or "https://api.github.com"
        self.token = token

    def _headers(self) -> Dict[str, str]:
        headers = {"Accept": "application/vnd.github.v3+json"}
        if self.token:
            headers["Authorization"] = f"token {self.token}"
        return headers

    def fetch_items(self, username: str) -> List[Dict]:
        """Return gists for the provided GitHub username."""
        url = f"{self.api_url}/users/{username}/gists"
        resp = requests.get(url, headers=self._headers())
        resp.raise_for_status()
        return resp.json()

    def parse_item(self, item: Dict) -> Dict:
        """Return gist description and file contents."""
        files: Dict[str, str] = {}
        for info in item.get("files", {}).values():
            raw_url = info.get("raw_url")
            if not raw_url:
                continue
            resp = requests.get(raw_url, headers=self._headers())
            resp.raise_for_status()
            files[info.get("filename", "file")] = resp.text
        record = {"description": item.get("description", ""), "files": files}
        record.setdefault("raw_code", "")
        record.setdefault("context", "")
        record.setdefault("problems", [])
        record.setdefault("fixed_version", "")
        record.setdefault("lessons", "")
        record.setdefault("origin_metrics", {})
        record.setdefault("challenge", "")
        return record


# Registry alias
Plugin = GistScraper
