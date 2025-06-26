"""GitHub pull request and issue collector for code review datasets."""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import requests

from .base import Plugin


logger = logging.getLogger(__name__)


class GitHubReviewScraper(Plugin):  # type: ignore[misc]
    """Fetch pull request diffs and comments from GitHub repositories."""

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
        """Return pull requests with diffs and comments for ``category`` repo."""
        owner, repo = self._parse_repo(category)
        params = {"state": "closed"}
        if since:
            params["since"] = since
        url = f"{self.api_url}/repos/{owner}/{repo}/pulls"
        resp = requests.get(url, headers=self._headers(), params=params, timeout=30)
        resp.raise_for_status()
        pulls = resp.json()
        items: List[Dict] = []
        for pr in pulls:
            number = pr.get("number")
            if number is None:
                continue
            diff_resp = requests.get(
                f"{self.api_url}/repos/{owner}/{repo}/pulls/{number}",
                headers={**self._headers(), "Accept": "application/vnd.github.v3.diff"},
                timeout=30,
            )
            diff_resp.raise_for_status()
            diff = diff_resp.text
            comments_resp = requests.get(
                f"{self.api_url}/repos/{owner}/{repo}/issues/{number}/comments",
                headers=self._headers(),
                timeout=30,
            )
            comments_resp.raise_for_status()
            comments = [c.get("body", "") for c in comments_resp.json()]
            items.append(
                {
                    "lang": lang,
                    "category": category,
                    "title": pr.get("title", ""),
                    "diff": diff,
                    "comments": comments,
                    "html_url": pr.get("html_url", ""),
                }
            )
        return items

    def parse_item(self, item: Dict) -> Dict:
        """Return record combining diff and comments."""
        record = {
            "title": item.get("title", ""),
            "language": item.get("lang", "en"),
            "category": item.get("category", ""),
            "diff": item.get("diff", ""),
            "comments": item.get("comments", []),
            "link": item.get("html_url", ""),
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
Plugin = GitHubReviewScraper
