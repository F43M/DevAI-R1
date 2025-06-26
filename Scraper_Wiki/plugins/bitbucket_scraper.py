"""Bitbucket repository scraping utilities."""

from __future__ import annotations

from typing import Dict, List, Tuple

import requests

from scraper_wiki import Config
from .base import Plugin


class BitbucketScraper:
    """Minimal interface for the Bitbucket API."""

    def __init__(self, api_url: str | None = None) -> None:
        self.api_url = api_url or "https://api.bitbucket.org/2.0"

    def _parse_repo(self, repo_url: str) -> Tuple[str, str]:
        parts = repo_url.rstrip("/").split("/")[-2:]
        if len(parts) != 2:
            raise ValueError("Invalid Bitbucket repository URL")
        return parts[0], parts[1]

    def get_readme(self, repo_url: str) -> str:
        owner, repo = self._parse_repo(repo_url)
        url = f"{self.api_url}/repositories/{owner}/{repo}/src/HEAD/README.md"
        resp = requests.get(url, timeout=Config.TIMEOUT)
        resp.raise_for_status()
        return resp.text

    def get_issues(self, repo_url: str) -> List[Dict]:
        owner, repo = self._parse_repo(repo_url)
        url = f"{self.api_url}/repositories/{owner}/{repo}/issues"
        resp = requests.get(url, timeout=Config.TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, list):
            return data
        return data.get("values", [])

    def build_dataset_record(self, repo_url: str) -> Dict:
        readme = self.get_readme(repo_url)
        issues = self.get_issues(repo_url)
        record = {
            "repository": repo_url,
            "readme": readme,
            "issues": issues,
        }
        record.setdefault("raw_code", "")
        record.setdefault("context", "")
        record.setdefault("problems", [])
        record.setdefault("fixed_version", "")
        record.setdefault("lessons", "")
        record.setdefault("origin_metrics", {})
        record.setdefault("challenge", "")
        return record


class Plugin(BitbucketScraper):
    """Alias for plugin registry."""
