"""GitHub code extraction utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import requests

from .base import Plugin


class CodeExtractor:
    """Extract repository code and metrics from GitHub."""

    def __init__(self, token: str | None = None, api_url: str | None = None) -> None:
        self.api_url = api_url or "https://api.github.com"
        self.token = token
        self._code_exts = {".py", ".js", ".ts", ".java", ".go", ".rb", ".php", ".rs"}

    def _headers(self) -> Dict[str, str]:
        headers = {"Accept": "application/vnd.github.v3+json"}
        if self.token:
            headers["Authorization"] = f"token {self.token}"
        return headers

    def search_repositories(
        self, language: str, min_stars: int, per_page: int = 10
    ) -> List[Dict]:
        """Return repositories matching language and star count."""
        query = f"language:{language} stars:>={min_stars}"
        params = {"q": query, "sort": "stars", "order": "desc", "per_page": per_page}
        resp = requests.get(
            f"{self.api_url}/search/repositories",
            headers=self._headers(),
            params=params,
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("items", [])

    def _list_code_files(self, full_name: str, branch: str) -> List[str]:
        url = f"{self.api_url}/repos/{full_name}/git/trees/{branch}?recursive=1"
        resp = requests.get(url, headers=self._headers())
        resp.raise_for_status()
        tree = resp.json().get("tree", [])
        paths = [
            it["path"]
            for it in tree
            if it.get("type") == "blob"
            and Path(it.get("path", "")).suffix in self._code_exts
        ]
        return paths

    def download_repository_files(
        self, full_name: str, branch: str | None = None
    ) -> Dict[str, str]:
        """Download source files for a repository."""
        branch = branch or "master"
        files = {}
        for path in self._list_code_files(full_name, branch):
            url = f"{self.api_url}/repos/{full_name}/contents/{path}"
            resp = requests.get(
                url,
                headers={"Accept": "application/vnd.github.v3.raw"},
                params={"ref": branch},
            )
            resp.raise_for_status()
            files[path] = resp.text
        return files

    def collect_repository_data(self, repo: Dict) -> Dict:
        """Return repository metrics and source file list."""
        full_name = repo.get("full_name")
        if not full_name:
            return {}
        branch = repo.get("default_branch", "master")
        file_paths = self._list_code_files(full_name, branch)
        has_tests = any(
            "test" in Path(p).parts[0].lower() or "test" in Path(p).name.lower()
            for p in file_paths
        )
        stars = repo.get("stargazers_count", 0)
        issues = repo.get("open_issues_count", 0)
        quality_score = stars / (issues + 1)
        record = {
            "repository": full_name,
            "stars": stars,
            "open_issues": issues,
            "has_tests": has_tests,
            "files": file_paths,
            "context": repo.get("description", ""),
            "tests": has_tests,
            "docstring": "",
            "quality_score": quality_score,
        }
        record.setdefault("raw_code", "")
        record.setdefault("problems", [])
        record.setdefault("fixed_version", "")
        record.setdefault("lessons", "")
        record.setdefault("origin_metrics", {})
        record.setdefault("challenge", "")
        return record


class Plugin(CodeExtractor):
    """Alias for plugin registry."""
