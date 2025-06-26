"""GitLab repository scraping utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import requests

from .base import Plugin


class GitLabScraper:
    """Fetch projects and code data from GitLab."""

    def __init__(self, token: str | None = None, api_url: str | None = None) -> None:
        self.api_url = api_url or "https://gitlab.com/api/v4"
        self.token = token
        self._code_exts = {".py", ".js", ".ts", ".java", ".go", ".rb", ".php", ".rs"}

    def _headers(self) -> Dict[str, str]:
        headers: Dict[str, str] = {}
        if self.token:
            headers["PRIVATE-TOKEN"] = self.token
        return headers

    def search_repositories(
        self,
        language: str,
        min_stars: int,
        per_page: int = 20,
        since: str | None = None,
    ) -> List[Dict]:
        """Return GitLab projects filtered by language and star count."""
        params = {
            "search": language,
            "order_by": "star_count",
            "sort": "desc",
            "per_page": per_page,
            "simple": True,
        }
        if since:
            params["last_activity_after"] = since
        resp = requests.get(
            f"{self.api_url}/projects",
            headers=self._headers(),
            params=params,
        )
        resp.raise_for_status()
        items = resp.json()
        return [it for it in items if it.get("star_count", 0) >= min_stars]

    def _list_code_files(self, project_id: int, branch: str) -> List[str]:
        url = f"{self.api_url}/projects/{project_id}/repository/tree"
        resp = requests.get(
            url,
            headers=self._headers(),
            params={"recursive": True, "ref": branch, "per_page": 100},
        )
        resp.raise_for_status()
        tree = resp.json()
        paths = [
            it.get("path")
            for it in tree
            if it.get("type") == "blob"
            and Path(it.get("path", "")).suffix in self._code_exts
        ]
        return paths

    def collect_repository_data(self, repo: Dict) -> Dict:
        """Return project metrics and file list."""
        project_id = repo.get("id")
        if project_id is None:
            return {}
        branch = repo.get("default_branch", "master")
        file_paths = self._list_code_files(project_id, branch)
        has_tests = any(
            "test" in Path(p).name.lower() or "test" in Path(p).parts[0].lower()
            for p in file_paths
        )
        stars = repo.get("star_count", 0)
        issues = repo.get("open_issues_count", 0)
        quality_score = stars / (issues + 1)
        record = {
            "repository": repo.get("path_with_namespace"),
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


class Plugin(GitLabScraper):
    """Alias for plugin registry."""
