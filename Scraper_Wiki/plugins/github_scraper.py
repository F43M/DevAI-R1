"""Utilities for scraping GitHub repositories."""

from __future__ import annotations

from typing import List, Dict, Tuple

import requests


class GitHubScraper:
    """Minimal interface for the GitHub API."""

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

    def get_repo_code(self, repo_url: str) -> str:
        """Return repository README text using the GitHub API."""
        owner, repo = self._parse_repo(repo_url)
        url = f"{self.api_url}/repos/{owner}/{repo}/readme"
        resp = requests.get(url, headers={"Accept": "application/vnd.github.v3.raw"})
        resp.raise_for_status()
        return resp.text

    def get_issues(self, repo_url: str, state: str = "open") -> List[Dict]:
        """Return issues for the repository."""
        owner, repo = self._parse_repo(repo_url)
        url = f"{self.api_url}/repos/{owner}/{repo}/issues"
        params = {"state": state}
        resp = requests.get(url, headers=self._headers(), params=params)
        resp.raise_for_status()
        return resp.json()

    def get_commits(self, repo_url: str, since: str | None = None) -> List[Dict]:
        """Return commit metadata for the repository."""
        owner, repo = self._parse_repo(repo_url)
        url = f"{self.api_url}/repos/{owner}/{repo}/commits"
        params = {"since": since} if since else None
        resp = requests.get(url, headers=self._headers(), params=params)
        resp.raise_for_status()
        return resp.json()

    def get_issue_commit_pairs(
        self, repo_url: str, state: str = "closed", since: str | None = None
    ) -> List[Tuple[Dict, Dict]]:
        """Return pairs of issues and commits that reference them."""
        issues = self.get_issues(repo_url, state=state)
        commits = self.get_commits(repo_url, since=since)
        pairs: List[Tuple[Dict, Dict]] = []
        for issue in issues:
            number = issue.get("number")
            if not number:
                continue
            token = f"#{number}"
            for commit in commits:
                message = commit.get("commit", {}).get("message", "")
                if token in message:
                    pairs.append((issue, commit))
                    break
        return pairs

    def build_problem_solution_records(
        self, repo_url: str, since: str | None = None
    ) -> List[Dict]:
        """Return dataset records combining issues and closing commits."""
        records = []
        for issue, commit in self.get_issue_commit_pairs(repo_url, since=since):
            problem = f"{issue.get('title', '')}\n\n{issue.get('body', '')}"
            solution = commit.get("commit", {}).get("message", "")
            link = issue.get("html_url") or issue.get("url")
            rec = {
                "problem": problem,
                "solution": solution,
            }
            rec["discussion_links"] = [link] if link else []
            records.append(rec)
        return records


class Plugin(GitHubScraper):
    """Alias for plugin registry compatibility."""
