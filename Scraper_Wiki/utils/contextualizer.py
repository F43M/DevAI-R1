"""Search external discussions for a code snippet."""

from __future__ import annotations

import os
from typing import List

import requests


def search_discussions(snippet_id: str) -> List[str]:
    """Return URLs of StackOverflow or GitHub threads mentioning ``snippet_id``."""
    from scraper_wiki import Config

    links: List[str] = []
    try:
        params = {
            "site": Config.STACKEXCHANGE_SITE,
            "order": "desc",
            "sort": "relevance",
            "q": snippet_id,
        }
        if Config.STACKEXCHANGE_API_KEY:
            params["key"] = Config.STACKEXCHANGE_API_KEY
        resp = requests.get(
            f"{Config.STACKEXCHANGE_API_ENDPOINT}/search/advanced",
            params=params,
            timeout=Config.TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        links.extend(it.get("link", "") for it in data.get("items", []))
    except Exception:
        pass
    try:
        headers = {"Accept": "application/vnd.github.v3+json"}
        token = os.environ.get("GITHUB_TOKEN")
        if token:
            headers["Authorization"] = f"token {token}"
        resp = requests.get(
            "https://api.github.com/search/issues",
            headers=headers,
            params={"q": snippet_id},
            timeout=Config.TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        links.extend(it.get("html_url", "") for it in data.get("items", []))
    except Exception:
        pass
    return [l for l in links if l]
