"""Documentation scraping plugin."""

from __future__ import annotations

from typing import List, Dict
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from scraper_wiki import Config
from .base import Plugin


class APIDocsScraper(Plugin):  # type: ignore[misc]
    """Crawl documentation pages and extract code blocks."""

    def __init__(self, allowed_domains: List[str] | None = None) -> None:
        self.allowed_domains = allowed_domains or [
            "docs.python.org",
            "developer.mozilla.org",
        ]

    def fetch_items(self, lang: str, category: str) -> List[Dict]:
        """Return pages linked from the base URL."""
        url = category
        resp = requests.get(url, timeout=Config.TIMEOUT)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        items = [{"url": url, "lang": lang, "category": category}]
        for a in soup.select("a[href]"):
            href = a["href"]
            abs_url = urljoin(url, href)
            if urlparse(abs_url).netloc in self.allowed_domains:
                items.append({"url": abs_url, "lang": lang, "category": category})
        return items

    def parse_item(self, item: Dict) -> Dict:
        """Extract code blocks and surrounding text."""
        resp = requests.get(item["url"], timeout=Config.TIMEOUT)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        blocks = []
        for pre in soup.find_all("pre"):
            code = pre.get_text("\n")
            explanation = ""
            if pre.previous_sibling and getattr(pre.previous_sibling, "get_text", None):
                explanation = pre.previous_sibling.get_text(" ", strip=True)
            blocks.append({"code": code, "text": explanation})
        record = {
            "url": item["url"],
            "language": item.get("lang", "en"),
            "blocks": blocks,
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
Plugin = APIDocsScraper
