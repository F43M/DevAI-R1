"""Fetch programming tutorials from Medium, Dev.to and freeCodeCamp."""

from __future__ import annotations

import logging
from typing import Dict, List

import requests
from bs4 import BeautifulSoup
import html2text

from .base import Plugin

logger = logging.getLogger(__name__)


class TutorialsScraper(Plugin):  # type: ignore[misc]
    """Retrieve tutorial articles for dataset generation."""

    def fetch_items(self, lang: str, category: str) -> List[Dict]:
        """Return a single item with the tutorial HTML."""
        resp = requests.get(category, timeout=30)
        resp.raise_for_status()
        return [
            {
                "lang": lang,
                "category": category,
                "url": category,
                "html": resp.text,
            }
        ]

    def parse_item(self, item: Dict) -> Dict:
        """Extract plain text from tutorial HTML."""
        html = item.get("html", "")
        soup = BeautifulSoup(html, "html.parser")
        title = soup.title.string if soup.title else item.get("url", "")
        text = html2text.html2text(html) if hasattr(html2text, "html2text") else html
        record = {
            "title": title,
            "language": item.get("lang", "en"),
            "category": item.get("category", ""),
            "content": text,
        }
        record.setdefault("raw_code", "")
        record.setdefault("context", "")
        record.setdefault("problems", [])
        record.setdefault("fixed_version", "")
        record.setdefault("lessons", "")
        record.setdefault("origin_metrics", {})
        record.setdefault("challenge", "")
        return record


# Alias for registry
Plugin = TutorialsScraper
