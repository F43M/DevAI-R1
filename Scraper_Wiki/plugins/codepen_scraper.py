"""Scrape HTML, CSS and JS from CodePen or JSFiddle."""

from __future__ import annotations

import json
import re
from typing import Dict, List
import logging

import requests
from scraper_wiki import log_error

from .base import Plugin


logger = logging.getLogger(__name__)


class CodePenScraper(Plugin):
    """Fetch code snippets from CodePen or JSFiddle URLs."""

    def fetch_items(self, lang: str, category: str) -> List[Dict]:
        """Return raw HTML for the provided URL."""
        resp = requests.get(category)
        resp.raise_for_status()
        return [{"url": category, "content": resp.text, "lang": lang}]

    def parse_item(self, item: Dict) -> Dict:
        """Return extracted HTML, CSS and JS."""
        content = item.get("content", "")
        html = ""
        css = ""
        js = ""

        # Try CodePen JSON data
        match = re.search(r"__INITIAL_DATA__\s*=\s*(\{.*\})", content, re.S)
        if match:
            try:
                data = json.loads(match.group(1))
                files = data.get("project", {}).get("files", {})
                html = files.get("html", {}).get("content", "")
                css = files.get("css", {}).get("content", "")
                js = files.get("js", {}).get("content", "")
            except json.JSONDecodeError as exc:
                logger.warning(
                    "Failed to parse CodePen JSON",
                    extra={"error_type": type(exc).__name__, "error_message": str(exc)},
                )

        # Fallback for JSFiddle markup
        if not any([html, css, js]):
            html_match = re.search(
                r"<textarea[^>]*id=\"(?:html|id_html)\"[^>]*>(.*?)</textarea>",
                content,
                re.S,
            )
            css_match = re.search(
                r"<textarea[^>]*id=\"(?:css|id_css)\"[^>]*>(.*?)</textarea>",
                content,
                re.S,
            )
            js_match = re.search(
                r"<textarea[^>]*id=\"(?:js|id_js)\"[^>]*>(.*?)</textarea>",
                content,
                re.S,
            )
            if html_match:
                html = html_match.group(1)
            if css_match:
                css = css_match.group(1)
            if js_match:
                js = js_match.group(1)

        record = {
            "url": item.get("url", ""),
            "language": item.get("lang", "en"),
            "html": html,
            "css": css,
            "js": js,
        }
        record.setdefault("raw_code", "")
        record.setdefault("context", "")
        record.setdefault("problems", [])
        record.setdefault("fixed_version", "")
        record.setdefault("lessons", "")
        record.setdefault("origin_metrics", {})
        record.setdefault("challenge", "")
        return record


Plugin = CodePenScraper
