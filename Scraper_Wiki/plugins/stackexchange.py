"""Stack Exchange scraping plugin."""

from typing import List, Dict
import logging

import html2text
import requests

from scraper_wiki import Config, advanced_clean_text, log_error
from .base import Plugin


logger = logging.getLogger(__name__)


class StackExchangePlugin(Plugin):
    """Fetch questions from a Stack Exchange site by tag."""

    def __init__(
        self,
        site: str | None = None,
        api_key: str | None = None,
        endpoint: str | None = None,
        min_score: int | None = None,
    ) -> None:
        self.site = site or Config.STACKEXCHANGE_SITE
        self.api_key = api_key or Config.STACKEXCHANGE_API_KEY
        self.endpoint = (endpoint or Config.STACKEXCHANGE_API_ENDPOINT).rstrip("/")
        self.min_score = (
            Config.STACKEXCHANGE_MIN_SCORE if min_score is None else min_score
        )

    def fetch_items(
        self, lang: str, category: str, since: str | None = None
    ) -> List[Dict]:
        params = {
            "site": self.site,
            "tagged": category,
            "pagesize": 10,
            "order": "desc",
            "sort": "votes",
            "filter": "withbody",
        }
        if self.api_key:
            params["key"] = self.api_key
        if since:
            try:
                from datetime import datetime

                ts = int(datetime.fromisoformat(since).timestamp())
                params["fromdate"] = ts
            except (ValueError, TypeError) as exc:
                logger.warning(
                    "Invalid 'since' parameter",
                    extra={"error_type": type(exc).__name__, "error_message": str(exc)},
                )
        resp = requests.get(
            f"{self.endpoint}/questions", params=params, timeout=Config.TIMEOUT
        )
        resp.raise_for_status()
        data = resp.json()
        items = data.get("items", [])
        filtered = [it for it in items if it.get("score", 0) >= self.min_score]
        for it in filtered:
            it["lang"] = lang
            it["category"] = category
        return filtered

    def parse_item(self, item: Dict) -> Dict:
        body = item.get("body", "")
        text = html2text.html2text(body) if hasattr(html2text, "html2text") else body
        clean = advanced_clean_text(text, item.get("lang", "en"))
        tags = item.get("tags", [])
        tag_data = [{"tag": t, "link": item.get("link", "")} for t in tags]
        record = {
            "title": item.get("title", ""),
            "language": item.get("lang", "en"),
            "category": item.get("category", ""),
            "score": item.get("score", 0),
            "link": item.get("link", ""),
            "content": clean,
            "tags": tag_data,
            "context": item.get("title", ""),
            "tests": [],
            "docstring": "",
            "quality_score": item.get("score", 0),
        }
        record.setdefault("raw_code", "")
        record.setdefault("problems", [])
        record.setdefault("fixed_version", "")
        record.setdefault("lessons", "")
        record.setdefault("origin_metrics", {})
        record.setdefault("challenge", "")
        record.setdefault("discussion_links", [item.get("link", "")])
        return record


# Backwards compatible alias
Plugin = StackExchangePlugin
