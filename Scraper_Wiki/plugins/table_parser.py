"""Plugin to extract table data from Wikipedia pages."""

from typing import List, Dict

from bs4 import BeautifulSoup

from core import WikipediaAdvanced
from scraper_wiki import fetch_html_content, extract_tables

from .base import Plugin


class Plugin(Plugin):  # type: ignore[misc]
    """Parse all HTML tables on a page."""

    def fetch_items(self, lang: str, category: str) -> List[Dict]:
        wiki = WikipediaAdvanced(lang)
        return wiki.get_category_members(category)

    def parse_item(self, item: Dict) -> Dict:
        html = fetch_html_content(item["title"], item.get("lang", "en"))
        if not html:
            return {}
        tables = extract_tables(html)
        if not tables:
            return {}
        record = {
            "title": item["title"],
            "language": item.get("lang", "en"),
            "tables": tables,
        }
        record.setdefault("raw_code", "")
        record.setdefault("context", "")
        record.setdefault("problems", [])
        record.setdefault("fixed_version", "")
        record.setdefault("lessons", "")
        record.setdefault("origin_metrics", {})
        record.setdefault("challenge", "")
        return record
