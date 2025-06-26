"""Wikidata scraping plugin."""

from typing import List, Dict

import requests

from scraper_wiki import Config
from .base import Plugin


class Plugin(Plugin):  # type: ignore[misc]
    """Query Wikidata items related to a category."""

    def __init__(self, endpoint: str | None = None) -> None:
        self.endpoint = endpoint or Config.WIKIDATA_API_ENDPOINT

    def fetch_items(self, lang: str, category: str) -> List[Dict]:
        params = {
            "action": "wbsearchentities",
            "search": category,
            "language": lang,
            "format": "json",
            "limit": 10,
        }
        resp = requests.get(self.endpoint, params=params, timeout=Config.TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        items = data.get("search", [])
        for it in items:
            it["lang"] = lang
            it["category"] = category
        return items

    def parse_item(self, item: Dict) -> Dict:
        qid = item.get("id")
        if not qid:
            return {}
        params = {
            "action": "wbgetentities",
            "ids": qid,
            "languages": item.get("lang", "en"),
            "format": "json",
        }
        resp = requests.get(self.endpoint, params=params, timeout=Config.TIMEOUT)
        resp.raise_for_status()
        data = resp.json().get("entities", {}).get(qid, {})
        labels = data.get("labels", {})
        descriptions = data.get("descriptions", {})
        claims = data.get("claims", {})
        image_name = None
        if isinstance(claims, dict) and "P18" in claims:
            image_claim = claims.get("P18", [{}])[0]
            image_name = (
                image_claim.get("mainsnak", {}).get("datavalue", {}).get("value")
            )
        image_url = (
            f"https://commons.wikimedia.org/wiki/Special:FilePath/{image_name}"
            if image_name
            else None
        )

        lang = item.get("lang", "en")
        record = {
            "title": labels.get(lang, {}).get("value", item.get("label", "")),
            "language": lang,
            "category": item.get("category", ""),
            "description": descriptions.get(lang, {}).get(
                "value", item.get("description", "")
            ),
            "wikidata_id": qid,
            "image_url": image_url,
        }
        record.setdefault("raw_code", "")
        record.setdefault("context", "")
        record.setdefault("problems", [])
        record.setdefault("fixed_version", "")
        record.setdefault("lessons", "")
        record.setdefault("origin_metrics", {})
        record.setdefault("challenge", "")
        return record
