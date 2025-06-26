"""Plugin fetching YouTube video transcripts."""

from __future__ import annotations

from typing import Dict, List
import logging
import requests

from youtube_transcript_api import YouTubeTranscriptApi
from scraper_wiki import binary_storage, log_error

from .base import BasePlugin


logger = logging.getLogger(__name__)


class YouTubeTranscriptPlugin(BasePlugin):
    """Retrieve transcripts for YouTube videos."""

    def fetch_items(self, lang: str, category: str) -> List[Dict]:
        """Return descriptor for the given video ID or URL."""
        video_id = category.split("v=")[-1].split("/")[-1]
        return [{"video_id": video_id, "lang": lang, "category": category}]

    def parse_item(self, item: Dict) -> Dict:
        """Download and join transcript lines."""
        vid = item.get("video_id")
        if not vid:
            return {}
        transcript = YouTubeTranscriptApi.get_transcript(vid)
        text = " ".join(seg.get("text", "") for seg in transcript)
        record = {
            "video_id": vid,
            "language": item.get("lang", "en"),
            "category": item.get("category", ""),
            "content": text,
            "video_urls": [f"https://www.youtube.com/watch?v={vid}"],
        }
        thumb_url = f"https://img.youtube.com/vi/{vid}/hqdefault.jpg"
        try:
            path = binary_storage.save(thumb_url)
            record["image_paths"] = [path]
        except (requests.RequestException, OSError) as exc:
            log_error("Failed to download thumbnail", exc)
            record["image_paths"] = []
        record["thumbnail_url"] = thumb_url
        record.setdefault("raw_code", "")
        record.setdefault("context", "")
        record.setdefault("problems", [])
        record.setdefault("fixed_version", "")
        record.setdefault("lessons", "")
        record.setdefault("origin_metrics", {})
        record.setdefault("challenge", "")
        return record


Plugin = YouTubeTranscriptPlugin
