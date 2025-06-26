"""Reddit scraping plugin using PRAW."""

from __future__ import annotations

from typing import Dict, List

import praw

from .base import BasePlugin


class RedditPlugin(BasePlugin):
    """Fetch posts from popular subreddits."""

    def __init__(
        self,
        client_id: str | None = None,
        client_secret: str | None = None,
        user_agent: str | None = None,
        subreddits: List[str] | None = None,
    ) -> None:
        super().__init__()
        self.reddit = praw.Reddit(
            client_id=client_id or "",
            client_secret=client_secret or "",
            user_agent=user_agent or "scraper-wiki",
        )
        self.subreddits = subreddits or ["programming", "learnpython"]

    def fetch_items(self, lang: str, category: str) -> List[Dict]:
        """Return recent posts for configured subreddits."""
        items: List[Dict] = []
        for sub in self.subreddits:
            for post in self.reddit.subreddit(sub).hot(limit=10):
                it = {
                    "id": post.id,
                    "title": post.title,
                    "selftext": getattr(post, "selftext", ""),
                    "url": post.url,
                    "subreddit": sub,
                    "lang": lang,
                    "category": category,
                }
                items.append(it)
        return self.deduplicate(items)

    def parse_item(self, item: Dict) -> Dict:
        """Convert a Reddit submission into a dataset record."""
        record = {
            "title": item.get("title", ""),
            "language": item.get("lang", "en"),
            "category": item.get("category", ""),
            "content": item.get("selftext", ""),
            "url": item.get("url", ""),
            "subreddit": item.get("subreddit", ""),
        }
        record.setdefault("raw_code", "")
        record.setdefault("context", "")
        record.setdefault("problems", [])
        record.setdefault("fixed_version", "")
        record.setdefault("lessons", "")
        record.setdefault("origin_metrics", {})
        record.setdefault("challenge", "")
        return record


Plugin = RedditPlugin
