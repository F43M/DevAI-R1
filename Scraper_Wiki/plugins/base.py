"""Plugin interface definition and basic helpers."""

from typing import List, Dict, Protocol, runtime_checkable

import requests

from scraper_wiki import Config, metrics
from utils.rate_limiter import RateLimiter


@runtime_checkable
class Plugin(Protocol):
    """Basic scraping plugin."""

    def fetch_items(self, lang: str, category: str) -> List[Dict]:
        """Return raw items for a given language and category."""
        ...

    def parse_item(self, item: Dict) -> Dict:
        """Convert a raw item into a dataset entry."""
        ...


class BasePlugin:
    """Base class implementing rate limiting and deduplication."""

    def __init__(self, domain: str | None = None) -> None:
        self.domain = domain or self.__class__.__name__.lower()
        delay = Config.PLUGIN_RATE_LIMITS.get(
            self.domain,
            Config.PLUGIN_RATE_LIMITS.get("default", Config.RATE_LIMIT_DELAY),
        )
        self.rate_limiter = RateLimiter(delay, metrics=metrics)
        self._seen: set[str] = set()

    def request(self, url: str, **kwargs) -> requests.Response:
        """Perform a GET request respecting rate limits."""
        self.rate_limiter.wait()
        resp = requests.get(url, timeout=Config.TIMEOUT, **kwargs)
        resp.raise_for_status()
        return resp

    def deduplicate(self, items: List[Dict], key: str = "id") -> List[Dict]:
        """Remove items with repeated identifiers."""
        unique: List[Dict] = []
        for it in items:
            identifier = it.get(key)
            if identifier is None or identifier not in self._seen:
                if identifier is not None:
                    self._seen.add(identifier)
                unique.append(it)
        return unique
