"""Wrapper utilities for the Scraper Wiki project."""

from __future__ import annotations

import asyncio
from typing import Optional, Any


def _run_sync(func: Any, *args: Any, **kwargs: Any) -> Any:
    """Execute a blocking function in a thread."""
    return asyncio.to_thread(func, *args, **kwargs)


async def run_scrape(topic: str, lang: Optional[str] = None, depth: int = 1, **opts: Any) -> Any:
    """Run Scraper Wiki to gather data about a topic.

    Parameters
    ----------
    topic:
        Topic, category or URL to scrape.
    lang:
        Optional language code.
    depth:
        Crawling depth when using dynamic scraping.
    opts:
        Extra options like ``plugin`` or ``threads``.
    """

    plugin = opts.get("plugin", "wikipedia")

    if plugin == "github":
        from Scraper_Wiki.cli import auto_scrape

        threads = int(opts.get("threads", 2))
        return await _run_sync(auto_scrape, [topic], depth=depth, threads=threads)

    import scraper_wiki

    fmt = opts.get("format", "all")
    rate_delay = opts.get("rate_limit_delay")
    revisions = bool(opts.get("revisions", False))
    rev_limit = int(opts.get("rev_limit", 5))
    translate_to = opts.get("translate_to")
    incremental = bool(opts.get("incremental", False))

    langs = [lang] if lang else None
    categories = [topic]

    return await _run_sync(
        scraper_wiki.main,
        langs,
        categories,
        fmt,
        rate_delay,
        start_pages=None,
        depth=depth,
        revisions=revisions,
        rev_limit=rev_limit,
        translate_to=translate_to,
        incremental=incremental,
    )

__all__ = ["run_scrape"]
