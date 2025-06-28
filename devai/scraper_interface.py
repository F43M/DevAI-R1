"""Wrapper utilities for the Scraper Wiki project."""

from __future__ import annotations

import asyncio
import os
from typing import Optional, Any


def _run_sync(func: Any, *args: Any, **kwargs: Any) -> Any:
    """Execute a blocking function in a thread."""
    return asyncio.to_thread(func, *args, **kwargs)


async def run_scrape(
    topic: str, lang: Optional[str] = None, depth: int = 1, **opts: Any
) -> Any:
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

    distributed = bool(opts.get("distributed"))
    client = opts.get("client")
    if distributed and client is None:
        try:
            from .config import config
            from Scraper_Wiki import cluster

            cfg = cluster.load_config(config.SCRAPER_CLUSTER_CONFIG)
            client = cluster.get_client(cfg)
        except Exception as exc:  # pragma: no cover - optional cluster
            import logging

            logging.getLogger(__name__).error("cluster_unavailable", error=str(exc))
            client = None

    if plugin == "github":
        from Scraper_Wiki.cli import auto_scrape

        threads = int(opts.get("threads", 2))
        return await _run_sync(auto_scrape, [topic], depth=depth, threads=threads)

    import scraper_wiki

    scraper_wiki.metrics.start_metrics_server(
        int(os.environ.get("METRICS_PORT", "8001"))
    )
    from .data_ingestion import ingest_directory
    from .memory import MemoryManager

    fmt = opts.get("format", "all")
    rate_delay = opts.get("rate_limit_delay")
    revisions = bool(opts.get("revisions", False))
    rev_limit = int(opts.get("rev_limit", 5))
    translate_to = opts.get("translate_to")
    incremental = bool(opts.get("incremental", False))

    langs = [lang] if lang else None
    categories = [topic]

    result = await _run_sync(
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
        client=client,
        incremental=incremental,
    )

    try:
        mem = opts.get("memory")
        if isinstance(mem, MemoryManager):
            ingest_directory(mem, scraper_wiki.Config.OUTPUT_DIR)
    except Exception as exc:  # pragma: no cover - ingestion failures
        import logging

        logging.getLogger(__name__).error("ingestion_failed", error=str(exc))

    return result


__all__ = ["run_scrape"]
