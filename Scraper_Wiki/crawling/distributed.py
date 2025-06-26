"""Distributed crawling utilities using Dask or Ray."""

from __future__ import annotations

import logging
import time
from typing import Iterable, List
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import requests

from cluster import get_client, load_config
from . import rate_limiter

logger = logging.getLogger(__name__)


class DistributedCrawler:
    """Simple crawler that distributes fetch tasks across a cluster."""

    def __init__(
        self,
        start_urls: Iterable[str],
        client=None,
        crawl_delay: float = 1.0,
        user_agent: str = "ScraperWikiBot",
        max_pages: int = 100,
    ) -> None:
        self.start_urls = list(start_urls)
        self.client = client
        self.crawl_delay = crawl_delay
        self.user_agent = user_agent
        self.max_pages = max_pages
        self.robot_cache: dict[str, RobotFileParser] = {}

    # robots.txt utilities
    def _robot_parser(self, base_url: str) -> RobotFileParser:
        rp = self.robot_cache.get(base_url)
        if rp:
            return rp
        rp = RobotFileParser()
        rp.set_url(urljoin(base_url, "robots.txt"))
        try:
            rp.read()
        except Exception as exc:  # pragma: no cover - network errors
            logger.warning("Failed to read robots.txt from %s: %s", base_url, exc)
        self.robot_cache[base_url] = rp
        return rp

    def can_fetch(self, url: str) -> bool:
        parsed = urlparse(url)
        base = f"{parsed.scheme}://{parsed.netloc}/"
        rp = self._robot_parser(base)
        return rp.can_fetch(self.user_agent, url)

    def fetch(self, url: str) -> str | None:
        if not self.can_fetch(url):
            logger.info("Blocked by robots.txt: %s", url)
            return None
        host = urlparse(url).netloc
        limiter = rate_limiter.get(host)
        limiter.wait()
        try:
            resp = requests.get(
                url, headers={"User-Agent": self.user_agent}, timeout=10
            )
            resp.raise_for_status()
            limiter.record_success()
            return resp.text
        except Exception as exc:  # pragma: no cover - network errors
            limiter.record_error()
            logger.error("Failed to fetch %s: %s", url, exc)
            return None

    def extract_links(self, html: str, base_url: str) -> List[str]:
        links: List[str] = []
        try:
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(html, "html.parser")
            for tag in soup.find_all("a", href=True):
                links.append(urljoin(base_url, tag["href"]))
        except Exception as exc:  # pragma: no cover - parsing errors
            logger.error("Failed to parse %s: %s", base_url, exc)
        return links

    def crawl_page(self, url: str) -> List[str]:
        html = self.fetch(url)
        if not html:
            return []
        return self.extract_links(html, url)

    def crawl(self) -> List[str]:
        visited: set[str] = set()
        queue: List[str] = list(self.start_urls)
        futures = []
        results: List[str] = []
        while queue and len(visited) < self.max_pages:
            url = queue.pop(0)
            if url in visited:
                continue
            visited.add(url)
            if self.client:
                futures.append(self.client.submit(self.crawl_page, url))
            else:
                links = self.crawl_page(url)
                for link in links:
                    if (
                        link not in visited
                        and len(visited) + len(queue) < self.max_pages
                    ):
                        queue.append(link)
                results.append(url)
        for fut in futures:
            links = fut.result()
            for link in links:
                if link not in visited and len(visited) + len(queue) < self.max_pages:
                    queue.append(link)
        results.extend(list(visited))
        return results


def start_crawler(config_path: str | None = None) -> None:
    """Start the distributed crawler using settings from ``cluster.yaml``."""

    cfg = load_config(config_path)
    crawler_cfg = cfg.get("crawler", {})
    client = get_client(cfg)
    start_urls = crawler_cfg.get("start_urls", [])
    delay = float(crawler_cfg.get("crawl_delay", 1.0))
    max_pages = int(crawler_cfg.get("max_pages", 100))
    user_agent = crawler_cfg.get("user_agent", "ScraperWikiBot")
    crawler = DistributedCrawler(
        start_urls,
        client=client,
        crawl_delay=delay,
        user_agent=user_agent,
        max_pages=max_pages,
    )
    crawler.crawl()


def benchmark_crawler(
    start_urls: Iterable[str],
    total_pages: int = 1000000,
    client=None,
    crawl_delay: float = 0.0,
    user_agent: str = "ScraperWikiBot",
) -> float:
    """Benchmark crawling a large number of pages.

    Args:
        start_urls: Initial URLs used as seeds.
        total_pages: Total number of pages to collect.
        client: Optional distributed client.
        crawl_delay: Base delay between requests.
        user_agent: HTTP user agent string.

    Returns:
        Total time in seconds to fetch ``total_pages`` pages.
    """
    crawler = DistributedCrawler(
        start_urls,
        client=client,
        crawl_delay=crawl_delay,
        user_agent=user_agent,
        max_pages=total_pages,
    )
    start = time.time()
    crawler.crawl()
    elapsed = time.time() - start
    logger.info(
        "Benchmark fetched %d pages in %.2fs (%.2f pages/s)",
        total_pages,
        elapsed,
        total_pages / elapsed if elapsed else 0,
    )
    return elapsed


def run_benchmark(config_path: str | None = None) -> None:
    """Execute ``benchmark_crawler`` using ``cluster.yaml`` settings."""
    cfg = load_config(config_path)
    crawler_cfg = cfg.get("crawler", {})
    client = get_client(cfg)
    start_urls = crawler_cfg.get("start_urls", [])
    pages = int(crawler_cfg.get("benchmark_pages", 1000000))
    delay = float(crawler_cfg.get("crawl_delay", 0.0))
    benchmark_crawler(
        start_urls,
        total_pages=pages,
        client=client,
        crawl_delay=delay,
        user_agent=crawler_cfg.get("user_agent", "ScraperWikiBot"),
    )


def stop_crawler() -> None:
    """Placeholder to stop the crawler."""

    logger.info("Crawler stopped")
