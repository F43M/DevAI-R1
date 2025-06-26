"""Automated dataset scraper using Selenium."""

from __future__ import annotations

import logging
from typing import Dict, List

from bs4 import BeautifulSoup
from fake_useragent import UserAgent
from selenium.webdriver import Chrome, ChromeOptions
from selenium.webdriver.common.by import By
from typing import Optional

logger = logging.getLogger(__name__)


class AutoLearnerScraper:
    """Scraper that collects pages rendered dynamically."""

    def __init__(
        self,
        base_url: str,
        driver_path: str | None = None,
        headless: bool = True,
        backend: str = "selenium",
    ) -> None:
        """Initialize the scraper.

        Args:
            base_url: Base website to crawl.
            driver_path: Optional path to the Chrome driver binary.
            headless: Run the browser without UI.
            backend: ``"selenium"`` or ``"playwright"``.
        """

        self.base_url = base_url.rstrip("/")
        self._pw: Optional[object] = None
        if backend == "playwright":
            from playwright.sync_api import sync_playwright

            self._pw = sync_playwright().start()
            browser = self._pw.chromium.launch(headless=headless)
            self.driver = browser.new_page()
        else:
            options = ChromeOptions()
            if headless:
                options.add_argument("--headless")
            ua = UserAgent()
            options.add_argument(f"user-agent={ua.random}")
            self.driver = (
                Chrome(driver_path, options=options)
                if driver_path
                else Chrome(options=options)
            )

    def _get_page_source(self, url: str) -> str:
        if self._pw is not None:
            self.driver.goto(url)
            return self.driver.content()
        self.driver.get(url)
        return self.driver.page_source

    def search(self, query: str) -> List[str]:
        """Search the site and return result links.

        Args:
            query: Term to search.

        Returns:
            List of absolute URLs extracted from the results page.
        """
        url = f"{self.base_url}/search?q={query}"
        html = self._get_page_source(url)
        if self._pw is not None:
            soup = BeautifulSoup(html, "html.parser")
            elems = soup.select("a")
            get_href = lambda el: el.get("href")
        else:
            elems = self.driver.find_elements(By.CSS_SELECTOR, "a")
            get_href = lambda el: el.get_attribute("href")
        links = []
        for el in elems:
            href = get_href(el)
            if href and href.startswith(self.base_url):
                links.append(href)
        return links

    def fetch_page(self, url: str) -> Dict[str, str]:
        """Download and parse a single page.

        Args:
            url: Page URL to fetch.

        Returns:
            Record with ``title``, ``content`` and ``url`` keys.
        """
        html = self._get_page_source(url)
        soup = BeautifulSoup(html, "html.parser")
        text = soup.get_text(" ", strip=True)
        title = soup.title.string if soup.title else url
        return {"title": title, "content": text, "url": url}

    def build_dataset(
        self, queries: List[str], max_pages: int = 5
    ) -> List[Dict[str, str]]:
        """Generate a dataset from multiple queries.

        Args:
            queries: Search queries to run.
            max_pages: Limit of pages fetched per query.

        Returns:
            List of parsed page records.
        """
        records = []
        for q in queries:
            for link in self.search(q)[:max_pages]:
                try:
                    records.append(self.fetch_page(link))
                except Exception as exc:  # pragma: no cover - network errors
                    logger.error("Failed to process %s: %s", link, exc)
        return records

    def close(self) -> None:
        """Close the browser driver."""
        if self._pw is not None:
            self._pw.stop()
        else:
            self.driver.quit()


__all__ = ["AutoLearnerScraper"]
