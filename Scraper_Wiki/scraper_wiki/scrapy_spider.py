"""Scrapy spider for large scale Wikipedia crawling."""

from __future__ import annotations

import scrapy

from . import Config, get_base_url


class WikiSpider(scrapy.Spider):
    """Spider that recursively downloads Wikipedia pages."""

    name = "wiki_spider"

    def __init__(self, lang: str = "en", category: str | None = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lang = lang
        self.category = category
        if category:
            self.start_urls = [f"{get_base_url(lang)}/wiki/{category}"]
        else:
            self.start_urls = []

    custom_settings = {"USER_AGENT": Config.get_random_user_agent()}

    def parse(self, response: scrapy.http.Response):
        yield {"url": response.url, "html": response.text}
        for href in response.css("a::attr(href)").getall():
            if not href.startswith("/wiki/"):
                continue
            if ":" in href:
                continue
            url = response.urljoin(href)
            yield scrapy.Request(url, callback=self.parse)
