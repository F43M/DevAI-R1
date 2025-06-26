"""Wikipedia scraping plugin."""
from core import WikipediaAdvanced, DatasetBuilder
from .base import Plugin


class Plugin(Plugin):  # type: ignore[misc]
    def __init__(self) -> None:
        self.builder = DatasetBuilder()

    def fetch_items(self, lang: str, category: str):
        wiki = WikipediaAdvanced(lang)
        return wiki.get_category_members(category)

    def parse_item(self, item: dict):
        return self.builder.process_page(item)

