"""Plugin utilities and registry."""

import importlib
from typing import List, Optional
from datetime import datetime

from core.builder import DatasetBuilder
from .base import Plugin
from scraper_wiki.models import DatasetRecord
from pydantic import ValidationError
from integrations import storage_sqlite

# Mapping of available plugin names to module paths
AVAILABLE_PLUGINS = {
    "wikipedia": "wikipedia",
    "wikidata": "wikidata",
    "stackexchange": "stackexchange",
    "stackoverflow": "stackoverflow",
    "infobox_parser": "infobox_parser",
    "table_parser": "table_parser",
    "github_scraper": "github_scraper",
    "code_extractor": "code_extractor",
    "gitlab_scraper": "gitlab_scraper",
    "gitlab_snippets": "gitlab_snippets",
    "gist_scraper": "gist_scraper",
    "api_docs": "api_docs",
    "competitions": "competitions",
    "codepen_scraper": "codepen_scraper",
    "legacy_forums": "legacy_forums",
    "pdf_books": "pdf_books",
    "bug_history_scraper": "bug_history_scraper",
    "mdn_docs": "mdn_docs",
    "devdocs": "devdocs",
    "university_courses": "university_courses",
    "rfc_scraper": "rfc_scraper",
    "npm_packages": "npm_packages",
    "pip_packages": "pip_packages",
    "cran_packages": "cran_packages",
    "jira_scraper": "jira_scraper",
    "bugzilla_scraper": "bugzilla_scraper",
    "leetcode": "leetcode",
    "hackerrank": "hackerrank",
    "codeforces": "codeforces",
    "kaggle": "kaggle",
    "bitbucket_scraper": "bitbucket_scraper",
    "sourceforge_scraper": "sourceforge_scraper",
    "notebooks": "notebooks",
    "reddit": "reddit",
    "twitter": "twitter",
    "youtube_transcripts": "youtube_transcripts",
    "python_docs": "python_docs",
}


def load_plugin(name: str) -> Plugin:
    """Load a plugin by its registry name."""
    module_name = AVAILABLE_PLUGINS.get(name, name)
    module = importlib.import_module(f"plugins.{module_name}")
    plugin_cls = getattr(module, "Plugin")
    return plugin_cls()


def run_plugin(
    plugin: Plugin,
    langs: List[str],
    categories: List[str],
    fmt: str = "all",
    incremental: bool = False,
) -> List[dict]:
    """Execute scraping using a plugin."""
    builder = DatasetBuilder()
    for lang in langs:
        for category in categories:
            key = f"{plugin.__class__.__name__}:{lang}:{category}"
            since = storage_sqlite.get_last_processed(key) if incremental else None
            if since:
                items = plugin.fetch_items(lang, category, since=since)
            else:
                items = plugin.fetch_items(lang, category)
            latest: Optional[str] = None
            for item in items:
                result = plugin.parse_item(item)
                if result:
                    record = DatasetRecord.parse_obj(result)
                    builder.dataset.append(record.dict())
                ts = (
                    item.get("created_at")
                    or item.get("creation_date")
                    or item.get("commit", {}).get("committer", {}).get("date")
                )
                if ts is not None:
                    if isinstance(ts, (int, float)):
                        ts = datetime.utcfromtimestamp(int(ts)).isoformat()
                    latest = ts if latest is None or ts > latest else latest
            if incremental and latest:
                storage_sqlite.set_last_processed(key, str(latest))
    builder.save_dataset(fmt, incremental=incremental)
    return builder.dataset


__all__ = ["load_plugin", "run_plugin", "AVAILABLE_PLUGINS"]
