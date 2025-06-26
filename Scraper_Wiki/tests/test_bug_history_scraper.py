import importlib
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def test_bug_history_scraper(monkeypatch):
    import requests

    core_stub = ModuleType("core")
    core_stub.builder = SimpleNamespace(DatasetBuilder=object)
    monkeypatch.setitem(sys.modules, "core", core_stub)
    monkeypatch.setitem(sys.modules, "core.builder", core_stub.builder)

    class DummyResp:
        def __init__(self, data=None):
            self._data = data

        def raise_for_status(self):
            pass

        def json(self):
            return self._data

    def fake_get(url, headers=None, params=None):
        if url.endswith("/commits"):
            return DummyResp(data=[{"sha": "abc", "commit": {"message": "Fix"}}])
        return DummyResp(data={})

    monkeypatch.setattr(requests, "get", fake_get)

    mod = importlib.import_module("plugins.bug_history_scraper")
    scraper = mod.BugHistoryScraper()

    items = scraper.fetch_items("en", "https://github.com/user/repo")
    assert items[0]["sha"] == "abc"
    record = scraper.parse_item(items[0])
    assert record["commit"] == "abc"
    assert record["message"] == "Fix"
    assert record["language"] == "en"
    assert record["category"] == "https://github.com/user/repo"
    for field in [
        "raw_code",
        "context",
        "problems",
        "fixed_version",
        "lessons",
        "origin_metrics",
        "challenge",
    ]:
        assert field in record
