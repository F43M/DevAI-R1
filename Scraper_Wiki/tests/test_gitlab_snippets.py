import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def test_gitlab_snippets(monkeypatch):
    import importlib
    import requests

    core_stub = ModuleType("core")
    core_stub.builder = SimpleNamespace(DatasetBuilder=object)
    monkeypatch.setitem(sys.modules, "core", core_stub)
    monkeypatch.setitem(sys.modules, "core.builder", core_stub.builder)

    called = []

    class DummyResp:
        def __init__(self, data=None, text=""):
            self._data = data
            self._text = text

        def raise_for_status(self):
            pass

        def json(self):
            return self._data

        @property
        def text(self):
            return self._text

    def fake_get(url, headers=None, params=None):
        called.append(url)
        if url.endswith("/snippets"):
            return DummyResp(data=[{"id": 1, "title": "t"}])
        if url.endswith("/snippets/1/raw"):
            return DummyResp(text="code1")
        return DummyResp()

    monkeypatch.setattr(requests, "get", fake_get)
    mod = importlib.import_module("plugins.gitlab_snippets")
    scraper = mod.GitLabSnippets()

    items = scraper.fetch_items("en", "example")
    assert called[0].endswith("/snippets")
    record = scraper.parse_item(items[0])
    assert record["title"] == "t"
    assert record["language"] == "en"
    assert record["category"] == "example"
    assert record["code"] == "code1"
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
