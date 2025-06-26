import importlib
import sys
from pathlib import Path
from types import SimpleNamespace, ModuleType

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Minimal configuration stub
scraper_stub = ModuleType("scraper_wiki")
scraper_stub.Config = SimpleNamespace(TIMEOUT=5)
sys.modules.setdefault("scraper_wiki", scraper_stub)


def _check_defaults(record):
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


def test_leetcode_plugin(monkeypatch):
    import requests

    core_stub = ModuleType("core")
    core_stub.builder = SimpleNamespace(DatasetBuilder=object)
    monkeypatch.setitem(sys.modules, "core", core_stub)
    monkeypatch.setitem(sys.modules, "core.builder", core_stub.builder)

    def fake_get(url, timeout=None):
        if "problems" in url:
            return DummyResp(data={"title": "Two Sum", "content": "P", "solution": "S"})
        return DummyResp(data={"content": "D"})

    monkeypatch.setattr(requests, "get", fake_get)
    mod = importlib.import_module("plugins.leetcode")
    plugin = mod.Plugin()
    items = plugin.fetch_items("en", "two-sum")
    assert items[0]["slug"] == "two-sum"
    parsed = plugin.parse_item(items[0])
    assert parsed["problem"] == "P"
    assert parsed["solution"] == "S"
    assert parsed["discussion"] == "D"
    _check_defaults(parsed)


def test_hackerrank_plugin(monkeypatch):
    import requests

    core_stub = ModuleType("core")
    core_stub.builder = SimpleNamespace(DatasetBuilder=object)
    monkeypatch.setitem(sys.modules, "core", core_stub)
    monkeypatch.setitem(sys.modules, "core.builder", core_stub.builder)

    def fake_get(url, timeout=None):
        if url.endswith("/discussions"):
            return DummyResp(data={"content": "D"})
        return DummyResp(data={"name": "Ch", "body": "P", "solution": "S"})

    monkeypatch.setattr(requests, "get", fake_get)
    mod = importlib.import_module("plugins.hackerrank")
    plugin = mod.Plugin()
    items = plugin.fetch_items("en", "challenge")
    assert items[0]["slug"] == "challenge"
    parsed = plugin.parse_item(items[0])
    assert parsed["problem"] == "P"
    assert parsed["solution"] == "S"
    assert parsed["discussion"] == "D"
    _check_defaults(parsed)


def test_codeforces_plugin(monkeypatch):
    import requests

    core_stub = ModuleType("core")
    core_stub.builder = SimpleNamespace(DatasetBuilder=object)
    monkeypatch.setitem(sys.modules, "core", core_stub)
    monkeypatch.setitem(sys.modules, "core.builder", core_stub.builder)

    def fake_get(url, timeout=None):
        if "discussion" in url:
            return DummyResp(data={"content": "D"})
        return DummyResp(data={"name": "Prob", "statement": "P", "solution": "S"})

    monkeypatch.setattr(requests, "get", fake_get)
    mod = importlib.import_module("plugins.codeforces")
    plugin = mod.Plugin()
    items = plugin.fetch_items("en", "1234/A")
    assert items[0]["slug"] == "1234/A"
    parsed = plugin.parse_item(items[0])
    assert parsed["problem"] == "P"
    assert parsed["solution"] == "S"
    assert parsed["discussion"] == "D"
    _check_defaults(parsed)


def test_kaggle_plugin(monkeypatch):
    import requests

    core_stub = ModuleType("core")
    core_stub.builder = SimpleNamespace(DatasetBuilder=object)
    monkeypatch.setitem(sys.modules, "core", core_stub)
    monkeypatch.setitem(sys.modules, "core.builder", core_stub.builder)

    def fake_get(url, timeout=None):
        if url.endswith("/topics"):
            return DummyResp(data={"content": "D"})
        return DummyResp(data={"title": "Comp", "description": "P", "solution": "S"})

    monkeypatch.setattr(requests, "get", fake_get)
    mod = importlib.import_module("plugins.kaggle")
    plugin = mod.Plugin()
    items = plugin.fetch_items("en", "competition")
    assert items[0]["slug"] == "competition"
    parsed = plugin.parse_item(items[0])
    assert parsed["problem"] == "P"
    assert parsed["solution"] == "S"
    assert parsed["discussion"] == "D"
    _check_defaults(parsed)


def test_bitbucket_scraper(monkeypatch):
    import requests

    core_stub = ModuleType("core")
    core_stub.builder = SimpleNamespace(DatasetBuilder=object)
    monkeypatch.setitem(sys.modules, "core", core_stub)
    monkeypatch.setitem(sys.modules, "core.builder", core_stub.builder)

    def fake_get(url, timeout=None):
        if url.endswith("README.md"):
            return DummyResp(text="readme")
        if url.endswith("/issues"):
            return DummyResp(data=[{"title": "bug"}])
        return DummyResp()

    monkeypatch.setattr(requests, "get", fake_get)
    mod = importlib.import_module("plugins.bitbucket_scraper")
    scraper = mod.Plugin()
    record = scraper.build_dataset_record("https://bitbucket.org/u/repo")
    assert record["readme"] == "readme"
    assert record["issues"] == [{"title": "bug"}]
    _check_defaults(record)


def test_sourceforge_scraper(monkeypatch):
    import requests

    core_stub = ModuleType("core")
    core_stub.builder = SimpleNamespace(DatasetBuilder=object)
    monkeypatch.setitem(sys.modules, "core", core_stub)
    monkeypatch.setitem(sys.modules, "core.builder", core_stub.builder)

    def fake_get(url, timeout=None):
        return DummyResp(data={"name": "proj", "summary": "S", "description": "D"})

    monkeypatch.setattr(requests, "get", fake_get)
    mod = importlib.import_module("plugins.sourceforge_scraper")
    plugin = mod.Plugin()
    items = plugin.fetch_items("en", "proj")
    assert items[0]["name"] == "proj"
    parsed = plugin.parse_item(items[0])
    assert parsed["summary"] == "S"
    assert parsed["description"] == "D"
    _check_defaults(parsed)


def test_notebooks_plugin(monkeypatch):
    import requests

    core_stub = ModuleType("core")
    core_stub.builder = SimpleNamespace(DatasetBuilder=object)
    monkeypatch.setitem(sys.modules, "core", core_stub)
    monkeypatch.setitem(sys.modules, "core.builder", core_stub.builder)

    def fake_get(url, timeout=None):
        return DummyResp(text='{"nb":1}')

    monkeypatch.setattr(requests, "get", fake_get)
    mod = importlib.import_module("plugins.notebooks")
    plugin = mod.Plugin()
    items = plugin.fetch_items("en", "http://example.com/nb.ipynb")
    parsed = plugin.parse_item(items[0])
    assert parsed["notebook"] == '{"nb":1}'
    _check_defaults(parsed)


def test_tutorials_scraper(monkeypatch):
    import requests

    core_stub = ModuleType("core")
    core_stub.builder = SimpleNamespace(DatasetBuilder=object)
    monkeypatch.setitem(sys.modules, "core", core_stub)
    monkeypatch.setitem(sys.modules, "core.builder", core_stub.builder)

    monkeypatch.setattr(
        requests,
        "get",
        lambda *a, **k: DummyResp(text="<html><title>T</title><p>A</p></html>"),
    )
    mod = importlib.import_module("plugins.tutorials_scraper")
    plugin = mod.Plugin()
    items = plugin.fetch_items("en", "https://example.com")
    parsed = plugin.parse_item(items[0])
    assert parsed["title"] == "T"
    _check_defaults(parsed)


def test_code_review_scraper(monkeypatch):
    import requests

    core_stub = ModuleType("core")
    core_stub.builder = SimpleNamespace(DatasetBuilder=object)
    monkeypatch.setitem(sys.modules, "core", core_stub)
    monkeypatch.setitem(sys.modules, "core.builder", core_stub.builder)

    def fake_get(url, headers=None, params=None, timeout=None):
        if url.endswith("/pulls"):
            return DummyResp(data=[{"number": 1, "title": "PR", "html_url": "u"}])
        if url.endswith("/pulls/1"):
            return DummyResp(text="diff")
        if url.endswith("/issues/1/comments"):
            return DummyResp(data=[{"body": "c"}])
        return DummyResp()

    monkeypatch.setattr(requests, "get", fake_get)
    mod = importlib.import_module("plugins.code_review_scraper")
    plugin = mod.Plugin()
    items = plugin.fetch_items("en", "https://github.com/u/r")
    parsed = plugin.parse_item(items[0])
    assert parsed["diff"] == "diff"
    assert parsed["comments"] == ["c"]
    _check_defaults(parsed)
