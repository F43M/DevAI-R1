import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def test_codepen_scraper(monkeypatch):
    import importlib
    import requests

    core_stub = ModuleType("core")
    core_stub.builder = SimpleNamespace(DatasetBuilder=object)
    monkeypatch.setitem(sys.modules, "core", core_stub)
    monkeypatch.setitem(sys.modules, "core.builder", core_stub.builder)

    called = []

    class DummyResp:
        def __init__(self, text=""):
            self._text = text

        def raise_for_status(self):
            pass

        @property
        def text(self):
            return self._text

    def fake_get(url):
        called.append(url)
        if "codepen" in url:
            html = (
                '<script>window.__INITIAL_DATA__ = {"project": {"files": {"html": '
                '{"content": "<h1>Hello</h1>"}, "css": {"content": "h1{color:red;}"}, '
                '"js": {"content": "console.log(1);"}}}};</script>'
            )
            return DummyResp(html)
        html = (
            '<textarea id="id_html"><h1>Hi</h1></textarea>'
            '<textarea id="id_css">body{}</textarea>'
            '<textarea id="id_js">console.log(2);</textarea>'
        )
        return DummyResp(html)

    monkeypatch.setattr(requests, "get", fake_get)

    mod = importlib.import_module("plugins.codepen_scraper")
    scraper = mod.CodePenScraper()

    items = scraper.fetch_items("en", "https://codepen.io/u/pen/1")
    assert called[0].endswith("/pen/1")
    record = scraper.parse_item(items[0])
    assert record["url"] == "https://codepen.io/u/pen/1"
    assert record["language"] == "en"
    assert record["html"] == "<h1>Hello</h1>"
    assert record["css"] == "h1{color:red;}"
    assert record["js"] == "console.log(1);"
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

    items2 = scraper.fetch_items("en", "https://jsfiddle.net/test")
    assert called[1].endswith("/test")
    record2 = scraper.parse_item(items2[0])
    assert record2["url"] == "https://jsfiddle.net/test"
    assert record2["language"] == "en"
    assert record2["html"] == "<h1>Hi</h1>"
    assert record2["css"] == "body{}"
    assert record2["js"] == "console.log(2);"
    for field in [
        "raw_code",
        "context",
        "problems",
        "fixed_version",
        "lessons",
        "origin_metrics",
        "challenge",
    ]:
        assert field in record2
