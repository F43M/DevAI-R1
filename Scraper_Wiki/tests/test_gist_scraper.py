import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def test_gist_scraper(monkeypatch):
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

    def fake_get(url, headers=None):
        called.append(url)
        if url.endswith("/users/user/gists"):
            return DummyResp(
                data=[
                    {
                        "id": "1",
                        "description": "d",
                        "files": {
                            "f1.txt": {
                                "filename": "f1.txt",
                                "raw_url": "raw1",
                            }
                        },
                    }
                ]
            )
        if url == "raw1":
            return DummyResp(text="content1")
        return DummyResp()

    monkeypatch.setattr(requests, "get", fake_get)
    mod = importlib.import_module("plugins.gist_scraper")
    scraper = mod.GistScraper()

    items = scraper.fetch_items("user")
    assert called[0].endswith("/users/user/gists")
    record = scraper.parse_item(items[0])
    assert record["description"] == "d"
    assert record["files"] == {"f1.txt": "content1"}
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
