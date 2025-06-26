import importlib
import sys
from pathlib import Path
from types import SimpleNamespace, ModuleType

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Minimal stub for html2text
sys.modules.setdefault("html2text", SimpleNamespace(html2text=lambda x: x))
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


def test_mdn_docs(monkeypatch):
    import requests

    core_stub = ModuleType("core")
    core_stub.builder = SimpleNamespace(DatasetBuilder=object)
    monkeypatch.setitem(sys.modules, "core", core_stub)
    monkeypatch.setitem(sys.modules, "core.builder", core_stub.builder)

    def fake_get(url, timeout=None):
        return DummyResp(text="<h1>MDN</h1><p>Doc</p>")

    monkeypatch.setattr(requests, "get", fake_get)
    mod = importlib.import_module("plugins.mdn_docs")
    plugin = mod.Plugin()
    items = plugin.fetch_items("en", "Web/API")
    assert items[0]["url"].endswith("/en/docs/Web/API")
    parsed = plugin.parse_item(items[0])
    assert parsed["language"] == "en"
    assert "Doc" in parsed["content"]
    _check_defaults(parsed)


def test_devdocs(monkeypatch):
    import requests

    core_stub = ModuleType("core")
    core_stub.builder = SimpleNamespace(DatasetBuilder=object)
    monkeypatch.setitem(sys.modules, "core", core_stub)
    monkeypatch.setitem(sys.modules, "core.builder", core_stub.builder)

    def fake_get(url, timeout=None):
        return DummyResp(text="<h1>DevDocs</h1>")

    monkeypatch.setattr(requests, "get", fake_get)
    mod = importlib.import_module("plugins.devdocs")
    plugin = mod.Plugin()
    items = plugin.fetch_items("en", "python")
    assert "devdocs.io" in items[0]["url"]
    parsed = plugin.parse_item(items[0])
    assert "DevDocs" in parsed["content"]
    _check_defaults(parsed)


def test_university_courses(monkeypatch):
    import requests

    core_stub = ModuleType("core")
    core_stub.builder = SimpleNamespace(DatasetBuilder=object)
    monkeypatch.setitem(sys.modules, "core", core_stub)
    monkeypatch.setitem(sys.modules, "core.builder", core_stub.builder)

    def fake_get(url, timeout=None):
        return DummyResp(text="<h1>Course</h1>")

    monkeypatch.setattr(requests, "get", fake_get)
    mod = importlib.import_module("plugins.university_courses")
    plugin = mod.Plugin()
    items = plugin.fetch_items("en", "http://example.com/course")
    assert items[0]["url"] == "http://example.com/course"
    parsed = plugin.parse_item(items[0])
    assert "Course" in parsed["content"]
    _check_defaults(parsed)


def test_rfc_scraper(monkeypatch):
    import requests

    core_stub = ModuleType("core")
    core_stub.builder = SimpleNamespace(DatasetBuilder=object)
    monkeypatch.setitem(sys.modules, "core", core_stub)
    monkeypatch.setitem(sys.modules, "core.builder", core_stub.builder)

    def fake_get(url, timeout=None):
        return DummyResp(text="RFC text")

    monkeypatch.setattr(requests, "get", fake_get)
    mod = importlib.import_module("plugins.rfc_scraper")
    plugin = mod.Plugin()
    items = plugin.fetch_items("en", "1234")
    assert items[0]["url"].endswith("rfc1234.txt")
    parsed = plugin.parse_item(items[0])
    assert "RFC" in parsed["content"]
    _check_defaults(parsed)


def test_npm_packages(monkeypatch):
    import requests

    core_stub = ModuleType("core")
    core_stub.builder = SimpleNamespace(DatasetBuilder=object)
    monkeypatch.setitem(sys.modules, "core", core_stub)
    monkeypatch.setitem(sys.modules, "core.builder", core_stub.builder)

    def fake_get(url, timeout=None):
        return DummyResp(data={"description": "desc", "readme": "readme"})

    monkeypatch.setattr(requests, "get", fake_get)
    mod = importlib.import_module("plugins.npm_packages")
    plugin = mod.Plugin()
    items = plugin.fetch_items("en", "pkg")
    assert items[0]["name"] == "pkg"
    parsed = plugin.parse_item(items[0])
    assert parsed["description"] == "desc"
    assert parsed["readme"] == "readme"
    _check_defaults(parsed)


def test_pip_packages(monkeypatch):
    import requests

    core_stub = ModuleType("core")
    core_stub.builder = SimpleNamespace(DatasetBuilder=object)
    monkeypatch.setitem(sys.modules, "core", core_stub)
    monkeypatch.setitem(sys.modules, "core.builder", core_stub.builder)

    def fake_get(url, timeout=None):
        return DummyResp(data={"info": {"summary": "desc", "home_page": "home"}})

    monkeypatch.setattr(requests, "get", fake_get)
    mod = importlib.import_module("plugins.pip_packages")
    plugin = mod.Plugin()
    items = plugin.fetch_items("en", "pkg")
    parsed = plugin.parse_item(items[0])
    assert parsed["description"] == "desc"
    assert parsed["home_page"] == "home"
    _check_defaults(parsed)


def test_cran_packages(monkeypatch):
    import requests

    core_stub = ModuleType("core")
    core_stub.builder = SimpleNamespace(DatasetBuilder=object)
    monkeypatch.setitem(sys.modules, "core", core_stub)
    monkeypatch.setitem(sys.modules, "core.builder", core_stub.builder)

    def fake_get(url, timeout=None):
        return DummyResp(data={"Title": "T", "Description": "D"})

    monkeypatch.setattr(requests, "get", fake_get)
    mod = importlib.import_module("plugins.cran_packages")
    plugin = mod.Plugin()
    items = plugin.fetch_items("en", "pkg")
    parsed = plugin.parse_item(items[0])
    assert parsed["title"] == "T"
    assert parsed["description"] == "D"
    _check_defaults(parsed)


def test_jira_scraper(monkeypatch):
    import requests

    core_stub = ModuleType("core")
    core_stub.builder = SimpleNamespace(DatasetBuilder=object)
    monkeypatch.setitem(sys.modules, "core", core_stub)
    monkeypatch.setitem(sys.modules, "core.builder", core_stub.builder)

    def fake_get(url, timeout=None):
        return DummyResp(data={"fields": {"summary": "Bug", "description": "Fix"}})

    monkeypatch.setattr(requests, "get", fake_get)
    mod = importlib.import_module("plugins.jira_scraper")
    plugin = mod.Plugin()
    items = plugin.fetch_items("en", "ABC-1")
    parsed = plugin.parse_item(items[0])
    assert parsed["title"] == "Bug"
    assert parsed["description"] == "Fix"
    _check_defaults(parsed)


def test_bugzilla_scraper(monkeypatch):
    import requests

    core_stub = ModuleType("core")
    core_stub.builder = SimpleNamespace(DatasetBuilder=object)
    monkeypatch.setitem(sys.modules, "core", core_stub)
    monkeypatch.setitem(sys.modules, "core.builder", core_stub.builder)

    def fake_get(url, timeout=None):
        return DummyResp(data={"bugs": [{"summary": "Bug", "description": "Desc"}]})

    monkeypatch.setattr(requests, "get", fake_get)
    mod = importlib.import_module("plugins.bugzilla_scraper")
    plugin = mod.Plugin()
    items = plugin.fetch_items("en", "123")
    parsed = plugin.parse_item(items[0])
    assert parsed["title"] == "Bug"
    assert parsed["description"] == "Desc"
    _check_defaults(parsed)
