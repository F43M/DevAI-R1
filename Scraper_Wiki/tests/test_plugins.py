import importlib
import sys
from pathlib import Path
from types import SimpleNamespace, ModuleType

# Ensure repository root is on sys.path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Stub heavy dependencies to avoid installing them
sys.modules.setdefault(
    "sentence_transformers", SimpleNamespace(SentenceTransformer=object)
)
sys.modules.setdefault(
    "datasets",
    SimpleNamespace(Dataset=object, concatenate_datasets=lambda *a, **k: None),
)
sys.modules.setdefault("spacy", SimpleNamespace(load=lambda *a, **k: None))
sys.modules.setdefault("tqdm", SimpleNamespace(tqdm=lambda x, **k: x))
sys.modules.setdefault("html2text", SimpleNamespace(html2text=lambda x: x))
sk_mod = SimpleNamespace(
    cluster=SimpleNamespace(KMeans=object),
    feature_extraction=SimpleNamespace(text=SimpleNamespace(TfidfVectorizer=object)),
)
sys.modules.setdefault("sklearn", sk_mod)
sys.modules.setdefault("sklearn.cluster", sk_mod.cluster)
sys.modules.setdefault("sklearn.feature_extraction", sk_mod.feature_extraction)
sys.modules.setdefault(
    "sklearn.feature_extraction.text", sk_mod.feature_extraction.text
)
sumy_mod = SimpleNamespace(
    parsers=SimpleNamespace(plaintext=SimpleNamespace(PlaintextParser=object)),
    nlp=SimpleNamespace(tokenizers=SimpleNamespace(Tokenizer=object)),
    summarizers=SimpleNamespace(lsa=SimpleNamespace(LsaSummarizer=object)),
)
sys.modules.setdefault("sumy", sumy_mod)
sys.modules.setdefault("sumy.parsers", sumy_mod.parsers)
sys.modules.setdefault("sumy.parsers.plaintext", sumy_mod.parsers.plaintext)
sys.modules.setdefault("sumy.nlp", sumy_mod.nlp)
sys.modules.setdefault("sumy.nlp.tokenizers", sumy_mod.nlp.tokenizers)
sys.modules.setdefault("sumy.summarizers", sumy_mod.summarizers)
sys.modules.setdefault("sumy.summarizers.lsa", sumy_mod.summarizers.lsa)
sys.modules.setdefault("streamlit", SimpleNamespace())
sys.modules.setdefault(
    "psutil",
    SimpleNamespace(
        cpu_percent=lambda interval=1: 0,
        virtual_memory=lambda: SimpleNamespace(percent=0),
    ),
)
sys.modules.setdefault(
    "prometheus_client",
    SimpleNamespace(
        Counter=lambda *a, **k: object, start_http_server=lambda *a, **k: None
    ),
)
wiki_mod = ModuleType("wikipediaapi")
wiki_mod.WikipediaException = Exception
wiki_mod.Namespace = SimpleNamespace(MAIN=0, CATEGORY=14)
wiki_mod.ExtractFormat = SimpleNamespace(HTML=0)
wiki_mod.WikipediaPage = object
wiki_mod.Wikipedia = lambda *a, **k: SimpleNamespace(
    page=lambda *a, **k: SimpleNamespace(exists=lambda: False),
    api=SimpleNamespace(article_url=lambda x: ""),
)
sys.modules.setdefault("wikipediaapi", wiki_mod)
aiohttp_stub = SimpleNamespace(
    ClientSession=object,
    ClientTimeout=lambda *a, **k: None,
    ClientError=Exception,
    ClientResponseError=Exception,
)
sys.modules.setdefault("aiohttp", aiohttp_stub)
sys.modules.setdefault(
    "backoff",
    SimpleNamespace(
        on_exception=lambda *a, **k: (lambda f: f), expo=lambda *a, **k: None
    ),
)

import scraper_wiki as sw
import core


class DummyWiki:
    def __init__(self, lang: str) -> None:
        self.lang = lang

    def get_category_members(self, category: str):
        return [{"title": "Page", "lang": self.lang, "category": category}]


def test_infobox_parser(monkeypatch):
    monkeypatch.setattr(core, "WikipediaAdvanced", DummyWiki)
    html = '<table class="infobox"><tr><th>Name</th><td>Python</td></tr></table>'
    monkeypatch.setattr(sw, "fetch_html_content", lambda title, lang: html)
    mod = importlib.reload(importlib.import_module("plugins.infobox_parser"))
    plugin = mod.Plugin()
    items = plugin.fetch_items("en", "Prog")
    assert items == [{"title": "Page", "lang": "en", "category": "Prog"}]
    parsed = plugin.parse_item(items[0])
    assert parsed["title"] == "Page"
    assert parsed["language"] == "en"
    assert parsed["infobox"] == {"Name": "Python"}
    for field in [
        "raw_code",
        "context",
        "problems",
        "fixed_version",
        "lessons",
        "origin_metrics",
        "challenge",
    ]:
        assert field in parsed


def test_table_parser(monkeypatch):
    monkeypatch.setattr(core, "WikipediaAdvanced", DummyWiki)
    html = "<table><tr><th>A</th><th>B</th></tr><tr><td>1</td><td>2</td></tr></table>"
    monkeypatch.setattr(sw, "fetch_html_content", lambda title, lang: html)
    mod = importlib.reload(importlib.import_module("plugins.table_parser"))
    plugin = mod.Plugin()
    items = plugin.fetch_items("en", "Prog")
    assert items[0]["title"] == "Page"
    parsed = plugin.parse_item(items[0])
    assert parsed["title"] == "Page"
    assert parsed["language"] == "en"
    assert parsed["tables"] == [[["A", "B"], ["1", "2"]]]
    for field in [
        "raw_code",
        "context",
        "problems",
        "fixed_version",
        "lessons",
        "origin_metrics",
        "challenge",
    ]:
        assert field in parsed


def test_stackexchange_plugin(monkeypatch):
    class DummyResp:
        def __init__(self, data):
            self._data = data

        def raise_for_status(self):
            pass

        def json(self):
            return self._data

    resp = {
        "items": [
            {
                "title": "Q1",
                "body": "<p>code</p>",
                "score": 5,
                "link": "url",
                "tags": ["python"],
            },
            {
                "title": "Q2",
                "body": "<p>low</p>",
                "score": 1,
                "link": "url2",
                "tags": ["python"],
            },
        ]
    }

    def fake_get(url, params=None, timeout=None):
        return DummyResp(resp)

    import requests

    monkeypatch.setattr(requests, "get", fake_get)
    mod = importlib.reload(importlib.import_module("plugins.stackexchange"))
    plugin = mod.Plugin(site="superuser", min_score=5)
    items = plugin.fetch_items("en", "python")
    assert len(items) == 1
    assert items[0]["category"] == "python"
    parsed = plugin.parse_item(items[0])
    assert parsed["title"] == "Q1"
    assert parsed["language"] == "en"
    assert parsed["category"] == "python"
    assert parsed["score"] == 5
    assert parsed["link"] == "url"
    assert parsed["tags"] == [{"tag": "python", "link": "url"}]
    for field in [
        "raw_code",
        "context",
        "problems",
        "fixed_version",
        "lessons",
        "origin_metrics",
        "challenge",
    ]:
        assert field in parsed


def test_wikidata_plugin(monkeypatch):
    class DummyResp:
        def __init__(self, data):
            self._data = data

        def raise_for_status(self):
            pass

        def json(self):
            return self._data

    search_resp = {"search": [{"id": "Q1", "label": "Item", "description": "Desc"}]}
    entity_resp = {
        "entities": {
            "Q1": {
                "labels": {"en": {"value": "Item"}},
                "descriptions": {"en": {"value": "Desc"}},
                "claims": {"P18": [{"mainsnak": {"datavalue": {"value": "Pic.jpg"}}}]},
            }
        }
    }

    def fake_get(url, params=None, timeout=None):
        if params.get("action") == "wbsearchentities":
            return DummyResp(search_resp)
        return DummyResp(entity_resp)

    import requests

    monkeypatch.setattr(requests, "get", fake_get)
    mod = importlib.reload(importlib.import_module("plugins.wikidata"))
    plugin = mod.Plugin()
    items = plugin.fetch_items("en", "python")
    assert items[0]["category"] == "python"
    parsed = plugin.parse_item(items[0])
    assert parsed["title"] == "Item"
    assert parsed["language"] == "en"
    assert parsed["category"] == "python"
    assert parsed["description"] == "Desc"
    assert parsed["wikidata_id"] == "Q1"
    assert (
        parsed["image_url"]
        == "https://commons.wikimedia.org/wiki/Special:FilePath/Pic.jpg"
    )
    for field in [
        "raw_code",
        "context",
        "problems",
        "fixed_version",
        "lessons",
        "origin_metrics",
        "challenge",
    ]:
        assert field in parsed


def test_competitions_plugin(monkeypatch):
    class DummyResp:
        def __init__(self, data):
            self._data = data

        def raise_for_status(self):
            pass

        def json(self):
            return self._data

    def fake_get(url, timeout=None):
        if "leetcode" in url:
            return DummyResp({"content": "LC problem", "solution": "LC solution"})
        return DummyResp({"description": "CW problem", "solutions": ["CW solution"]})

    import requests

    monkeypatch.setattr(requests, "get", fake_get)
    mod = importlib.reload(importlib.import_module("plugins.competitions"))
    plugin = mod.Plugin()
    items = plugin.fetch_items("en", "two-sum")
    assert items[0]["slug"] == "two-sum"
    parsed = plugin.parse_item(items[0])
    assert parsed["problem"] == "LC problem"
    assert parsed["solution"] == "LC solution"
    for field in [
        "raw_code",
        "context",
        "problems",
        "fixed_version",
        "lessons",
        "origin_metrics",
        "challenge",
    ]:
        assert field in parsed
    parsed2 = plugin.parse_item(items[1])
    assert parsed2["problem"] == "CW problem"
    assert parsed2["solution"] == "CW solution"
    for field in [
        "raw_code",
        "context",
        "problems",
        "fixed_version",
        "lessons",
        "origin_metrics",
        "challenge",
    ]:
        assert field in parsed2
