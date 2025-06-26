import importlib
import sys
from pathlib import Path
from types import SimpleNamespace, ModuleType

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Stub heavy dependencies similar to other plugin tests
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


def test_legacy_forums_plugin(monkeypatch):
    import requests

    class DummyResp:
        def __init__(self, data):
            self._data = data

        def raise_for_status(self):
            pass

        def json(self):
            return self._data

    def fake_get(url, params=None, timeout=None):
        if url.startswith("http://yahoo"):
            return DummyResp({"question": "YQ", "answer": "YA"})
        if url.startswith("http://delphi"):
            return DummyResp({"title": "DQ", "reply": "DA"})
        if url.startswith("http://php"):
            return DummyResp({"question": "PQ", "answer": "PA"})
        return DummyResp({})

    monkeypatch.setattr(requests, "get", fake_get)
    mod = importlib.reload(importlib.import_module("plugins.legacy_forums"))
    plugin = mod.Plugin(
        yahoo_url="http://yahoo", delphi_url="http://delphi", php_url="http://php"
    )

    items = plugin.fetch_items("pt", "duvida")
    assert len(items) == 3
    assert items[0]["source"] == "yahoo"
    py = plugin.parse_item(items[0])
    assert py["question"] == "YQ"
    assert py["answer"] == "YA"
    assert py["source"] == "yahoo"
    assert py["language"] == "pt"
    for field in [
        "raw_code",
        "context",
        "problems",
        "fixed_version",
        "lessons",
        "origin_metrics",
        "challenge",
    ]:
        assert field in py
    pd = plugin.parse_item(items[1])
    assert pd["question"] == "DQ"
    assert pd["answer"] == "DA"
    assert pd["source"] == "delphi"
    assert pd["language"] == "pt"
    for field in [
        "raw_code",
        "context",
        "problems",
        "fixed_version",
        "lessons",
        "origin_metrics",
        "challenge",
    ]:
        assert field in pd
    pp = plugin.parse_item(items[2])
    assert pp["question"] == "PQ"
    assert pp["answer"] == "PA"
    assert pp["source"] == "php"
    assert pp["language"] == "pt"
    for field in [
        "raw_code",
        "context",
        "problems",
        "fixed_version",
        "lessons",
        "origin_metrics",
        "challenge",
    ]:
        assert field in pp
