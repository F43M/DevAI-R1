import sys
from types import SimpleNamespace

sys.modules.setdefault("spacy", SimpleNamespace(load=lambda *a, **k: None))
sys.modules.setdefault(
    "sentence_transformers", SimpleNamespace(SentenceTransformer=object)
)
sys.modules.setdefault(
    "datasets",
    SimpleNamespace(Dataset=object, concatenate_datasets=lambda *a, **k: None),
)
sys.modules.setdefault("tqdm", SimpleNamespace(tqdm=lambda x, **k: x))
sys.modules.setdefault("unidecode", SimpleNamespace(unidecode=lambda x: x))
sys.modules.setdefault("networkx", SimpleNamespace())
sys.modules.setdefault(
    "sklearn",
    SimpleNamespace(
        cluster=SimpleNamespace(KMeans=object),
        feature_extraction=SimpleNamespace(
            text=SimpleNamespace(TfidfVectorizer=object)
        ),
    ),
)
sys.modules.setdefault("sklearn.cluster", SimpleNamespace(KMeans=object))
sys.modules.setdefault(
    "sklearn.feature_extraction",
    SimpleNamespace(text=SimpleNamespace(TfidfVectorizer=object)),
)
sys.modules.setdefault(
    "sklearn.feature_extraction.text",
    SimpleNamespace(TfidfVectorizer=object),
)
sys.modules.setdefault("sumy", SimpleNamespace())
sys.modules.setdefault("sumy.parsers", SimpleNamespace())
sys.modules.setdefault(
    "sumy.parsers.plaintext", SimpleNamespace(PlaintextParser=object)
)
sys.modules.setdefault("sumy.nlp", SimpleNamespace())
sys.modules.setdefault("sumy.nlp.tokenizers", SimpleNamespace(Tokenizer=object))
sys.modules.setdefault("sumy.summarizers", SimpleNamespace())
sys.modules.setdefault("sumy.summarizers.lsa", SimpleNamespace(LsaSummarizer=object))
sys.modules.setdefault("simhash", SimpleNamespace(Simhash=object, SimhashIndex=object))
sys.modules.setdefault("psutil", SimpleNamespace())
sys.modules.setdefault("graphviz", SimpleNamespace(Digraph=object))

import scraper_wiki as sw


class DummyPage:
    def __init__(self, text, categories):
        self.text = text
        self.categories = {c: None for c in categories}

    def exists(self):
        return True


def test_should_queue_rejects_disambiguation(monkeypatch):
    page = DummyPage("a" * 200, ["Disambiguation pages"])
    monkeypatch.setattr(sw.WikipediaAdvanced, "fetch_page", lambda self, t: page)
    monkeypatch.setattr(sw, "fetch_pageviews", lambda t, l: sw.Config.MIN_PAGEVIEWS + 1)
    wiki = sw.WikipediaAdvanced("en")
    assert not wiki.should_queue("A")


def test_should_queue_rejects_short_page(monkeypatch):
    page = DummyPage("short", [])
    monkeypatch.setattr(sw.WikipediaAdvanced, "fetch_page", lambda self, t: page)
    monkeypatch.setattr(sw, "fetch_pageviews", lambda t, l: sw.Config.MIN_PAGEVIEWS + 1)
    monkeypatch.setattr(sw.Config, "MIN_TEXT_LENGTH", 10)
    wiki = sw.WikipediaAdvanced("en")
    assert not wiki.should_queue("A")


def test_should_queue_rejects_low_pageviews(monkeypatch):
    page = DummyPage("a" * 200, [])
    monkeypatch.setattr(sw.WikipediaAdvanced, "fetch_page", lambda self, t: page)
    monkeypatch.setattr(sw, "fetch_pageviews", lambda t, l: sw.Config.MIN_PAGEVIEWS - 1)
    wiki = sw.WikipediaAdvanced("en")
    assert not wiki.should_queue("A")


def test_should_queue_accepts_valid(monkeypatch):
    page = DummyPage("a" * 200, [])
    monkeypatch.setattr(sw.WikipediaAdvanced, "fetch_page", lambda self, t: page)
    monkeypatch.setattr(sw, "fetch_pageviews", lambda t, l: sw.Config.MIN_PAGEVIEWS + 1)
    monkeypatch.setattr(sw.Config, "MIN_TEXT_LENGTH", 10)
    wiki = sw.WikipediaAdvanced("en")
    assert wiki.should_queue("A")
