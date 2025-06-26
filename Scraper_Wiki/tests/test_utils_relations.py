import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


# Dummy NLP components
class DummyToken:
    def __init__(self, text, dep, i=0, lemma=None):
        self.text = text
        self.dep_ = dep
        self.i = i
        self.lemma_ = lemma or text
        self.children = []


class DummyEnt:
    def __init__(self, text, start, end):
        self.text = text
        self.start = start
        self.end = end
        self.label_ = "PERSON"


class DummySent(list):
    pass


class DummyDoc:
    def __init__(self):
        sub = DummyToken("Guido", "nsubj", i=0)
        root = DummyToken("created", "ROOT", i=1, lemma="create")
        obj = DummyToken("Python", "dobj", i=2)
        root.children = [sub, obj]
        self.tokens = [sub, root, obj]
        self.sents = [DummySent(self.tokens)]
        self.ents = [DummyEnt("Guido", 0, 1), DummyEnt("Python", 2, 3)]

    def __iter__(self):
        return iter(self.tokens)


class DummyNLP:
    def __call__(self, text):
        return DummyDoc()


class DummyProc:
    @classmethod
    def get_instance(cls, lang):
        return DummyNLP()


def test_extract_relations_simple(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=lambda *a, **k: None),
    )
    monkeypatch.setitem(
        sys.modules, "spacy", SimpleNamespace(load=lambda *a, **k: DummyNLP())
    )
    monkeypatch.setitem(
        sys.modules, "scraper_wiki", SimpleNamespace(NLPProcessor=DummyProc)
    )

    relations_mod = importlib.reload(importlib.import_module("utils.relation"))
    rels = relations_mod.extract_relations("Guido created Python", "en")
    assert rels == [{"subject": "Guido", "relation": "create", "object": "Python"}]


def test_extract_relations_returns_empty_on_error(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=lambda *a, **k: None),
    )
    monkeypatch.setitem(
        sys.modules, "spacy", SimpleNamespace(load=lambda *a, **k: None)
    )
    monkeypatch.setitem(
        sys.modules,
        "scraper_wiki",
        SimpleNamespace(
            NLPProcessor=SimpleNamespace(
                get_instance=lambda lang: (_ for _ in ()).throw(Exception())
            )
        ),
    )
    relations_mod = importlib.reload(importlib.import_module("utils.relation"))
    rels = relations_mod.extract_relations("text", "en")
    assert rels == []


def test_token_to_ent_text_fallback():
    import utils.relation as rel

    token = DummyToken("Word", "nsubj", i=0)
    ent = DummyEnt("Other", 1, 2)
    assert rel._token_to_ent_text(token, [ent]) == "Word"


def test_extract_relations_regex_basic(monkeypatch):
    monkeypatch.setitem(
        sys.modules, "spacy", SimpleNamespace(load=lambda *a, **k: None)
    )
    import utils.relation as rel

    text = "Ada worked at IBM. Bob studied at MIT."
    res = rel.extract_relations_regex(text)
    assert {"subject": "Ada", "relation": "worked at", "object": "IBM"} in res
    assert {"subject": "Bob", "relation": "studied at", "object": "MIT"} in res


def test_relations_to_graph_builds_edges(monkeypatch):
    monkeypatch.setitem(
        sys.modules, "spacy", SimpleNamespace(load=lambda *a, **k: None)
    )
    import utils.relation as rel

    relations = [
        {"subject": "Ada", "relation": "worked at", "object": "IBM"},
        {"subject": "Bob", "relation": "studied at", "object": "MIT"},
    ]
    g = rel.relations_to_graph(relations)
    assert g.has_edge("Ada", "IBM")
    assert g["Ada"]["IBM"]["relation"] == "worked at"


def test_relations_to_graph_persists(monkeypatch):
    monkeypatch.setitem(
        sys.modules, "spacy", SimpleNamespace(load=lambda *a, **k: None)
    )
    saved = {}
    monkeypatch.setitem(
        sys.modules,
        "integrations.neo4j_backend",
        SimpleNamespace(save_graph=lambda g: saved.update(nodes=g.number_of_nodes())),
    )
    import importlib

    rel = importlib.reload(importlib.import_module("utils.relation"))
    relations = [{"subject": "Ada", "relation": "worked at", "object": "IBM"}]
    rel.relations_to_graph(relations, persist=True)
    assert saved["nodes"] == 2
