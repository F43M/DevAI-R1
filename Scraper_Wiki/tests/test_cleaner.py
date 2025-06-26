import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Avoid loading real spaCy models during import
sys.modules.setdefault("spacy", SimpleNamespace(load=lambda *a, **k: SimpleNamespace()))


def test_clean_wiki_text_removes_markup():
    mod = importlib.import_module("utils.cleaner")
    raw = "<p>Hello [[Link|World]] {{temp}}</p>"
    assert mod.clean_wiki_text(raw) == "Hello World"


def test_split_sentences_spacy(monkeypatch):
    class DummySent:
        def __init__(self, text):
            self.text = text

    class DummyDoc:
        def __init__(self, text):
            parts = [p.strip().rstrip('.') for p in text.split('.') if p.strip()]
            self.sents = [DummySent(p + '.') for p in parts]

    class DummyNLP:
        def __call__(self, text):
            return DummyDoc(text)

    monkeypatch.setitem(sys.modules, "spacy", SimpleNamespace(load=lambda *a, **k: DummyNLP()))
    mod = importlib.reload(importlib.import_module("utils.cleaner"))
    result = mod.split_sentences("A. B.", "en")
    assert result == ["A.", "B."]


def test_split_sentences_nltk_fallback(monkeypatch):
    monkeypatch.setitem(sys.modules, "spacy", SimpleNamespace(load=lambda *a, **k: (_ for _ in ()).throw(Exception())))
    nltk_mod = SimpleNamespace(tokenize=SimpleNamespace(sent_tokenize=lambda text, language="en": ["X", "Y"]))
    monkeypatch.setitem(sys.modules, "nltk", nltk_mod)
    monkeypatch.setitem(sys.modules, "nltk.tokenize", nltk_mod.tokenize)
    mod = importlib.reload(importlib.import_module("utils.cleaner"))
    result = mod.split_sentences("X Y", "en")
    assert result == ["X", "Y"]
