import importlib
import sys
from types import SimpleNamespace
from pathlib import Path

# Ensure repository root is on sys.path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class DummyEnt:
    def __init__(self, text: str, label: str) -> None:
        self.text = text
        self.label_ = label


class DummyDoc:
    def __init__(self) -> None:
        self.ents = [DummyEnt("Python", "ORG"), DummyEnt("Guido", "PERSON")]


class DummyNLP:
    def __call__(self, text: str):
        return DummyDoc()


# Load utils.text with spaCy and googletrans mocked so that importing does not require the real packages
sys.modules["spacy"] = SimpleNamespace(load=lambda *a, **k: DummyNLP())
sys.modules["googletrans"] = SimpleNamespace(Translator=lambda: SimpleNamespace(translate=lambda text, dest: SimpleNamespace(text=f"{text}-{dest}")))
text_mod = importlib.import_module("utils.text")
# After importing, replace spacy and googletrans with no-op modules to avoid accidental use
sys.modules["spacy"] = SimpleNamespace(load=lambda *a, **k: None)
sys.modules["googletrans"] = SimpleNamespace(Translator=lambda: SimpleNamespace(translate=lambda text, dest: SimpleNamespace(text=f"{text}-{dest}")))


def test_clean_text_removes_refs_and_spaces():
    raw = "Example [1] text  with\n  extra spaces [23]"
    assert text_mod.clean_text(raw) == "Example text with extra spaces"


def test_normalize_person_extracts_fields():
    data = {"name": "Guido", "birth_date": "1956", "occupation": "Programmer|BDFL"}
    result = text_mod.normalize_person(data)
    assert result == {
        "name": "Guido",
        "birth_date": "1956",
        "occupation": ["Programmer", "BDFL"],
    }


def test_extract_entities_returns_entities(monkeypatch):
    monkeypatch.setattr(text_mod, "nlp", DummyNLP())
    ents = text_mod.extract_entities("Any text")
    assert ents == [
        {"text": "Python", "type": "ORG"},
        {"text": "Guido", "type": "PERSON"},
    ]


def test_parse_date_iso():
    assert text_mod.parse_date("January 1, 2020") == "2020-01-01"


def test_normalize_infobox_converts_dates(monkeypatch):
    monkeypatch.setattr(text_mod, "parse_date", lambda v: "2020-01-01")
    info = {"title": "Page", "date": "1 Jan 2020", "other": "x"}
    out = text_mod.normalize_infobox(info)
    assert out == {"title": "Page", "date": "2020-01-01", "other": "x"}


def test_clean_text_handles_wikilinks_and_sup():
    raw = "Hello [[Link|World]]<sup>1</sup> [[Page]]"
    assert text_mod.clean_text(raw) == "Hello World Page"


def test_clean_text_normalizes_nfkc():
    raw = "ＡＢＣ"
    assert text_mod.clean_text(raw) == "ABC"


def test_translate_text_returns_translated():
    out = text_mod.translate_text("hello", "es")
    assert out == "hello-es"


def test_advanced_clean_text_with_stemming(monkeypatch):
    class DummyTok:
        def __init__(self, text):
            self.text = text
            self.lemma_ = text.lower() + "_lemma"
            self.is_stop = False

    class DummyStemNLP:
        def __call__(self, text):
            return [DummyTok(t) for t in text.split()]

    import scraper_wiki as sw

    monkeypatch.setattr(sw.NLPProcessor, "get_instance", classmethod(lambda cls, lang: DummyStemNLP()))
    cleaned = sw.advanced_clean_text("RUNS Running", "en", stem=True)
    assert cleaned == "runs_lemma running_lemma"

