import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

sw = importlib.import_module("scraper_wiki")


class DummyEmbed:
    def encode(self, *a, **k):
        import numpy as np

        return np.array([0.0])


sw.NLPProcessor.get_embedding_model = classmethod(lambda cls: DummyEmbed())


def _setup_builder(monkeypatch, **kwargs):
    builder = sw.DatasetBuilder(**kwargs)
    monkeypatch.setattr(builder, "_generate_questions", lambda *a, **k: [])
    monkeypatch.setattr(builder, "_generate_answers", lambda *a, **k: [])
    monkeypatch.setattr(sw, "extract_relations", lambda *a, **k: [])
    import numpy as np

    monkeypatch.setattr(
        builder.embedding_model, "encode", lambda *a, **k: np.array([0.0])
    )
    return builder


def test_trivial_function_filtered(monkeypatch):
    builder = _setup_builder(monkeypatch, min_complexity=2)
    code = "def foo():\n    pass"
    result = builder.generate_qa_pairs("T", code, "S", "en", "c")
    assert result == {}


def test_complex_function_filtered(monkeypatch):
    builder = _setup_builder(monkeypatch, max_complexity=3)
    code = """
def bar(x):
    if x > 0:
        for i in range(x):
            if i % 2 == 0:
                print(i)
    else:
        while x < 0:
            x += 1
    return x
"""
    result = builder.generate_qa_pairs("T", code, "S", "en", "c")
    assert result == {}
