import importlib
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

pipeline = importlib.import_module("processing.pipeline")
analysis = importlib.import_module("utils.gap_analysis")


def test_topic_entropy():
    recs = [{"topic": "a"}, {"topic": "a"}, {"topic": "b"}]
    ent = pipeline.compute_topic_entropy(recs)
    expected = -(2 / 3 * math.log2(2 / 3) + 1 / 3 * math.log2(1 / 3))
    assert round(ent, 4) == round(expected, 4)


def test_balance_languages():
    recs = [
        {"metadata": {"code_language": "py"}},
        {"metadata": {"code_language": "py"}},
        {"metadata": {"code_language": "js"}},
    ]
    balanced = pipeline.balance_languages(recs)
    langs = [r["metadata"]["code_language"] for r in balanced]
    assert langs.count("py") == langs.count("js") == 1


def test_identify_gaps():
    recs = [
        {"language": "en", "topic": "a"},
        {"language": "en", "topic": "b"},
        {"language": "pt", "topic": "a"},
    ]
    gaps = analysis.identify_gaps(recs, min_ratio=0.4)
    assert gaps == {"languages": ["pt"], "topics": ["b"]}
