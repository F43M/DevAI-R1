import importlib
import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

pp = importlib.import_module("training.postprocessing")


def test_analyze_code_ast_returns_metrics():
    code = "def foo(x):\n    if x:\n        return 1\n    return 0"
    res = pp.analyze_code_ast(code)
    assert res["language"] == "python"
    assert res["complexities"]["foo"] >= 2


def test_filter_by_complexity_filters_records():
    data = [
        {"content": "def a():\n    pass"},
        {"content": "def b(x):\n    if x:\n        return x"},
    ]
    filtered = pp.filter_by_complexity(data, min_complexity=2)
    assert len(filtered) == 1
    assert filtered[0]["content"].startswith("def b")
