import importlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT))

gen = importlib.import_module("enrichment.generator")


def test_generate_diagram_creates_file(tmp_path):
    code = """\ndef a():\n    b()\n\ndef b():\n    pass\n"""
    path = gen.generate_diagram(code, tmp_path)
    assert Path(path).exists()


def test_generate_explanations_returns_levels():
    text = "First. Second sentence. Third one."
    ex = gen.generate_explanations(text)
    assert set(ex.keys()) == {"high", "medium", "low"}


def test_link_theory_matches_keywords():
    refs = gen.link_theory("quick sort implementation")
    assert any("Algorithms" in r or "Introduction" in r for r in refs)
