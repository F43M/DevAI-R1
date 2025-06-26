import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from types import SimpleNamespace

sys.modules.setdefault("spacy", SimpleNamespace(load=lambda *a, **k: None))
sys.modules.setdefault(
    "googletrans",
    SimpleNamespace(
        Translator=lambda: SimpleNamespace(
            translate=lambda text, dest: SimpleNamespace(text=f"{text}-{dest}")
        )
    ),
)

from utils.code_sniffer import scan


def test_scan_detects_and_fixes():
    code = """\
def foo(x):
    if x == True:
        eval('x')
        exec('y')
    return x
"""
    problems, fixed = scan(code)
    assert any("eval" in p for p in problems)
    assert any("exec" in p for p in problems)
    assert any("True" in p for p in problems)
    lines = fixed.splitlines()
    # line with eval should be commented
    assert lines[3].lstrip().startswith("#")
    assert "is True" in fixed


def test_scan_returns_original_on_invalid():
    code = "foo ="
    problems, fixed = scan(code)
    assert problems == []
    assert fixed == code
