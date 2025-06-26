import importlib
import sys
from types import SimpleNamespace
from pathlib import Path

# Ensure repository root is on sys.path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

sys.modules.setdefault("spacy", SimpleNamespace(load=lambda *a, **k: None))
sys.modules.setdefault(
    "googletrans",
    SimpleNamespace(
        Translator=lambda: SimpleNamespace(
            translate=lambda text, dest: SimpleNamespace(text=f"{text}-{dest}")
        )
    ),
)

code_mod = importlib.import_module("utils.code")


def test_normalize_indentation():
    raw = "    def f():\n        pass\n"
    assert code_mod.normalize_indentation(raw) == "def f():\n    pass"


def test_remove_comments_python():
    raw = "def f():\n    # c\n    return 1"
    assert code_mod.remove_comments(raw, "python") == "def f():\n    return 1"


def test_detect_programming_language():
    assert code_mod.detect_programming_language("def f():\n    pass") == "python"
    assert code_mod.detect_programming_language("console.log('x');") == "javascript"


def test_docstring_to_google_converts_rst():
    doc = "Short.\n:param x: number\n:type x: int\n:return: result\n:rtype: int"
    converted = code_mod.docstring_to_google(doc)
    assert "Args:" in converted
    assert "x (int): number" in converted
    assert "Returns:" in converted


def test_parse_function_signature_from_string():
    code = "def foo(x, y=1, *args, **kwargs):\n    pass"
    assert code_mod.parse_function_signature(code) == "foo(x, y, *args, **kwargs)"
