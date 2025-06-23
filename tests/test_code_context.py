import importlib.util
import sys
from pathlib import Path

# Import CodeContext without triggering heavy dependencies
spec = importlib.util.spec_from_file_location(
    "devai.context_manager",
    Path(__file__).resolve().parents[1] / "devai" / "context_manager.py",
)
context_module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = context_module
spec.loader.exec_module(context_module)
CodeContext = context_module.CodeContext


def test_basic_update_accumulates_imports_and_functions():
    ctx = CodeContext()
    ctx.update("import os\n")
    ctx.update("def foo():\n    pass\n")
    assert ctx.imports == {"os"}
    assert ctx.functions == {"foo"}


def test_function_signature_split_across_updates():
    ctx = CodeContext()
    ctx.update("def bar(x,\n")
    assert "bar" not in ctx.functions
    ctx.update("        y):\n    return x + y\n")
    assert "bar" in ctx.functions
