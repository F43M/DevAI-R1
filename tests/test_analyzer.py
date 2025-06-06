import ast
import asyncio
from devai.analyzer import CodeAnalyzer

class DummyMemory:
    def __init__(self):
        self.saved = []
    def save(self, entry, update_feedback=False):
        self.saved.append(entry)
    conn = type('conn', (), {'cursor': lambda self: type('cur', (), {'execute': lambda self, *a, **k: None, 'fetchall': lambda self: []})()})()
    def add_semantic_relation(self, *a, **k):
        pass


def test_get_function_calls():
    code = """
def bar():
    pass

def foo():
    bar()
"""
    tree = ast.parse(code)
    node = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "foo"][0]
    analyzer = CodeAnalyzer(".", DummyMemory())
    calls = analyzer._get_function_calls(node)
    assert {"function": "bar", "line": 6} in calls


def test_file_operations(tmp_path):
    root = tmp_path / "app"
    root.mkdir()
    sub = root / "sub"
    sub.mkdir()
    file = sub / "f.py"
    file.write_text("a\nb\nc\n")
    analyzer = CodeAnalyzer(str(root), DummyMemory())

    async def run():
        ls = await analyzer.list_dir("sub")
        lines = await analyzer.read_lines("sub/f.py", 2, 2)
        ok = await analyzer.edit_line("sub/f.py", 2, "x")
        return ls, lines, ok

    ls, lines, ok = asyncio.run(run())
    assert "sub/f.py" in ls
    assert lines == ["b"]
    assert ok
    assert file.read_text().splitlines()[1] == "x"
