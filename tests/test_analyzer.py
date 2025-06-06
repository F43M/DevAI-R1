import ast
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
