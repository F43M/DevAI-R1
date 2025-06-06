import asyncio
from devai.tasks import TaskManager
import devai.tasks as tasks_module

class DummyAnalyzer:
    def __init__(self):
        self.code_root = "."
        self.code_chunks = {"foo": {"name": "foo", "file": "foo.py", "code": "def foo(): return 1", "dependencies": [], "docstring": ""}}
        self.code_graph = tasks_module.nx.DiGraph()
        self.code_graph.add_node("foo")
        self.learned_rules = {}

class DummyMemory:
    def __init__(self):
        self.saved = []
    def save(self, entry, update_feedback=False):
        self.saved.append(entry)


def test_run_default_task():
    analyzer = DummyAnalyzer()
    mem = DummyMemory()
    tm = TaskManager("missing.yaml", analyzer, mem)
    async def run():
        return await tm.run_task("impact_analysis", "foo")
    res = asyncio.run(run())
    assert res is not None
    assert mem.saved
