import asyncio
from pathlib import Path
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


def test_run_test_and_static_tasks(monkeypatch):
    analyzer = DummyAnalyzer()
    mem = DummyMemory()
    tm = TaskManager("missing.yaml", analyzer, mem)

    async def fake_test_task(task, *args):
        return ["ok"]

    async def fake_static_task(task, *args):
        return ["lint ok"]

    monkeypatch.setattr(tm, "_perform_test_task", fake_test_task)
    monkeypatch.setattr(tm, "_perform_static_analysis_task", fake_static_task)

    async def run():
        res1 = await tm.run_task("run_tests")
        res2 = await tm.run_task("static_analysis")
        return res1, res2

    r1, r2 = asyncio.run(run())
    assert r1 == ["ok"]
    assert r2 == ["lint ok"]


def test_security_analysis_task(monkeypatch):
    analyzer = DummyAnalyzer()
    mem = DummyMemory()
    tm = TaskManager("missing.yaml", analyzer, mem)

    async def fake_sec_task(task, *args):
        return ["secure"]

    monkeypatch.setattr(tm, "_perform_security_analysis_task", fake_sec_task)

    async def run():
        return await tm.run_task("security_analysis")

    res = asyncio.run(run())
    assert res == ["secure"]


def test_auto_refactor(monkeypatch, tmp_path):
    analyzer = DummyAnalyzer()
    mem = DummyMemory()
    tm = TaskManager("missing.yaml", analyzer, mem)

    test_file = tmp_path / "test.py"
    test_file.write_text("def foo():\n    return 1\n")

    async def fake_generate(self, prompt, max_length=0):
        return "def foo():\n    return 2\n"

    class DummyUpdater:
        def __init__(self):
            self.called = False
        def safe_apply(
            self,
            file_path,
            apply_func,
            max_attempts=1,
            capture_output=False,
            keep_backup=False,
        ):
            apply_func(Path(file_path))
            self.called = True
            return True

    async def fake_safe(self, prompt, max_tokens, context="", memory=None):
        return await fake_generate(self, prompt, max_tokens)

    tm.ai_model = type("AI", (), {"generate": fake_generate, "safe_api_call": fake_safe})()
    import devai.update_manager as upd
    monkeypatch.setattr(upd, "UpdateManager", lambda tests_cmd=None: DummyUpdater())

    async def run():
        return await tm.run_task("auto_refactor", str(test_file))

    res = asyncio.run(run())
    assert res["success"] is True
    assert test_file.read_text() == "def foo():\n    return 2\n"

