import asyncio
from pathlib import Path
from devai.tasks import TaskManager
import devai.tasks as tasks_module


class DummyAnalyzer:
    def __init__(self):
        self.code_root = "."
        self.code_chunks = {
            "foo": {
                "name": "foo",
                "file": "foo.py",
                "code": "def foo(): return 1",
                "dependencies": [],
                "docstring": "",
            }
        }
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

    async def fake_test_task(task, *args, progress_cb=None):
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
        return (
            "--- test.py\n"
            "+++ test.py\n"
            "@@\n"
            "-def foo():\n"
            "-    return 1\n"
            "+def foo():\n"
            "+    return 2\n"
        )

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

    tm.ai_model = type(
        "AI", (), {"generate": fake_generate, "safe_api_call": fake_safe}
    )()
    import devai.update_manager as upd
    import devai.patch_utils as patch_utils

    monkeypatch.setattr(upd, "UpdateManager", lambda tests_cmd=None: DummyUpdater())
    monkeypatch.setattr(patch_utils, "split_diff_by_file", lambda diff: {str(test_file): diff})
    monkeypatch.setattr(
        patch_utils,
        "apply_patch_to_file",
        lambda path, diff: Path(path).write_text("def foo():\n    return 2\n"),
    )

    async def run():
        return await tm.run_task("auto_refactor", str(test_file))

    res = asyncio.run(run())
    assert res["success"] is True
    assert test_file.read_text() == "def foo():\n    return 2\n"

def test_auto_refactor_apply_patch_success(monkeypatch, tmp_path):
    analyzer = DummyAnalyzer()
    mem = DummyMemory()
    tm = TaskManager("missing.yaml", analyzer, mem)

    test_file = tmp_path / "code.py"
    test_file.write_text("def foo():\n    return 1\n")

    diff = "\n".join([
        f"--- a/{test_file.name}",
        f"+++ b/{test_file.name}",
        "@@",
        "-def foo():",
        "-    return 1",
        "+def foo():",
        "+    return 2",
    ]) + "\n"

    async def fake_safe(self, prompt, max_tokens, context="", memory=None):
        return diff

    tm.ai_model = type("AI", (), {"safe_api_call": fake_safe})()

    class DummyUpd:
        def __init__(self):
            self.called = False
        def safe_apply(self, path, func, *a, **k):
            func(Path(path))
            self.called = True
            return True

    upd_inst = DummyUpd()
    import devai.update_manager as upd_mod
    monkeypatch.setattr(upd_mod, "UpdateManager", lambda tests_cmd=None: upd_inst)
    monkeypatch.setattr(tasks_module, "split_diff_by_file", lambda d: {str(test_file): d})
    monkeypatch.setattr(tasks_module, "apply_patch_to_file", lambda p, d: Path(p).write_text("def foo():\n    return 2\n"))
    from devai.config import config
    monkeypatch.setattr(config, "APPROVAL_MODE", "full_auto")

    res = asyncio.run(tm._perform_auto_refactor_task({}, str(test_file)))
    assert res["success"] is True
    assert upd_inst.called
    assert test_file.read_text().splitlines()[-1] == "    return 2"
