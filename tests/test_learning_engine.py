import asyncio
from pathlib import Path

from devai.learning_engine import LearningEngine
from devai.analyzer import CodeAnalyzer
from devai.memory import MemoryManager
from devai.config import config


class DummyModel:
    async def safe_api_call(self, prompt, max_tokens, context="", memory=None):
        return "resumo"


def test_explain_learning_lessons(tmp_path, monkeypatch):
    db = tmp_path / "mem.sqlite"
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    monkeypatch.setattr(config, "LOG_DIR", str(log_dir))
    mem = MemoryManager(str(db), "dummy", model=None, index=None)
    analyzer = CodeAnalyzer(str(tmp_path), mem)
    mem.save({"type": "l", "memory_type": "erro_resolvido", "content": "fix", "metadata": {}})
    mem.save({"type": "l", "memory_type": "refatoracao_aplicada", "content": "change", "metadata": {}})
    engine = LearningEngine(analyzer, mem, DummyModel())

    async def run():
        return await engine.explain_learning_lessons()

    result = asyncio.run(run())
    summary_file = Path("logs/learning_summary.md")
    assert summary_file.exists()
    assert result == "resumo"


def test_skip_already_processed(tmp_path):
    db = tmp_path / "mem.sqlite"
    mem = MemoryManager(str(db), "dummy", model=None, index=None)
    analyzer = CodeAnalyzer(str(tmp_path), mem)
    file = tmp_path / "f.py"
    file.write_text("def a():\n    pass")
    analyzer.code_chunks = {
        "a": {
            "name": "a",
            "file": str(file),
            "code": "def a(): pass",
            "hash": "h1",
        }
    }

    engine = LearningEngine(analyzer, mem, DummyModel(), rate_limit=2)
    count = 0

    async def fake_call(*a, **k):
        nonlocal count
        count += 1
        return "ok"

    engine._rate_limited_call = fake_call
    asyncio.run(engine.learn_from_codebase())
    progress_file = Path("logs/learning_progress.json")
    assert progress_file.exists()
    assert count == 3
    count = 0
    asyncio.run(engine.learn_from_codebase())
    assert count == 0


def test_concurrent_calls(tmp_path):
    db = tmp_path / "mem.sqlite"
    mem = MemoryManager(str(db), "dummy", model=None, index=None)
    analyzer = CodeAnalyzer(str(tmp_path), mem)
    files = []
    chunks = {}
    for i in range(3):
        f = tmp_path / f"f{i}.py"
        f.write_text(f"def f{i}():\n    pass")
        files.append(f)
        chunks[f"f{i}"] = {
            "name": f"f{i}",
            "file": str(f),
            "code": f"def f{i}(): pass",
            "hash": str(i),
        }

    analyzer.code_chunks = chunks

    engine = LearningEngine(analyzer, mem, DummyModel(), rate_limit=2)
    active = 0
    observed = 0

    async def fake_call(*a, **k):
        nonlocal active, observed
        active += 1
        observed = max(observed, active)
        await asyncio.sleep(0.01)
        active -= 1
        return "ok"

    engine._rate_limited_call = fake_call
    asyncio.run(engine.learn_from_codebase())
    assert observed > 1
    assert observed <= 2
