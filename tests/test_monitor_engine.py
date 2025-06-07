import os
import asyncio
from datetime import datetime, timedelta
from pathlib import Path

from devai.monitor_engine import auto_monitor_cycle
from devai.analyzer import CodeAnalyzer
from devai.memory import MemoryManager
from devai.config import config


class DummyModel:
    async def generate(self, prompt, max_length=0):
        return "ok"

    async def safe_api_call(self, prompt, max_tokens, context="", memory=None):
        return "ok"


def _set_time(path: Path, dt: datetime) -> None:
    ts = dt.timestamp()
    os.utime(path, (ts, ts))


def test_auto_monitor_no_trigger(tmp_path, monkeypatch):
    code_root = tmp_path / "app"
    code_root.mkdir()
    for i in range(2):
        f = code_root / f"f{i}.py"
        f.write_text("print('x')")
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    training = log_dir / "symbolic_training_report.md"
    training.write_text("old")
    _set_time(training, datetime.now())

    monkeypatch.setattr(config, "CODE_ROOT", str(code_root))
    monkeypatch.setattr(config, "LOG_DIR", str(log_dir))

    mem = MemoryManager(str(tmp_path / "mem.sqlite"), "dummy", model=None, index=None)
    analyzer = CodeAnalyzer(str(code_root), mem)
    model = DummyModel()

    called = []

    async def fake_run(*a, **k):
        called.append(True)
        return {}

    monkeypatch.setattr("devai.monitor_engine.run_symbolic_training", fake_run)

    async def run():
        return await auto_monitor_cycle(analyzer, mem, model)

    result = asyncio.run(run())
    assert not result["triggered"]
    assert not called


def test_auto_monitor_trigger(tmp_path, monkeypatch):
    code_root = tmp_path / "app"
    code_root.mkdir()
    for i in range(6):
        f = code_root / f"f{i}.py"
        f.write_text("print('x')")
        _set_time(f, datetime.now())
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    training = log_dir / "symbolic_training_report.md"
    training.write_text("old")
    _set_time(training, datetime.now() - timedelta(hours=80))

    monkeypatch.setattr(config, "CODE_ROOT", str(code_root))
    monkeypatch.setattr(config, "LOG_DIR", str(log_dir))

    mem = MemoryManager(str(tmp_path / "mem.sqlite"), "dummy", model=None, index=None)
    analyzer = CodeAnalyzer(str(code_root), mem)
    model = DummyModel()

    for i in range(3):
        mem.save({"type": "err", "memory_type": "erro_reincidente", "content": "x", "metadata": {}})

    called = []

    async def fake_run(*a, **k):
        called.append(True)
        return {}

    monkeypatch.setattr("devai.monitor_engine.run_symbolic_training", fake_run)

    async def run():
        return await auto_monitor_cycle(analyzer, mem, model)

    result = asyncio.run(run())
    assert result["triggered"]
    assert called
