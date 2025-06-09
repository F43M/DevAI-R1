from datetime import datetime, timedelta
from pathlib import Path
import types
import asyncio
import os

from devai.monitor_engine import auto_monitor_cycle
from devai.analyzer import CodeAnalyzer
from devai.memory import MemoryManager
from devai.core import CodeMemoryAI
from devai.conversation_handler import ConversationHandler
from devai.config import config

class DummyModel:
    async def generate(self, prompt, max_length=0):
        return "ok"

    async def safe_api_call(self, prompt, max_tokens, context="", memory=None):
        return "ok"


def _set_time(path: Path, dt: datetime) -> None:
    ts = dt.timestamp()
    os.utime(path, (ts, ts))

def _setup_ai(memory, analyzer, model):
    ai = object.__new__(CodeMemoryAI)
    ai.memory = memory
    ai.analyzer = analyzer
    ai.ai_model = model
    ai.tasks = types.SimpleNamespace(last_actions=lambda: [])
    ai.log_monitor = types.SimpleNamespace()
    ai.conv_handler = ConversationHandler(memory=memory)
    ai.reason_stack = []
    ai.double_check = False
    record = {}
    app = types.SimpleNamespace()
    def fake_get(path):
        def decorator(fn):
            record[path] = fn
            return fn
        return decorator
    app.get = app.post = fake_get
    app.mount = lambda *a, **k: None
    ai.app = app
    CodeMemoryAI._setup_api_routes(ai)
    return ai, record


def test_monitor_history_endpoint(tmp_path, monkeypatch):
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

    async def fake_run(*a, **k):
        return {"report": "", "data": {"new_rules": 1}}
    monkeypatch.setattr("devai.monitor_engine.run_symbolic_training", fake_run)

    asyncio.run(auto_monitor_cycle(analyzer, mem, model))

    ai, routes = _setup_ai(mem, analyzer, model)
    hist_fn = routes["/monitor/history"]
    rows = asyncio.run(hist_fn())
    assert rows
    assert rows[0]["new_rules"] == 1

    cur = mem.conn.cursor()
    cur.execute("SELECT COUNT(*) FROM monitoring_history")
    assert cur.fetchone()[0] == 1

