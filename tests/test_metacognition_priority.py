import asyncio
from pathlib import Path
import json
from devai.learning_engine import LearningEngine
from devai.analyzer import CodeAnalyzer
from devai.memory import MemoryManager
from devai.config import config
from devai.core import CodeMemoryAI
from devai.conversation_handler import ConversationHandler
import types
from datetime import datetime


class DummyModel:
    async def safe_api_call(self, prompt, max_tokens, context="", memory=None):
        return "ok"


def test_priority_from_score_map(tmp_path, monkeypatch):
    db = tmp_path / "mem.sqlite"
    mem = MemoryManager(str(db), "dummy", model=None, index=None)
    analyzer = CodeAnalyzer(str(tmp_path), mem)

    f1 = tmp_path / "a.py"
    f2 = tmp_path / "b.py"
    f1.write_text("def a(): pass")
    f2.write_text("def b(): pass")
    analyzer.code_chunks = {
        "a": {"name": "a", "file": str(f1), "code": "def a(): pass", "hash": "h1"},
        "b": {"name": "b", "file": str(f2), "code": "def b(): pass", "hash": "h2"},
    }
    score_file = Path("devai/meta/score_map.json")
    score_file.parent.mkdir(parents=True, exist_ok=True)
    score_file.write_text(json.dumps({str(f1): -3}))

    engine = LearningEngine(analyzer, mem, DummyModel(), rate_limit=2)

    order = []

    async def fake_call(prompt, max_len=800):
        if "def a()" in prompt:
            order.append("a")
        elif "def b()" in prompt:
            order.append("b")
        return "ok"

    engine._rate_limited_call = fake_call
    asyncio.run(engine.learn_from_codebase())
    assert order[0] == "a"

    cur = mem.conn.cursor()
    cur.execute("SELECT metadata FROM memory WHERE memory_type='reflection'")
    row = cur.fetchone()
    assert row and json.loads(row[0])["file"] == str(f1)


def _setup_ai(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "LOG_DIR", str(tmp_path / "logs"))
    ai = object.__new__(CodeMemoryAI)
    ai.memory = types.SimpleNamespace(search=lambda *a, **k: [], save=lambda *a, **k: None, conn=MemoryManager(str(tmp_path/"m.sqlite"), "d", model=None, index=None).conn)
    ai.analyzer = types.SimpleNamespace(code_chunks={}, last_analysis_time=datetime.now(), graph_summary=lambda: "")
    ai.tasks = types.SimpleNamespace(last_actions=lambda: [])
    ai.log_monitor = types.SimpleNamespace()
    ai.conv_handler = ConversationHandler(memory=ai.memory)
    ai.learning_engine = types.SimpleNamespace()
    ai.ai_model = DummyModel()
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

def test_summary_endpoint(monkeypatch, tmp_path):
    path = Path("devai/meta/score_map.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"x.py": -2, "y.py": 1}
    path.write_text(json.dumps(data))

    ai, routes = _setup_ai(monkeypatch, tmp_path)
    fn = routes["/metacognition/summary"]
    result = asyncio.run(fn())
    assert result["critical"] == {"x.py": -2}

