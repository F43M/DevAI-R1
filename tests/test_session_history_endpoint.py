import asyncio
import types
from datetime import datetime

from devai.core import CodeMemoryAI
from devai.conversation_handler import ConversationHandler
from devai.config import config


class DummyModel:
    async def safe_api_call(self, prompt, max_tokens, context="", memory=None, temperature=0.0):
        return "ok"


def _setup_ai(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "LOG_DIR", str(tmp_path))
    ai = object.__new__(CodeMemoryAI)
    ai.memory = types.SimpleNamespace(search=lambda *a, **k: [])
    ai.analyzer = types.SimpleNamespace(graph_summary=lambda: "", code_chunks={}, last_analysis_time=datetime.now())
    ai.tasks = types.SimpleNamespace(last_actions=lambda: [])
    ai.ai_model = DummyModel()
    ai.conv_handler = ConversationHandler(memory=ai.memory)
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


def test_session_history_endpoint(monkeypatch, tmp_path):
    ai, record = _setup_ai(monkeypatch, tmp_path)
    ai.conv_handler.append("s", "user", "hi")
    ai.conv_handler.append("s", "assistant", "hello")
    fn = record["/session/history"]
    hist = asyncio.run(fn(session_id="s"))
    assert len(hist) == 2
    assert hist[0]["role"] == "user"

    fn2 = record["/history"]
    hist2 = asyncio.run(fn2(session_id="s"))
    assert hist2 == hist


def test_session_reset_isolated(monkeypatch, tmp_path):
    ai, record = _setup_ai(monkeypatch, tmp_path)
    ai.conv_handler.append("a", "user", "1")
    ai.conv_handler.append("b", "user", "2")
    reset_fn = record["/session/reset"]
    asyncio.run(reset_fn(session_id="a"))
    assert ai.conv_handler.history("a") == []
    assert ai.conv_handler.history("b")
