import json
import asyncio
import types
import sys
from types import SimpleNamespace
from datetime import datetime

sys.modules.setdefault('transformers', SimpleNamespace())
sys.modules.setdefault('devai.rlhf', SimpleNamespace(RLFineTuner=lambda *a, **k: None))

from devai.conversation_handler import ConversationHandler
from devai.core import CodeMemoryAI
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


def test_history_auto_saved(tmp_path):
    handler = ConversationHandler(history_dir=tmp_path)
    handler.append("s", "user", "hi")
    handler.append("s", "assistant", "hello")
    file = handler._history_file("s")
    assert file.exists()
    data = json.loads(file.read_text())
    assert data == handler.history("s")


def test_history_limit_enforced(tmp_path):
    handler = ConversationHandler(history_dir=tmp_path)
    for i in range(12):
        handler.append("s", "user", str(i))
    assert len(handler.history("s")) == handler.max_history
    data = json.loads(handler._history_file("s").read_text())
    assert len(data) == handler.max_history
    assert data[0]["content"] == "2"
    new_handler = ConversationHandler(history_dir=tmp_path)
    assert len(new_handler.history("s")) == handler.max_history


def test_reset_conversation_endpoint(monkeypatch, tmp_path):
    ai, record = _setup_ai(monkeypatch, tmp_path)
    ai.conv_handler.append("s", "user", "hi")
    file = ai.conv_handler._history_file("s")
    assert file.exists()
    fn = record["/reset_conversation"]
    result = asyncio.run(fn(session_id="s"))
    assert result["status"] == "reset"
    assert ai.conv_handler.history("s") == []
    assert not file.exists()
