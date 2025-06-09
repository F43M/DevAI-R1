import asyncio
import types
from datetime import datetime

from devai.core import CodeMemoryAI
from devai.conversation_handler import ConversationHandler
from devai.config import config

class DummyModel:
    async def safe_api_call(self, prompt, max_tokens, context="", memory=None, temperature=0.0):
        return "ok"


def test_analyze_restores_history(monkeypatch, tmp_path):
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
    def fake_post(path):
        def decorator(fn):
            record[path] = fn
            return fn
        return decorator
    app.get = app.post = fake_post
    app.mount = lambda *a, **k: None
    ai.app = app

    CodeMemoryAI._setup_api_routes(ai)

    fn = record['/analyze']
    asyncio.run(fn(query='tell me something', session_id='sess'))
    first_file = ai.conv_handler._history_file('sess')
    assert first_file.exists()
    assert len(ai.conv_handler.history('sess')) == 2

    ai.conv_handler.conversation_context.pop('sess')

    asyncio.run(fn(query='another question please', session_id='sess'))
    hist = ai.conv_handler.history('sess')
    assert len(hist) == 4

