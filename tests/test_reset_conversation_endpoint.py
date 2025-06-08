import asyncio
import types
from datetime import datetime
from devai.core import CodeMemoryAI
from devai.conversation_handler import ConversationHandler

class DummyModel:
    async def safe_api_call(self, prompt, max_tokens, context="", memory=None, temperature=0.0):
        return "ok"


def test_reset_conversation_endpoint(monkeypatch):
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

    ai.conv_handler.append("s", "user", "hi")
    ai.conv_handler.symbolic_memories.append("pref")

    fn = record['/reset_conversation']
    result = asyncio.run(fn(session_id='s'))

    assert result['status'] == 'reset'
    assert ai.conv_handler.history('s') == []
    assert ai.conv_handler.symbolic_memories == ['pref']
