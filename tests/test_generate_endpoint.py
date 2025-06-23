import asyncio
import types
from datetime import datetime

from devai.core import CodeMemoryAI
from devai.conversation_handler import ConversationHandler
from devai.config import config
from devai import generation_chain


def fake_long_code(prompt, model, context=None, window_size=50, overlap_ratio=0.1):
    lines = [f"line{i}" for i in range(120)]
    return "\n".join(lines)


class DummyModel:
    async def safe_api_call(self, prompt, max_tokens, context="", memory=None, temperature=0.7):
        return "ok"


def test_generate_chunks(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "LOG_DIR", str(tmp_path))
    monkeypatch.setattr(generation_chain, "generate_long_code", fake_long_code)
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

    res1 = asyncio.run(record['/generate'](prompt='anything', session_id='s', cont=False))
    assert not res1['done']
    assert res1['chunk'].startswith('line0')

    res2 = asyncio.run(record['/generate'](prompt='', session_id='s', cont=True))
    assert res2['done']
    assert 'line50' in res2['chunk']
