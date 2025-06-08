import asyncio
import types
from datetime import datetime
from devai.core import CodeMemoryAI
from devai.conversation_handler import ConversationHandler

class DummyModel:
    async def safe_api_call(self, prompt, max_tokens, context="", memory=None, temperature=0.0):
        return "ok"

def test_session_context_endpoint(monkeypatch):
    ai = object.__new__(CodeMemoryAI)
    ai.memory = types.SimpleNamespace(
        search=lambda q, top_k=5, level=None, memory_type=None: [
            {"content": "#preferencia_usuario: Usa função pura", "metadata": {"tag": "#preferencia_usuario"}}
        ]
    )
    ai.analyzer = types.SimpleNamespace(
        graph_summary=lambda: "grafo", code_chunks={}, last_analysis_time=datetime.now()
    )
    ai.tasks = types.SimpleNamespace(last_actions=lambda: [])
    ai.ai_model = DummyModel()
    ai.conv_handler = ConversationHandler(memory=ai.memory)
    ai.reason_stack = []
    ai.double_check = False
    ai.last_context = {"context_blocks": {"logs": "log"}}

    # register fake app
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

    # simulate conversation
    ai.conv_handler.append("s", "user", "Como funciona?")
    ai.conv_handler.append("s", "assistant", "Explicação")
    ai.conv_handler.append("s", "user", "Outra dúvida")

    fn = record['/session/context']
    result = asyncio.run(fn(session_id='s'))
    assert "symbolic_memories" in result
    assert len(result["history_preview"]) > 1
    assert "#preferencia_usuario" in result["symbolic_memories"][0]


def test_session_context_empty(monkeypatch):
    ai = object.__new__(CodeMemoryAI)
    ai.memory = types.SimpleNamespace(search=lambda *a, **k: [])
    ai.analyzer = types.SimpleNamespace(
        graph_summary=lambda: "", code_chunks={}, last_analysis_time=datetime.now()
    )
    ai.tasks = types.SimpleNamespace(last_actions=lambda: [])
    ai.ai_model = DummyModel()
    ai.conv_handler = ConversationHandler(memory=ai.memory)
    ai.reason_stack = []
    ai.double_check = False
    ai.last_context = {}

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

    fn = record['/session/context']
    result = asyncio.run(fn(session_id='new'))
    assert "error" in result

