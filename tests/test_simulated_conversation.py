import asyncio
import types
from datetime import datetime
from devai.core import CodeMemoryAI
from devai.conversation_handler import ConversationHandler


class DummyModel:
    async def safe_api_call(self, prompt, max_tokens, context="", memory=None):
        return "ok"


def test_simulated_conversation(monkeypatch):
    ai = object.__new__(CodeMemoryAI)
    ai.memory = types.SimpleNamespace(search=lambda q, top_k=5, level=None: [])
    ai.analyzer = types.SimpleNamespace(
        graph_summary=lambda: "",
        code_chunks={},
        last_analysis_time=datetime.now(),
    )
    ai.tasks = types.SimpleNamespace(last_actions=lambda: [])
    ai.ai_model = DummyModel()
    ai.conv_handler = ConversationHandler(memory=ai.memory)
    ai.conversation_history = []
    ai.reason_stack = []

    async def run():
        first = await ai.generate_response("Como funciona o login?")
        second = await ai.generate_response("E o logout?")
        return first, second

    resp1, resp2 = asyncio.run(run())
    assert resp1 == "ok"
    assert ai.conversation_history[-2]["content"] == "E o logout?"
