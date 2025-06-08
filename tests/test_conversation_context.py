import asyncio
import types
from datetime import datetime
from devai.core import CodeMemoryAI
from devai.conversation_handler import ConversationHandler

class DummyModel:
    async def safe_api_call(self, prompt, max_tokens, context="", memory=None, temperature=0.0):
        return "ok"


def test_conversation_context():
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
    ai.reason_stack = []
    ai.double_check = False

    async def run():
        await ai.generate_response("Como funciona o login?", session_id="sess1")
        await ai.generate_response("E o logout?", session_id="sess1")
        return ai.conv_handler.conversation_context["sess1"]

    history = asyncio.run(run())
    assert history[-2]["content"] == "E o logout?"
