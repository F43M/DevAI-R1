import asyncio
import types
from datetime import datetime

from devai.core import CodeMemoryAI
from devai.conversation_handler import ConversationHandler


class DummyModel:
    async def safe_api_call(self, messages, max_tokens, context="", memory=None, temperature=0.0):
        text = messages[-1]["content"].lower()
        if "forma segura" in text:
            return "Para fazer de forma segura use try/except"
        if "classe" in text:
            return "Você aplica o decorator nos métodos da classe"
        return "Um decorator é criado com def decorator(func)"


def test_multi_turn_cohesion():
    ai = object.__new__(CodeMemoryAI)
    ai.memory = types.SimpleNamespace(search=lambda q, top_k=5, level=None: [])
    ai.analyzer = types.SimpleNamespace(
        graph_summary=lambda: "",
        code_chunks={},
        last_analysis_time=datetime.now(),
    )
    ai.tasks = types.SimpleNamespace(last_actions=lambda: [])
    ai.ai_model = DummyModel()
    ai.conv_handler = ConversationHandler()
    ai.reason_stack = []
    ai.double_check = False

    async def run():
        r1 = await ai.generate_response("Como funciona um decorator em Python?")
        r2 = await ai.generate_response("E como aplico isso em uma classe?")
        r3 = await ai.generate_response("E como faço isso de forma segura?")
        return r1, r2, r3

    r1, r2, r3 = asyncio.run(run())
    assert "def" in r1.lower()
    assert "decorator" in r2.lower()
    assert "class" in r2.lower() or "método" in r2.lower()
    assert any(term in r3.lower() for term in ["seguro", "try", "except", "validação"])
