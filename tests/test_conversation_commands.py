import asyncio
import types
from datetime import datetime

from devai.core import CodeMemoryAI
from devai.conversation_handler import ConversationHandler
from devai.config import config


class DummyModel:
    async def safe_api_call(self, prompt, max_tokens, context="", memory=None, temperature=0.0):
        return "ok"


def _setup_ai():
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
    ai.double_check = False
    return ai


def test_messages_followed_by_command():
    ai = _setup_ai()

    async def run():
        await ai.generate_response("Primeira mensagem aleatoria")
        await ai.generate_response("Segunda mensagem qualquer")
        before = ai.conv_handler.history("default").copy()
        cmd_resp = await ai.generate_response("/resetar")
        after = ai.conv_handler.history("default")
        return before, after, cmd_resp

    hist_before, hist_after, cmd_resp = asyncio.run(run())
    assert len(hist_before) == 4
    assert hist_before[0]["content"] == "Primeira mensagem aleatoria"
    assert "resetada" in cmd_resp
    assert hist_after == []


def test_history_order_and_trim(monkeypatch):
    handler = ConversationHandler()
    monkeypatch.setattr(config, "MAX_SESSION_TOKENS", 7)

    handler.append("s", "user", "a b c d e")
    handler.append("s", "assistant", "f g")
    assert [m["content"] for m in handler.history("s")] == ["a b c d e", "f g"]

    handler.append("s", "user", "h i j k l")
    assert [m["content"] for m in handler.history("s")] == ["f g", "h i j k l"]

    handler.append("s", "assistant", "m n")
    hist = handler.history("s")
    assert [m["content"] for m in hist] == ["h i j k l", "m n"]
    total = sum(handler._estimate_tokens(m["content"]) for m in hist)
    assert total <= 7
