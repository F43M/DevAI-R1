from datetime import datetime
import types
from devai.core import CodeMemoryAI
from devai.conversation_handler import ConversationHandler

class DummyModel:
    async def safe_api_call(self, prompt, max_tokens, context="", memory=None, temperature=0.0):
        return "ok"


def test_history_limit():
    ai = object.__new__(CodeMemoryAI)
    ai.memory = types.SimpleNamespace(search=lambda *a, **k: [])
    ai.analyzer = types.SimpleNamespace(graph_summary=lambda: "", code_chunks={}, last_analysis_time=datetime.now())
    ai.tasks = types.SimpleNamespace(last_actions=lambda: [])
    ai.ai_model = DummyModel()
    ai.conv_handler = ConversationHandler(memory=ai.memory)
    ai.reason_stack = []
    ai.double_check = False

    for i in range(12):
        ai.conv_handler.append("s", "user", f"q{i}")
        ai.conv_handler.append("s", "assistant", f"a{i}")

    history = ai._build_history_messages("s")
    assert len(history) <= ai.conv_handler.max_history
