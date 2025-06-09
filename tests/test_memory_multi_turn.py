import asyncio
from devai.conversation_handler import ConversationHandler
from devai.config import config


class DummyMemory:
    def __init__(self):
        self.saved = []

    def save(self, item):
        self.saved.append(item)


def test_memory_multi_turn(monkeypatch):
    mem = DummyMemory()
    handler = ConversationHandler(max_history=5, summary_threshold=2, memory=mem)
    monkeypatch.setattr(config, "MAX_SESSION_TOKENS", 100)

    async def run():
        handler.append("s", "user", "Prefiro usar funcoes puras")
        handler.append("s", "assistant", "Ok")
        assert mem.saved == []
        await asyncio.sleep(0.01)
        assert mem.saved

    asyncio.run(run())
    assert mem.saved[0]["memory_type"] == "dialog_summary"
