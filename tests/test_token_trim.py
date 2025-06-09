import asyncio
from devai.conversation_handler import ConversationHandler
from devai.config import config

class DummyMemory:
    def save(self, item):
        pass

def test_token_trim(monkeypatch):
    handler = ConversationHandler()
    monkeypatch.setattr(config, "MAX_SESSION_TOKENS", 7)
    handler.append("s", "user", "a b c d e")
    handler.append("s", "assistant", "f g")
    handler.append("s", "user", "h i j k l")
    hist = handler.history("s")
    total = sum(handler._estimate_tokens(m["content"]) for m in hist)
    assert total <= 7
    assert hist[0]["content"] == "f g"
