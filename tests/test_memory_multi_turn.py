from devai.conversation_handler import ConversationHandler


class DummyMemory:
    def __init__(self):
        self.saved = []

    def save(self, item):
        self.saved.append(item)


def test_memory_multi_turn():
    mem = DummyMemory()
    handler = ConversationHandler(max_history=5, summary_threshold=2, memory=mem)
    handler.append("s", "user", "Prefiro usar funcoes puras")
    handler.append("s", "assistant", "Ok")
    assert mem.saved
    assert mem.saved[0]["memory_type"] == "dialog_summary"
