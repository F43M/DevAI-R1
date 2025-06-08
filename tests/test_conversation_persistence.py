import os
from devai.conversation_handler import ConversationHandler


def test_history_persistence(tmp_path):
    handler = ConversationHandler(history_dir=tmp_path)
    handler.append("s", "user", "hi")
    handler.append("s", "assistant", "hello")
    new_handler = ConversationHandler(history_dir=tmp_path)
    hist = new_handler.history("s")
    assert len(hist) == 2
    assert hist[0]["content"] == "hi"


def test_history_cleanup(tmp_path):
    handler = ConversationHandler(history_dir=tmp_path)
    handler.append("x", "user", "a")
    handler.clear_session("x")
    new_handler = ConversationHandler(history_dir=tmp_path)
    assert new_handler.history("x") == []
