from devai.conversation_handler import ConversationHandler

def test_reset_session_clears_conversation():
    handler = ConversationHandler()
    handler.conversation_context["default"] = [{"role": "user", "content": "Oi"}]
    handler.clear_session()
    assert handler.conversation_context["default"] == []
