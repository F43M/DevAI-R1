from typing import Dict, List

from .config import logger


class ConversationHandler:
    """Gerencia o contexto de conversa multi-turno."""

    def __init__(self, max_history: int = 10):
        self.conversation_context: Dict[str, List[Dict[str, str]]] = {}
        self.max_history = max_history

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        return len(text.split())

    def history(self, session_id: str) -> List[Dict[str, str]]:
        return self.conversation_context.setdefault(session_id, [])

    def append(self, session_id: str, role: str, content: str) -> None:
        hist = self.history(session_id)
        hist.append({"role": role, "content": content})
        self.prune(session_id)

    def prune(self, session_id: str) -> None:
        """Mantém apenas as últimas mensagens definidas por max_history."""
        hist = self.history(session_id)
        if len(hist) > self.max_history:
            self.conversation_context[session_id] = hist[-self.max_history:]

    def last(self, session_id: str, n: int) -> List[Dict[str, str]]:
        return self.history(session_id)[-n:]

    def reset(self, session_id: str) -> None:
        self.conversation_context[session_id] = []
        logger.info("messages_cleared", session=session_id)
