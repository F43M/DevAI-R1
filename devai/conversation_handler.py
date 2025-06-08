from typing import Dict, List, Any

from .config import logger
from .dialog_summarizer import DialogSummarizer


class ConversationHandler:
    """Gerencia o contexto de conversa multi-turno."""

    def __init__(self, max_history: int = 10, summary_threshold: int = 20, memory: Any = None):
        self.conversation_context: Dict[str, List[Dict[str, str]]] = {}
        self.max_history = max_history
        self.summary_threshold = summary_threshold
        self.memory = memory
        self.summarizer = DialogSummarizer()

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        return len(text.split())

    def history(self, session_id: str) -> List[Dict[str, str]]:
        return self.conversation_context.setdefault(session_id, [])

    def append(self, session_id: str, role: str, content: str) -> None:
        hist = self.history(session_id)
        hist.append({"role": role, "content": content})
        self.prune(session_id)
        if len(hist) >= self.summary_threshold:
            summary = self.summarizer.summarize_conversation(hist)
            if summary and self.memory:
                for item in summary:
                    try:
                        self.memory.save(
                            {
                                "type": "dialog",
                                "memory_type": "dialog_summary",
                                "content": item.get("content", ""),
                                "metadata": {"tag": item.get("tag"), "origin": "dialog_summary"},
                                "tags": [item.get("tag", "")],
                            }
                        )
                    except Exception:
                        pass
            self.conversation_context[session_id] = hist[-self.max_history:]

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
