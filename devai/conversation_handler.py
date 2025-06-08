from typing import Dict, List, Any
import json
from pathlib import Path

from .config import logger, config
from .dialog_summarizer import DialogSummarizer


class ConversationHandler:
    """Gerencia o contexto de conversa multi-turno."""

    def __init__(
        self,
        max_history: int = 10,
        summary_threshold: int = 20,
        memory: Any = None,
        history_dir: str | None = None,
    ) -> None:
        self.conversation_context: Dict[str, List[Dict[str, str]]] = {}
        self.max_history = max_history
        self.summary_threshold = summary_threshold
        self.memory = memory
        self.summarizer = DialogSummarizer()
        self.symbolic_memories: List[str] = []
        self.history_dir = Path(history_dir or Path(config.LOG_DIR) / "sessions")
        self.history_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        return len(text.split())

    def _history_file(self, session_id: str) -> Path:
        return self.history_dir / f"{session_id}.json"

    def _load_session(self, session_id: str) -> List[Dict[str, str]]:
        file = self._history_file(session_id)
        if file.exists():
            try:
                return json.loads(file.read_text())
            except Exception:
                return []
        return []

    def _save_session(self, session_id: str) -> None:
        try:
            self._history_file(session_id).write_text(
                json.dumps(self.conversation_context.get(session_id, []), indent=2)
            )
        except Exception:
            pass

    def history(self, session_id: str) -> List[Dict[str, str]]:
        if session_id not in self.conversation_context:
            self.conversation_context[session_id] = self._load_session(session_id)
        return self.conversation_context[session_id]

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
        self._save_session(session_id)

    def prune(self, session_id: str) -> None:
        """Mantém apenas as últimas mensagens definidas por max_history."""
        hist = self.history(session_id)
        if len(hist) > self.max_history:
            self.conversation_context[session_id] = hist[-self.max_history:]
            self._save_session(session_id)

    def last(self, session_id: str, n: int) -> List[Dict[str, str]]:
        return self.history(session_id)[-n:]

    def reset(self, session_id: str) -> None:
        self.conversation_context[session_id] = []
        self._save_session(session_id)
        logger.info("messages_cleared", session=session_id)

    def clear_session(self, session_id: str = "default") -> None:
        """Clear conversation history and temporary symbolic memories."""
        self.conversation_context[session_id] = []
        try:
            self._history_file(session_id).unlink()
        except Exception:
            pass
        self.symbolic_memories = []
        logger.info("session_reset", session=session_id)
