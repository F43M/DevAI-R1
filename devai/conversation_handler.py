from typing import Dict, List, Any
import json
from pathlib import Path
import asyncio

from .memory import np, faiss

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
        self._mtimes: Dict[str, float] = {}
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
                data = json.loads(file.read_text())
                self._mtimes[session_id] = file.stat().st_mtime
                return data
            except Exception:
                return []
        return []

    def _trim_by_tokens(self, session_id: str, max_tokens: int) -> None:
        """Remove oldest messages until total tokens fits within limit."""
        hist = self.history(session_id)
        total = sum(self._estimate_tokens(m.get("content", "")) for m in hist)
        changed = False
        while hist and total > max_tokens:
            msg = hist.pop(0)
            total -= self._estimate_tokens(msg.get("content", ""))
            changed = True
        if changed:
            self.conversation_context[session_id] = hist
            self._save_session(session_id)

    def _save_session(self, session_id: str) -> None:
        try:
            file = self._history_file(session_id)
            file.write_text(
                json.dumps(self.conversation_context.get(session_id, []), indent=2)
            )
            self._mtimes[session_id] = file.stat().st_mtime
        except Exception:
            pass

    def history(self, session_id: str) -> List[Dict[str, str]]:
        file = self._history_file(session_id)
        if file.exists():
            mtime = file.stat().st_mtime
            if (
                session_id not in self.conversation_context
                or self._mtimes.get(session_id) != mtime
            ):
                self.conversation_context[session_id] = self._load_session(session_id)
        else:
            self.conversation_context.setdefault(session_id, [])
        return self.conversation_context[session_id]

    async def _summarize_and_store(self, session_id: str, hist: List[Dict[str, str]]) -> None:
        """Summarize history and persist symbolic memories."""
        try:
            summary = self.summarizer.summarize_conversation(hist, self.memory)
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
        except Exception:
            pass

    def append(self, session_id: str, role: str, content: str) -> None:
        hist = self.history(session_id)
        message_id = len(hist)
        hist.append({"role": role, "content": content})
        self.prune(session_id)
        self._trim_by_tokens(session_id, config.MAX_SESSION_TOKENS)
        if (
            self.memory
            and hasattr(self.memory, "_get_embedding")
            and hasattr(self.memory, "conn")
        ):
            try:
                vec = self.memory._get_embedding(content)
                if vec is not None:
                    if np is not None:
                        blob = np.array(vec, dtype=np.float32).tobytes()
                    else:
                        blob = json.dumps(vec).encode()
                else:
                    blob = None
                cur = self.memory.conn.cursor()
                cur.execute(
                    "INSERT OR REPLACE INTO conversation_embeddings (session_id, message_id, vector) VALUES (?, ?, ?)",
                    (session_id, message_id, blob),
                )
                self.memory.conn.commit()
            except Exception:
                logger.error("failed_to_store_conv_embedding", session=session_id)
        if len(hist) >= self.summary_threshold:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                asyncio.run(self._summarize_and_store(session_id, hist.copy()))
            else:
                asyncio.create_task(self._summarize_and_store(session_id, hist.copy()))
        else:
            self._save_session(session_id)

    def prune(self, session_id: str) -> None:
        """Mantém apenas as últimas mensagens definidas por max_history."""
        hist = self.history(session_id)
        if len(hist) > self.max_history:
            self.conversation_context[session_id] = hist[-self.max_history:]
            self._save_session(session_id)

    def last(self, session_id: str, n: int) -> List[Dict[str, str]]:
        return self.history(session_id)[-n:]

    def search_history(self, session_id: str, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Return messages similar to ``query`` using FAISS search."""
        if (
            not self.memory
            or not hasattr(self.memory, "_get_embedding")
            or not hasattr(self.memory, "conn")
            or faiss is None
        ):
            return []

        cur = self.memory.conn.cursor()
        cur.execute(
            "SELECT message_id, vector FROM conversation_embeddings WHERE session_id = ?",
            (session_id,),
        )
        rows = cur.fetchall()
        if not rows:
            return []

        vecs = []
        ids: list[int] = []
        for mid, blob in rows:
            if blob is None:
                continue
            if np is not None:
                vec = np.frombuffer(blob, dtype=np.float32).reshape(1, -1)
            else:
                vec = json.loads(blob.decode())
            vecs.append(vec)
            ids.append(mid)
        if not vecs:
            return []

        if np is not None:
            dim = vecs[0].shape[1]
            index = faiss.IndexFlatL2(dim)
            index.add(np.concatenate(vecs))
            q = np.array(self.memory._get_embedding(query), dtype=np.float32).reshape(1, -1)
        else:
            dim = len(vecs[0])
            index = faiss.IndexFlatL2(dim)
            index.add(vecs)
            q = self.memory._get_embedding(query)

        distances, indices = index.search(q, min(top_k, len(ids)))
        hist = self.history(session_id)
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < len(ids) and ids[idx] < len(hist):
                msg = hist[ids[idx]]
                results.append(
                    {
                        "message_id": ids[idx],
                        "role": msg.get("role"),
                        "content": msg.get("content"),
                        "score": 1 - float(dist),
                    }
                )
        return results

    def reset(self, session_id: str) -> None:
        self.conversation_context[session_id] = []
        try:
            self._history_file(session_id).unlink()
            self._mtimes.pop(session_id, None)
        except Exception:
            pass
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
