import json
import os
import sqlite3
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta

try:
    import numpy as np
except Exception:  # pragma: no cover - optional dependency
    np = None

from .config import config, logger

try:
    from sentence_transformers import SentenceTransformer
    import faiss
except Exception:  # pragma: no cover - optional heavy deps
    SentenceTransformer = None
    faiss = None


class MemoryManager:
    """Persistent memory with vector search."""

    def __init__(
        self,
        db_file: str,
        embedding_model: str,
        model: Optional[Any] = None,
        index: Optional[Any] = None,
    ):
        self.conn = sqlite3.connect(db_file)
        self._init_db()
        if model is None:
            if SentenceTransformer is None:
                raise RuntimeError("sentence_transformers not available")
            model = SentenceTransformer(embedding_model)
        self.embedding_model = model
        self.dimension = self.embedding_model.get_sentence_embedding_dimension()
        if index is None:
            if faiss is None:
                raise RuntimeError("faiss not available")
            index = faiss.IndexFlatL2(self.dimension)
        self.index = index
        self.indexed_ids: List[int] = []
        if index is None:
            self._load_index()

    def _init_db(self):
        cursor = self.conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                type TEXT NOT NULL,
                content TEXT NOT NULL,
                metadata TEXT NOT NULL,
                embedding BLOB,
                feedback_score INTEGER DEFAULT 0,
                last_accessed TEXT,
                access_count INTEGER DEFAULT 0,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS semantic_relations (
                source_id INTEGER,
                target_id INTEGER,
                relation_type TEXT,
                strength REAL,
                PRIMARY KEY (source_id, target_id),
                FOREIGN KEY (source_id) REFERENCES memory (id),
                FOREIGN KEY (target_id) REFERENCES memory (id)
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS tags (
                memory_id INTEGER,
                tag TEXT,
                PRIMARY KEY (memory_id, tag),
                FOREIGN KEY (memory_id) REFERENCES memory (id)
            )
            """
        )
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_memory_feedback ON memory(feedback_score)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_memory_access ON memory(access_count)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_tags_tag ON tags(tag)")
        self.conn.commit()

    def _load_index(self):
        if not (faiss and os.path.exists(config.INDEX_FILE) and os.path.exists(config.INDEX_IDS_FILE)):
            return
        self.index = faiss.read_index(config.INDEX_FILE)
        with open(config.INDEX_IDS_FILE, "r") as f:
            self.indexed_ids = json.load(f)
        logger.info("Índice de memória carregado do disco", items=len(self.indexed_ids))

    def _persist_index(self):
        if not faiss:
            return
        faiss.write_index(self.index, config.INDEX_FILE)
        with open(config.INDEX_IDS_FILE, "w") as f:
            json.dump(self.indexed_ids, f)

    def _rebuild_index(self):
        if not faiss:
            return
        self.index = faiss.IndexFlatL2(self.dimension)
        self.indexed_ids = []
        cursor = self.conn.cursor()
        cursor.execute("SELECT id, embedding FROM memory WHERE embedding IS NOT NULL")
        embeddings = []
        for row in cursor.fetchall():
            self.indexed_ids.append(row[0])
            if np is not None:
                embeddings.append(np.frombuffer(row[1], dtype=np.float32).reshape(1, -1))
        if embeddings and np is not None:
            self.index.add(np.concatenate(embeddings))
        self._persist_index()

    def save(self, entry: Dict, update_feedback: bool = False):
        entry["metadata"] = json.dumps(entry.get("metadata", {}))
        content = self._generate_content_for_embedding(entry)
        embedding_vec = self.embedding_model.encode(content)
        if np is not None:
            embedding = np.array(embedding_vec, dtype=np.float32).tobytes()
        else:
            embedding = json.dumps(embedding_vec).encode()
        cursor = self.conn.cursor()
        if update_feedback and "id" in entry:
            cursor.execute(
                "UPDATE memory SET feedback_score = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (entry.get("feedback_score", 0), entry["id"]),
            )
        else:
            cursor.execute(
                """INSERT INTO memory
                (type, content, metadata, embedding, feedback_score)
                VALUES (?, ?, ?, ?, ?)""",
                (
                    entry.get("type", "generic"),
                    entry.get("content", ""),
                    entry["metadata"],
                    embedding,
                    entry.get("feedback_score", 0),
                ),
            )
            entry["id"] = cursor.lastrowid
            vec = self.embedding_model.encode(content)
            if np is not None:
                self.index.add(np.array([vec], dtype=np.float32))
            else:
                self.index.add([vec])
            self.indexed_ids.append(entry["id"])
            self._persist_index()
        self.conn.commit()
        logger.info("Memória salva" if not update_feedback else "Feedback atualizado", entry_type=entry.get("type"))

    def _generate_content_for_embedding(self, entry: Dict) -> str:
        parts = [entry.get("content", ""), entry.get("type", ""), " ".join(entry.get("tags", []))]
        if "metadata" in entry:
            if isinstance(entry["metadata"], dict):
                parts.append(json.dumps(entry["metadata"]))
            else:
                parts.append(str(entry["metadata"]))
        return " ".join(parts)

    def search(self, query: str, top_k: int = 5, min_score: float = 0.7) -> List[Dict]:
        query_embedding = self.embedding_model.encode(query)
        if np is not None:
            q = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
        else:
            q = query_embedding
        distances, indices = self.index.search(q, top_k)
        results = []
        cursor = self.conn.cursor()
        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(self.indexed_ids) and distance <= (1 - min_score):
                memory_id = self.indexed_ids[idx]
                cursor.execute(
                    """
                    SELECT m.*, GROUP_CONCAT(t.tag, ', ') as tags
                    FROM memory m
                    LEFT JOIN tags t ON m.id = t.memory_id
                    WHERE m.id = ?
                    GROUP BY m.id
                    """,
                    (memory_id,),
                )
                row = cursor.fetchone()
                if row:
                    results.append(
                        {
                            "id": row[0],
                            "type": row[1],
                            "content": row[2],
                            "metadata": json.loads(row[3]),
                            "feedback_score": row[5],
                            "tags": row[9].split(", ") if row[9] else [],
                            "similarity_score": 1 - distance,
                            "last_accessed": row[6],
                            "access_count": row[7],
                        }
                    )
                    cursor.execute(
                        "UPDATE memory SET last_accessed = CURRENT_TIMESTAMP, access_count = access_count + 1 WHERE id = ?",
                        (memory_id,),
                    )
        self.conn.commit()
        logger.info("Busca de memória realizada", query=query, results=len(results))
        return sorted(results, key=lambda x: x["similarity_score"], reverse=True)

    def add_semantic_relation(self, source_id: int, target_id: int, relation_type: str, strength: float = 1.0):
        cursor = self.conn.cursor()
        cursor.execute(
            """INSERT OR REPLACE INTO semantic_relations (source_id, target_id, relation_type, strength)
            VALUES (?, ?, ?, ?)""",
            (source_id, target_id, relation_type, strength),
        )
        self.conn.commit()
        logger.info("Relação semântica adicionada", source=source_id, target=target_id, type=relation_type)

    def get_related_memories(self, memory_id: int, relation_type: Optional[str] = None) -> List[Dict]:
        cursor = self.conn.cursor()
        query = """
            SELECT m.*, r.relation_type, r.strength, GROUP_CONCAT(t.tag, ', ') as tags
            FROM memory m
            JOIN semantic_relations r ON m.id = r.target_id
            LEFT JOIN tags t ON m.id = t.memory_id
            WHERE r.source_id = ?
        """
        params = [memory_id]
        if relation_type:
            query += " AND r.relation_type = ?"
            params.append(relation_type)
        query += " GROUP BY m.id"
        cursor.execute(query, params)
        results = []
        for row in cursor.fetchall():
            results.append(
                {
                    "id": row[0],
                    "type": row[1],
                    "content": row[2],
                    "metadata": json.loads(row[3]),
                    "relation_type": row[9],
                    "strength": row[10],
                    "tags": row[11].split(", ") if row[11] else [],
                }
            )
        return results

    def cleanup(self, max_age_days: int = 30, min_access_count: int = 0):
        cutoff = (datetime.now() - timedelta(days=max_age_days)).isoformat()
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT id FROM memory WHERE created_at < ? AND access_count <= ?",
            (cutoff, min_access_count),
        )
        ids = [row[0] for row in cursor.fetchall()]
        if not ids:
            return
        placeholders = ",".join(["?"] * len(ids))
        cursor.execute(f"DELETE FROM memory WHERE id IN ({placeholders})", ids)
        cursor.execute(f"DELETE FROM tags WHERE memory_id IN ({placeholders})", ids)
        cursor.execute(
            f"DELETE FROM semantic_relations WHERE source_id IN ({placeholders}) OR target_id IN ({placeholders})",
            ids * 2,
        )
        self.conn.commit()
        self._rebuild_index()
        logger.info("Limpeza de memória executada", removed=len(ids))

    def record_feedback(self, memory_id: int, is_positive: bool):
        score_change = 1 if is_positive else -1
        cursor = self.conn.cursor()
        cursor.execute(
            "UPDATE memory SET feedback_score = feedback_score + ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (score_change, memory_id),
        )
        self.conn.commit()
        logger.info("Feedback registrado", memory_id=memory_id, positive=is_positive)
