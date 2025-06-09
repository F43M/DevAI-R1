import json
import os
import sqlite3
from typing import List, Dict, Any, Optional
from pathlib import Path
from collections import OrderedDict
from datetime import datetime, timedelta

MEMORY_TYPES = [
    "explicacao",
    "bug corrigido",
    "feedback negativo",
    "refatoracao aprovada",
    "boas_praticas",
    "risco_oculto",
    "erro_estudado",
    "padrao_positivo",
    "aprendizado_importado",
    "regra do usuario",
    "licao aprendida",
    "refatoracao_sugerida",
    "risco_reincidente",
    "ponto_critico",
    "resposta_cortada",
]

try:
    import numpy as np
except Exception:  # pragma: no cover - optional dependency
    np = None

from .config import config, logger
from .feedback import FeedbackDB

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
        cache_size: int = 128,
    ):
        self.conn = sqlite3.connect(db_file)
        self.feedback_db = FeedbackDB()
        self._init_db()
        if model is None:
            if SentenceTransformer is None:
                logger.warning(
                    "sentence_transformers não instalado; busca vetorial desabilitada"
                )
                self.embedding_model = None
            else:
                self.embedding_model = SentenceTransformer(embedding_model)
        else:
            self.embedding_model = model

        self.dimension = (
            self.embedding_model.get_sentence_embedding_dimension()
            if self.embedding_model
            else 0
        )

        self.index = index
        self.indexed_ids: List[int] = []
        self.embedding_cache: "OrderedDict[str, Any]" = OrderedDict()
        self.embedding_cache_size = cache_size

        # Attempt to load an existing index from disk before creating a new one
        self.load_index()

        if self.index is None:
            if self.embedding_model is not None and faiss is not None:
                self.index = faiss.IndexFlatL2(self.dimension)
            else:
                if self.embedding_model is not None and faiss is None:
                    logger.warning("faiss não instalado; indexação desabilitada")
                self.index = None

    def _init_db(self):
        cursor = self.conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                type TEXT NOT NULL,
                memory_type TEXT,
                content TEXT NOT NULL,
                metadata TEXT NOT NULL,
                embedding BLOB,
                feedback_score INTEGER DEFAULT 0,
                context_level TEXT DEFAULT 'long',
                disabled INTEGER DEFAULT 0,
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
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS conversation_embeddings (
                session_id TEXT,
                message_id INTEGER,
                vector BLOB,
                PRIMARY KEY (session_id, message_id)
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS monitoring_history (
                timestamp TEXT,
                reason TEXT,
                training_executed INTEGER,
                new_rules INTEGER
            )
            """
        )
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_memory_feedback ON memory(feedback_score)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_memory_access ON memory(access_count)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_memory_type ON memory(memory_type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_tags_tag ON tags(tag)")
        cursor.execute("PRAGMA table_info(memory)")
        cols = [r[1] for r in cursor.fetchall()]
        if "disabled" not in cols:
            cursor.execute("ALTER TABLE memory ADD COLUMN disabled INTEGER DEFAULT 0")
        self.conn.commit()

    def _load_index(self, index_file: str | None = None, ids_file: str | None = None) -> None:
        if not faiss:
            return
        index_file = index_file or config.INDEX_FILE
        ids_file = ids_file or config.INDEX_IDS_FILE
        if not (os.path.exists(index_file) and os.path.exists(ids_file)):
            return
        self.index = faiss.read_index(index_file)
        with open(ids_file, "r") as f:
            self.indexed_ids = json.load(f)
        logger.info("Índice de memória carregado do disco", items=len(self.indexed_ids))

    def _write_index_file(self, index_file: str) -> None:
        """Write the FAISS index to ``index_file`` if possible."""
        if not faiss or self.index is None:
            return
        faiss.write_index(self.index, index_file)

    def _write_ids_file(self, ids_file: str) -> None:
        """Persist the list of indexed IDs to ``ids_file``."""
        if self.index is None:
            return
        with open(ids_file, "w") as f:
            json.dump(self.indexed_ids, f)

    def _persist_index(self, index_file: str | None = None, ids_file: str | None = None) -> None:
        if not faiss or self.index is None:
            return
        index_file = index_file or config.INDEX_FILE
        ids_file = ids_file or config.INDEX_IDS_FILE
        self._write_index_file(index_file)
        self._write_ids_file(ids_file)

    def persist_index(self, index_file: str | None = None, ids_file: str | None = None) -> None:
        """Public wrapper to persist the current index to disk."""
        self._persist_index(index_file, ids_file)

    def load_index(self, index_file: str | None = None, ids_file: str | None = None) -> None:
        """Public wrapper to load index and ids if files exist."""
        self._load_index(index_file, ids_file)

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
        self.persist_index()

    def _get_embedding(self, text: str):
        if self.embedding_model is None:
            return [0.0]
        if text in self.embedding_cache:
            self.embedding_cache.move_to_end(text)
            logger.info("embedding_cache_hit", text=text[:30])
            return self.embedding_cache[text]
        try:
            vec = self.embedding_model.encode(text)
        except Exception as e:  # pragma: no cover - runtime embedding failure
            logger.error("embedding_error", error=str(e))
            self._register_embedding_fallback()
            self.embedding_model = None
            self.index = None
            vec = [0.0]
        self.embedding_cache[text] = vec
        if len(self.embedding_cache) > self.embedding_cache_size:
            self.embedding_cache.popitem(last=False)
        return vec

    def save(self, entry: Dict, update_feedback: bool = False):
        meta = entry.get("metadata", {})
        if entry.get("resposta_recomposta"):
            if not isinstance(meta, dict):
                try:
                    meta = json.loads(str(meta))
                except Exception:
                    meta = {"raw": str(meta)}
            meta["resposta_recomposta"] = True
        entry["metadata"] = json.dumps(meta)
        content = self._generate_content_for_embedding(entry)
        if self.index is not None:
            embedding_vec = self._get_embedding(content)
            if np is not None:
                embedding = np.array(embedding_vec, dtype=np.float32).tobytes()
            else:
                embedding = json.dumps(embedding_vec).encode()
        else:
            embedding_vec = None
            embedding = None
        cursor = self.conn.cursor()
        if update_feedback and "id" in entry:
            cursor.execute(
                "UPDATE memory SET feedback_score = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (entry.get("feedback_score", 0), entry["id"]),
            )
        else:
            cursor.execute(
                """INSERT INTO memory
                (type, memory_type, content, metadata, embedding, feedback_score, context_level)
                VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    entry.get("type", "generic"),
                    entry.get("memory_type"),
                    entry.get("content", ""),
                    entry["metadata"],
                    embedding,
                    entry.get("feedback_score", 0),
                    entry.get("context_level", "long"),
                ),
            )
            entry["id"] = cursor.lastrowid
            if self.index is not None and embedding_vec is not None:
                vec = embedding_vec
                if np is not None:
                    self.index.add(np.array([vec], dtype=np.float32))
                else:
                    self.index.add([vec])
                self.indexed_ids.append(entry["id"])
                self.persist_index()
        self.conn.commit()
        logger.info("Memória salva" if not update_feedback else "Feedback atualizado", entry_type=entry.get("type"))

    def _generate_content_for_embedding(self, entry: Dict) -> str:
        parts = [
            entry.get("content", ""),
            entry.get("type", ""),
            entry.get("memory_type", ""),
            entry.get("context_level", ""),
            " ".join(entry.get("tags", [])),
        ]
        if "metadata" in entry:
            if isinstance(entry["metadata"], dict):
                parts.append(json.dumps(entry["metadata"]))
            else:
                parts.append(str(entry["metadata"]))
        return " ".join(parts)

    def search(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.7,
        level: str | None = None,
        memory_type: str | None = None,
    ) -> List[Dict]:
        results = []
        cursor = self.conn.cursor()
        if self.index is None:
            sql = (
                "SELECT m.*, GROUP_CONCAT(t.tag, ', ') as tags FROM memory m "
                "LEFT JOIN tags t ON m.id = t.memory_id WHERE m.disabled = 0 AND m.content LIKE ?"
            )
            params = [f"%{query}%"]
            if memory_type:
                sql += " AND m.memory_type = ?"
                params.append(memory_type)
            if level:
                sql += " AND m.context_level = ?"
                params.append(level)
            sql += " GROUP BY m.id LIMIT ?"
            params.append(top_k)
            cursor.execute(sql, params)
            for row in cursor.fetchall():
                results.append(
                    {
                        "id": row[0],
                        "type": row[1],
                        "content": row[3],
                        "metadata": json.loads(row[4]),
                        "feedback_score": row[6],
                        "context_level": row[7],
                        "tags": row[12].split(", ") if row[12] else [],
                        "similarity_score": 1.0,
                        "last_accessed": row[8],
                        "access_count": row[9],
                    }
                )
        else:
            query_embedding = self._get_embedding(query)
            if np is not None:
                q = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
            else:
                q = query_embedding
            distances, indices = self.index.search(q, top_k)
            for idx, distance in zip(indices[0], distances[0]):
                if idx < len(self.indexed_ids) and distance <= (1 - min_score):
                    memory_id = self.indexed_ids[idx]
                    sql = (
                        "SELECT m.*, GROUP_CONCAT(t.tag, ', ') as tags FROM memory m "
                        "LEFT JOIN tags t ON m.id = t.memory_id WHERE m.id = ? AND m.disabled = 0"
                    )
                    params = [memory_id]
                    if memory_type:
                        sql += " AND m.memory_type = ?"
                        params.append(memory_type)
                    if level:
                        sql += " AND m.context_level = ?"
                        params.append(level)
                    sql += " GROUP BY m.id"
                    cursor.execute(sql, params)
                    row = cursor.fetchone()
                    if row:
                        results.append(
                            {
                                "id": row[0],
                                "type": row[1],
                                "content": row[3],
                                "metadata": json.loads(row[4]),
                                "feedback_score": row[6],
                                "context_level": row[7],
                                "tags": row[12].split(", ") if row[12] else [],
                                "similarity_score": 1 - distance,
                                "last_accessed": row[8],
                                "access_count": row[9],
                            }
                        )
                        cursor.execute(
                            "UPDATE memory SET last_accessed = CURRENT_TIMESTAMP, access_count = access_count + 1 WHERE id = ?",
                            (memory_id,),
                        )
        self.conn.commit()
        logger.info("Busca de memória realizada", query=query, results=len(results))

        def _score(item: Dict) -> float:
            recency = 0.0
            if item.get("last_accessed"):
                try:
                    age = datetime.now() - datetime.fromisoformat(item["last_accessed"])
                    recency = max(0.0, 1 - age.days / 30)
                except Exception:
                    pass
            return item["similarity_score"] + recency * 0.1

        return sorted(results, key=_score, reverse=True)

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
                    "relation_type": row[12],
                    "strength": row[13],
                    "tags": row[14].split(", ") if row[14] else [],
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
        self.persist_index()
        logger.info("Limpeza de memória executada", removed=len(ids))

    def deactivate_memories(self, term: str) -> int:
        """Deactivate memories related to the given term."""
        results = self.search(term, top_k=20)
        ids = [m["id"] for m in results]
        if not ids:
            return 0
        cursor = self.conn.cursor()
        placeholders = ",".join(["?"] * len(ids))
        cursor.execute(
            f"UPDATE memory SET disabled = 1 WHERE id IN ({placeholders})",
            ids,
        )
        self.conn.commit()
        logger.info("Memorias desativadas", term=term, count=len(ids))
        return len(ids)

    def record_feedback(self, memory_id: int, is_positive: bool):
        score_change = 1 if is_positive else -1
        cursor = self.conn.cursor()
        cursor.execute(
            "UPDATE memory SET feedback_score = feedback_score + ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (score_change, memory_id),
        )
        self.conn.commit()
        if not is_positive:
            try:
                cur = self.conn.cursor()
                cur.execute("SELECT content FROM memory WHERE id = ?", (memory_id,))
                row = cur.fetchone()
                if row:
                    self.feedback_db.add(str(memory_id), "negativo", row[0])
            except Exception:
                pass
        logger.info("Feedback registrado", memory_id=memory_id, positive=is_positive)

    def compress_memory(self) -> int:
        """Collapse highly similar memories into canonical entries."""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT id, content, memory_type, metadata FROM memory WHERE disabled = 0"
        )
        rows = cursor.fetchall()
        if not rows:
            return 0

        canonical: dict[int, str] = {}
        duplicates: list[tuple[int, int]] = []
        for mem_id, content, mtype, meta_json in rows:
            key = (mtype, content.strip().lower())
            found = None
            for cid, ccontent in canonical.items():
                if ccontent == key[1] and mtype == rows[cid - 1][2]:
                    found = cid
                    break
            if found:
                duplicates.append((mem_id, found))
            else:
                canonical[mem_id] = key[1]

        for dup_id, parent_id in duplicates:
            cursor.execute("SELECT metadata FROM memory WHERE id = ?", (parent_id,))
            meta = json.loads(cursor.fetchone()[0])
            merged = meta.get("merged_ids", [])
            merged.append(dup_id)
            meta["merged_ids"] = merged
            meta["data_compressao"] = datetime.now().isoformat()
            cursor.execute(
                "UPDATE memory SET metadata = ? WHERE id = ?",
                (json.dumps(meta), parent_id),
            )

            cursor.execute("SELECT metadata FROM memory WHERE id = ?", (dup_id,))
            dmeta = json.loads(cursor.fetchone()[0])
            dmeta["colapsado_em"] = parent_id
            cursor.execute(
                "UPDATE memory SET metadata = ?, disabled = 1 WHERE id = ?",
                (json.dumps(dmeta), dup_id),
            )

        self.conn.commit()
        logger.info("Memoria comprimida", count=len(duplicates))
        return len(duplicates)

    def recent_entries(self, entry_type: str, limit: int = 10) -> List[Dict]:
        """Return the most recent memory entries of a given type."""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT id, type, content, metadata, created_at FROM memory "
            "WHERE type = ? ORDER BY id DESC LIMIT ?",
            (entry_type, limit),
        )
        rows = cursor.fetchall()
        return [
            {
                "id": r[0],
                "type": r[1],
                "content": r[2],
                "metadata": json.loads(r[3]),
                "created_at": r[4],
            }
            for r in rows
        ]

    def prune_old_memories(self, threshold_days: int = 30) -> int:
        """Archive old memories to latent_memory.json with a symbolic reference."""
        cutoff = datetime.now() - timedelta(days=threshold_days)
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT id, type, memory_type, content, metadata, created_at FROM memory "
            "WHERE disabled = 0 AND access_count = 0 AND created_at < ?",
            (cutoff.isoformat(),),
        )
        rows = cursor.fetchall()
        if not rows:
            return 0

        latent_file = Path(config.LOG_DIR) / "latent_memory.json"
        try:
            data = json.loads(latent_file.read_text()) if latent_file.exists() else []
        except Exception:
            data = []

        for row in rows:
            mid, typ, mtype, content, meta_json, created = row
            data.append(
                {
                    "id": mid,
                    "type": typ,
                    "memory_type": mtype,
                    "content": content,
                    "metadata": json.loads(meta_json),
                    "created_at": created,
                }
            )
            meta = json.loads(meta_json)
            meta["latente"] = True
            meta["arquivo"] = str(latent_file)
            cursor.execute(
                "UPDATE memory SET content = ?, metadata = ?, disabled = 1 WHERE id = ?",
                ("essa memoria foi desativada por antiguidade", json.dumps(meta), mid),
            )

        latent_file.write_text(json.dumps(data, indent=2))
        self.conn.commit()
        logger.info("Memorias antigas arquivadas", count=len(rows))
        return len(rows)

    def close(self) -> None:
        """Persist index and close database connection."""
        try:
            if self.index is not None:
                self.persist_index()
            self.conn.commit()
            self.conn.close()
        except Exception:
            pass

    def _register_embedding_fallback(self) -> None:
        """Record embedding fallback notice in INTERNAL_DOCS."""
        path = Path("INTERNAL_DOCS.md")
        try:
            lines = path.read_text().splitlines() if path.exists() else []
        except Exception:
            lines = []
        header = "## pending_features"
        if header not in lines:
            lines.append(header)
        if "- embedding_fallback" not in lines:
            idx = lines.index(header) + 1 if header in lines else len(lines)
            lines.insert(idx, "- embedding_fallback")
        try:
            path.write_text("\n".join(lines) + "\n")
        except Exception as e:  # pragma: no cover - file system issues
            logger.error("Erro ao registrar fallback", error=str(e))
