import os
import ast
import json
import yaml
import sqlite3
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from datetime import datetime, timedelta
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
import faiss
import structlog
import asyncio
import hashlib
import aiofiles
import aiofiles.os
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn
import networkx as nx
import aiohttp
from collections import defaultdict
from asteval import Interpreter
import logging.handlers

# --- Configuração ---
class Config:
    def __init__(self):
        self.OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
        self.MODEL_NAME = "deepseek/deepseek-r1-0528:free"
        self.CODE_ROOT = "./app"
        self.MEMORY_DB = "memory.sqlite"
        self.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
        self.TASK_DEFINITIONS = "tasks.yaml"
        self.LOG_DIR = "./logs"
        self.API_PORT = 8000
        self.LEARNING_LOOP_INTERVAL = 300  # 5 minutos
        # Ajustado para aproveitar o limite de 160k tokens oferecido pela API
        # da OpenRouter.
        self.MAX_CONTEXT_LENGTH = 160000
        self.OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
        self.INDEX_FILE = "faiss.index"
        self.INDEX_IDS_FILE = "faiss_ids.json"

config = Config()

# --- Métricas simples para monitoramento ---
class Metrics:
    def __init__(self):
        self.api_calls = 0
        self.total_response_time = 0.0
        self.errors = 0

    def record_call(self, duration: float):
        self.api_calls += 1
        self.total_response_time += duration

    def record_error(self):
        self.errors += 1

    def summary(self) -> Dict[str, Any]:
        avg_time = self.total_response_time / self.api_calls if self.api_calls else 0
        return {
            "api_calls": self.api_calls,
            "avg_response_time": avg_time,
            "errors": self.errors,
        }


# --- Configuração de Logging Avançada ---
def configure_logging():
    if not os.path.exists(config.LOG_DIR):
        os.makedirs(config.LOG_DIR)
    
    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.stdlib.add_log_level,
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    file_handler = logging.handlers.RotatingFileHandler(
        filename=os.path.join(config.LOG_DIR, "ai_core.log"),
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    
    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)
    root_logger.setLevel(logging.INFO)

logger = structlog.get_logger()
metrics = Metrics()

# --- Módulo de Memória Avançado ---
class MemoryManager:
    def __init__(self, db_file: str, embedding_model: str):
        self.conn = sqlite3.connect(db_file)
        self._init_db()
        self.embedding_model = SentenceTransformer(embedding_model)
        self.dimension = self.embedding_model.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatL2(self.dimension)
        self.indexed_ids = []
        self._load_index()
    
    def _init_db(self):
        cursor = self.conn.cursor()
        cursor.execute("""
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
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS semantic_relations (
                source_id INTEGER,
                target_id INTEGER,
                relation_type TEXT,
                strength REAL,
                PRIMARY KEY (source_id, target_id),
                FOREIGN KEY (source_id) REFERENCES memory (id),
                FOREIGN KEY (target_id) REFERENCES memory (id)
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tags (
                memory_id INTEGER,
                tag TEXT,
                PRIMARY KEY (memory_id, tag),
                FOREIGN KEY (memory_id) REFERENCES memory (id)
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_memory_feedback ON memory(feedback_score)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_memory_access ON memory(access_count)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_tags_tag ON tags(tag)")
        self.conn.commit()
    
    def _load_index(self):
        if os.path.exists(config.INDEX_FILE) and os.path.exists(config.INDEX_IDS_FILE):
            self.index = faiss.read_index(config.INDEX_FILE)
            with open(config.INDEX_IDS_FILE, "r") as f:
                self.indexed_ids = json.load(f)
            logger.info("Índice de memória carregado do disco", items=len(self.indexed_ids))
            return

        cursor = self.conn.cursor()
        cursor.execute("SELECT id, embedding FROM memory WHERE embedding IS NOT NULL")
        embeddings = []
        for row in cursor.fetchall():
            self.indexed_ids.append(row[0])
            embedding = np.frombuffer(row[1], dtype=np.float32).reshape(1, -1)
            embeddings.append(embedding)

        if embeddings:
            self.index.add(np.concatenate(embeddings))

        self._persist_index()
        logger.info("Índice de memória carregado", items=len(self.indexed_ids))

    def _persist_index(self):
        faiss.write_index(self.index, config.INDEX_FILE)
        with open(config.INDEX_IDS_FILE, "w") as f:
            json.dump(self.indexed_ids, f)

    def _rebuild_index(self):
        self.index = faiss.IndexFlatL2(self.dimension)
        self.indexed_ids = []
        cursor = self.conn.cursor()
        cursor.execute("SELECT id, embedding FROM memory WHERE embedding IS NOT NULL")
        embeddings = []
        for row in cursor.fetchall():
            self.indexed_ids.append(row[0])
            embeddings.append(np.frombuffer(row[1], dtype=np.float32).reshape(1, -1))
        if embeddings:
            self.index.add(np.concatenate(embeddings))
        self._persist_index()
    
    def save(self, entry: Dict, update_feedback: bool = False):
        entry["metadata"] = json.dumps(entry.get("metadata", {}))
        content = self._generate_content_for_embedding(entry)
        embedding = self.embedding_model.encode(content).astype(np.float32).tobytes()
        
        cursor = self.conn.cursor()
        
        if update_feedback and "id" in entry:
            cursor.execute(
                "UPDATE memory SET feedback_score = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (entry.get("feedback_score", 0), entry["id"])
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
                    entry.get("feedback_score", 0)
                )
            )
            entry_id = cursor.lastrowid
            
            if "tags" in entry:
                tags = entry["tags"] if isinstance(entry["tags"], list) else [entry["tags"]]
                cursor.executemany(
                    "INSERT OR IGNORE INTO tags (memory_id, tag) VALUES (?, ?)",
                    [(entry_id, tag) for tag in tags]
                )
            
            self.index.add(np.frombuffer(embedding, dtype=np.float32).reshape(1, -1))
            self.indexed_ids.append(entry_id)

        self._persist_index()

        self.conn.commit()
        logger.info("Memória salva" if not update_feedback else "Feedback atualizado",
                   entry_type=entry.get("type"))
    
    def _generate_content_for_embedding(self, entry: Dict) -> str:
        content_parts = [
            entry.get("content", ""),
            entry.get("type", ""),
            " ".join(entry.get("tags", []))
        ]
        if "metadata" in entry:
            if isinstance(entry["metadata"], dict):
                content_parts.append(json.dumps(entry["metadata"]))
            else:
                content_parts.append(str(entry["metadata"]))
        return " ".join(content_parts)
    
    def search(self, query: str, top_k: int = 5, min_score: float = 0.7) -> List[Dict]:
        query_embedding = self.embedding_model.encode(query).astype(np.float32)
        distances, indices = self.index.search(query_embedding.reshape(1, -1), top_k)
        
        results = []
        cursor = self.conn.cursor()
        
        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(self.indexed_ids) and distance <= (1 - min_score):
                memory_id = self.indexed_ids[idx]
                cursor.execute("""
                    SELECT m.*, GROUP_CONCAT(t.tag, ', ') as tags
                    FROM memory m
                    LEFT JOIN tags t ON m.id = t.memory_id
                    WHERE m.id = ?
                    GROUP BY m.id
                """, (memory_id,))
                
                row = cursor.fetchone()
                if row:
                    result = {
                        "id": row[0],
                        "type": row[1],
                        "content": row[2],
                        "metadata": json.loads(row[3]),
                        "feedback_score": row[5],
                        "tags": row[9].split(", ") if row[9] else [],
                        "similarity_score": 1 - distance,
                        "last_accessed": row[6],
                        "access_count": row[7]
                    }
                    results.append(result)
                    
                    cursor.execute(
                        "UPDATE memory SET last_accessed = CURRENT_TIMESTAMP, access_count = access_count + 1 WHERE id = ?",
                        (memory_id,)
                    )
        
        self.conn.commit()
        logger.info("Busca de memória realizada", query=query, results=len(results))
        return sorted(results, key=lambda x: x["similarity_score"], reverse=True)
    
    def add_semantic_relation(self, source_id: int, target_id: int, relation_type: str, strength: float = 1.0):
        cursor = self.conn.cursor()
        cursor.execute(
            """INSERT OR REPLACE INTO semantic_relations 
            (source_id, target_id, relation_type, strength) 
            VALUES (?, ?, ?, ?)""",
            (source_id, target_id, relation_type, strength)
        )
        self.conn.commit()
        logger.info("Relação semântica adicionada", 
                   source=source_id, target=target_id, type=relation_type)
    
    def get_related_memories(self, memory_id: int, relation_type: str = None) -> List[Dict]:
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
            results.append({
                "id": row[0],
                "type": row[1],
                "content": row[2],
                "metadata": json.loads(row[3]),
                "relation_type": row[9],
                "strength": row[10],
                "tags": row[11].split(", ") if row[11] else []
            })

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
            (score_change, memory_id)
        )
        self.conn.commit()
        logger.info("Feedback registrado", memory_id=memory_id, positive=is_positive)

# --- Módulo de Análise de Código ---
class CodeAnalyzer:
    def __init__(self, code_root: str, memory: MemoryManager):
        self.code_root = Path(code_root)
        self.memory = memory
        self.code_chunks = {}
        self.code_graph = nx.DiGraph()
        self.vectorizer = TfidfVectorizer()
        self.file_hashes = {}
        self.external_dependencies = set()
        self.learned_rules = {}
        self.last_analysis_time = datetime.now()
    
    async def deep_scan_app(self):
        logger.info("Iniciando varredura profunda do aplicativo")
        await self.scan_app()
        await self._build_semantic_relations()
        await self._analyze_patterns()
        logger.info("Varredura profunda concluída", 
                   functions=len(self.code_chunks), 
                   relations=self.code_graph.number_of_edges())
    
    async def scan_app(self):
        tasks = []
        for file_path in self.code_root.rglob("*.py"):
            if file_path.is_file():
                tasks.append(self.parse_file(file_path))
        await asyncio.gather(*tasks)
        logger.info("Aplicativo escaneado", files_processed=len(tasks))
    
    async def parse_file(self, file_path: Path):
        try:
            async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
                content = await f.read()
            
            file_hash = hashlib.md5(content.encode()).hexdigest()
            
            if str(file_path) in self.file_hashes and self.file_hashes[str(file_path)] == file_hash:
                return
            
            self.file_hashes[str(file_path)] = file_hash
            tree = ast.parse(content)
            chunks = []
            
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
                    chunk = {
                        "name": node.name,
                        "type": type(node).__name__,
                        "code": ast.unparse(node),
                        "file": str(file_path),
                        "hash": file_hash,
                        "last_modified": datetime.now().isoformat(),
                        "dependencies": self._get_dependencies(node),
                        "external_deps": self._get_external_dependencies(node),
                        "docstring": ast.get_docstring(node) or "",
                        "line_start": node.lineno,
                        "line_end": node.end_lineno
                    }
                    
                    self.code_graph.add_node(node.name, **chunk)
                    for dep in chunk["dependencies"]:
                        self.code_graph.add_edge(node.name, dep)
                    
                    self.external_dependencies.update(chunk["external_deps"])
                    chunks.append(chunk)
                    
                    if node.name in self.code_chunks and self.code_chunks[node.name]["hash"] != file_hash:
                        logger.warning(
                            "Código modificado",
                            chunk=node.name,
                            old_hash=self.code_chunks[node.name]["hash"],
                            new_hash=file_hash
                        )
            
            self.code_chunks.update({c["name"]: c for c in chunks})
            
            for chunk in chunks:
                memory_entry = {
                    "type": "code_chunk",
                    "content": f"{chunk['type']} {chunk['name']} em {chunk['file']}",
                    "metadata": chunk,
                    "tags": ["code", chunk['type'].lower(), os.path.basename(chunk['file'])]
                }
                self.memory.save(memory_entry)
            
        except SyntaxError as e:
            logger.error("Erro de sintaxe", file=str(file_path), error=str(e))
        except Exception as e:
            logger.error("Erro ao analisar arquivo", file=str(file_path), error=str(e))
    
    def _get_dependencies(self, node):
        deps = set()
        for child in ast.walk(node):
            if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
                deps.add(child.id)
            elif isinstance(child, ast.Attribute):
                deps.add(child.attr)
        return list(deps)
    
    def _get_external_dependencies(self, node):
        deps = set()
        for child in ast.walk(node):
            if isinstance(child, ast.Import):
                for n in child.names:
                    deps.add(n.name.split(".")[0])
            elif isinstance(child, ast.ImportFrom):
                if child.module:
                    deps.add(child.module.split(".")[0])
        return list(deps)
    
    async def _build_semantic_relations(self):
        logger.info("Construindo relações semânticas")
        
        cursor = self.memory.conn.cursor()
        cursor.execute("""
            SELECT id, metadata FROM memory 
            WHERE type = 'code_chunk' 
            AND json_extract(metadata, '$.name') IS NOT NULL
        """)
        
        code_memories = {json.loads(row[1])["name"]: row[0] for row in cursor.fetchall()}
        
        for source, target in self.code_graph.edges():
            if source in code_memories and target in code_memories:
                self.memory.add_semantic_relation(
                    code_memories[source],
                    code_memories[target],
                    "depends_on",
                    strength=0.9
                )
        
        all_code = [(mid, json.loads(meta)["name"]) for name, mid in code_memories.items()]
        
        for i, (mid1, name1) in enumerate(all_code):
            for j, (mid2, name2) in enumerate(all_code[i+1:], i+1):
                if self._name_similarity(name1, name2) > 0.7:
                    self.memory.add_semantic_relation(
                        mid1, mid2, "similar_name", strength=0.7)
        
        logger.info("Relações semânticas construídas", relations=len(self.code_graph.edges()))
    
    def _name_similarity(self, name1: str, name2: str) -> float:
        from difflib import SequenceMatcher
        return SequenceMatcher(None, name1.lower(), name2.lower()).ratio()
    
    async def _analyze_patterns(self):
        logger.info("Analisando padrões para regras aprendidas")
        
        for chunk in self.code_chunks.values():
            try:
                tree = ast.parse(chunk["code"])
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef) and len(node.args.args) > 5:
                        rule_name = "funcao_com_muitos_parametros"
                        condition = "len(chunk['code'].split('def ')[1].split('(')[1].split(')')[0].split(',')) > 5"
                        
                        if rule_name not in self.learned_rules:
                            self.learned_rules[rule_name] = condition
                            self.memory.save({
                                "type": "learned_rule",
                                "content": f"Função com muitos parâmetros detectada: {chunk['name']}",
                                "metadata": {
                                    "rule": rule_name,
                                    "condition": condition,
                                    "example": chunk["name"]
                                },
                                "tags": ["rule", "code_smell"]
                            })
            except Exception as e:
                logger.error("Erro ao analisar padrão", chunk=chunk["name"], error=str(e))
        
        logger.info("Análise de padrões concluída", rules=len(self.learned_rules))
    
    def get_code_graph(self) -> Dict:
        return {
            "nodes": [{"id": n, **data} for n, data in self.code_graph.nodes(data=True)],
            "links": [{"source": u, "target": v} for u, v in self.code_graph.edges()]
        }

# --- Módulo de Monitoramento de Logs ---
class LogMonitor:
    def __init__(self, memory: MemoryManager, log_dir: str = "./logs"):
        self.memory = memory
        self.log_dir = Path(log_dir)
        self.patterns = {
            "error": r"ERROR|CRITICAL|FAILED|Exception",
            "warning": r"WARNING|Deprecation",
            "performance": r"Timeout|Slow|Latency"
        }
        self.last_checked = datetime.now()
    
    async def monitor_logs(self):
        while True:
            try:
                if not self.log_dir.exists():
                    await aiofiles.os.makedirs(self.log_dir, exist_ok=True)
                
                log_files = [f for f in self.log_dir.glob("*.log") if f.is_file()]
                
                for log_file in log_files:
                    await self._analyze_log_file(log_file)
                
                self.last_checked = datetime.now()
                await asyncio.sleep(60)
            except Exception as e:
                logger.error("Erro no monitor de logs", error=str(e))
                await asyncio.sleep(30)
    
    async def _analyze_log_file(self, log_file: Path):
        try:
            async with aiofiles.open(log_file, "r") as f:
                lines = await f.readlines()
            
            for line in lines:
                if not line.strip():
                    continue
                
                timestamp = datetime.now().isoformat()
                log_entry = json.loads(line) if line.startswith("{") else {"message": line.strip()}
                
                detected_patterns = []
                for pattern_type, pattern in self.patterns.items():
                    if re.search(pattern, line, re.IGNORECASE):
                        detected_patterns.append(pattern_type)
                
                if detected_patterns:
                    context = "\n".join(lines[max(0, lines.index(line)-5):lines.index(line)+1])
                    
                    self.memory.save({
                        "type": "log_analysis",
                        "content": f"Padrão detectado em logs: {', '.join(detected_patterns)}",
                        "metadata": {
                            "file": str(log_file),
                            "patterns": detected_patterns,
                            "context": context,
                            "timestamp": timestamp
                        },
                        "tags": ["log"] + detected_patterns
                    })
        except json.JSONDecodeError:
            logger.warning("Formato de log inválido", file=str(log_file))
        except Exception as e:
            logger.error("Erro ao analisar arquivo de log", file=str(log_file), error=str(e))

# --- Módulo de Tarefas ---
class TaskManager:
    def __init__(self, task_file: str, code_analyzer: CodeAnalyzer, memory: MemoryManager):
        self.tasks = self._load_tasks(task_file)
        self.code_analyzer = code_analyzer
        self.memory = memory
        self.aeval = Interpreter()
        self._setup_default_tasks()
    
    def _load_tasks(self, task_file: str) -> Dict:
        if os.path.exists(task_file):
            with open(task_file, "r") as f:
                return yaml.safe_load(f) or {}
        return {}
    
    def _setup_default_tasks(self):
        if "impact_analysis" not in self.tasks:
            self.tasks["impact_analysis"] = {
                "name": "Análise de Impacto",
                "type": "analysis",
                "scope": "dependents",
                "condition": "True",
                "description": "Analisa o impacto de mudanças em funções"
            }
        
        if "code_review" not in self.tasks:
            self.tasks["code_review"] = {
                "name": "Revisão de Código",
                "type": "verification",
                "scope": "all",
                "condition": "'🚩' in findings or '⚠️' in findings",
                "description": "Revisão geral de qualidade de código"
            }
    
    async def run_task(self, task_name: str, *args) -> Any:
        if task_name not in self.tasks:
            logger.error("Tarefa não encontrada", task=task_name)
            return {"error": f"Tarefa '{task_name}' não encontrada"}
        
        task = self.tasks[task_name]
        logger.info("Executando tarefa", task=task_name)
        
        if task["type"] == "analysis":
            result = await self._perform_analysis_task(task, *args)
        elif task["type"] == "verification":
            result = await self._perform_verification_task(task, *args)
        elif task["type"] == "learning":
            result = await self._perform_learning_task(task, *args)
        else:
            logger.error("Tipo de tarefa inválido", task_type=task["type"])
            result = {"error": f"Tipo de tarefa '{task['type']}' não suportado"}
        
        self.memory.save({
            "type": "task_result",
            "content": f"Resultado da tarefa {task_name}",
            "metadata": {
                "task": task_name,
                "args": args,
                "result": result,
                "timestamp": datetime.now().isoformat()
            },
            "tags": ["task", task_name.split("_")[0]]
        })
        
        return result
    
    async def _perform_analysis_task(self, task: Dict, *args) -> List[Dict]:
        target = args[0] if args else None
        findings = []
        
        if task["scope"] == "all":
            chunks = list(self.code_analyzer.code_chunks.values())
        elif task["scope"] == "dependents":
            chunks = []
            for name in args:
                if name in self.code_analyzer.code_graph.nodes:
                    for dependent in nx.descendants(self.code_analyzer.code_graph, name):
                        if dependent in self.code_analyzer.code_chunks:
                            chunks.append(self.code_analyzer.code_chunks[dependent])
        else:
            chunks = [self.code_analyzer.code_chunks[target]] if target in self.code_analyzer.code_chunks else []
        
        for chunk in chunks:
            try:
                findings.append({
                    "chunk": chunk["name"],
                    "file": chunk["file"],
                    "issues": self._check_dependencies(chunk["name"]),
                    "rule_findings": self._apply_learned_rules(chunk),
                    "custom_findings": self._evaluate_custom_condition(task["condition"], chunk, args)
                })
            except Exception as e:
                logger.error("Erro na análise", chunk=chunk["name"], error=str(e))
                findings.append({
                    "chunk": chunk["name"],
                    "error": str(e)
                })
        
        return findings
    
    def _evaluate_custom_condition(self, condition: str, chunk: Dict, args: tuple) -> List[str]:
        findings = []
        try:
            if self.aeval(condition, chunk=chunk, args=args, graph=self.code_analyzer.code_graph):
                findings.append("✅ Condição personalizada atendida")
            else:
                findings.append("❌ Condição personalizada não atendida")
        except Exception as e:
            findings.append(f"⚠️ Erro ao avaliar condição: {str(e)}")
        return findings
    
    async def _perform_verification_task(self, task: Dict, *args) -> List[str]:
        findings = []
        
        if task["scope"] == "all":
            chunks = list(self.code_analyzer.code_chunks.values())
        else:
            target = args[0] if args else None
            chunks = [self.code_analyzer.code_chunks[target]] if target in self.code_analyzer.code_chunks else []
        
        for chunk in chunks:
            issues = self._check_dependencies(chunk["name"])
            rule_findings = self._apply_learned_rules(chunk)
            
            if issues or rule_findings:
                findings.append(f"Verificação {task['name']} em {chunk['name']}:")
                findings.extend(issues)
                findings.extend(rule_findings)
        
        return findings if findings else ["✅ Nenhum problema encontrado"]
    
    async def _perform_learning_task(self, task: Dict, *args) -> Dict:
        if task["operation"] == "add_rule":
            rule_name = args[0]
            condition = args[1]
            self.code_analyzer.learned_rules[rule_name] = condition
            
            self.memory.save({
                "type": "learned_rule",
                "content": f"Nova regra aprendida: {rule_name}",
                "metadata": {
                    "rule": rule_name,
                    "condition": condition,
                    "learned_at": datetime.now().isoformat()
                },
                "tags": ["learning", "rule"]
            })
            
            return {"status": "success", "rule_added": rule_name}
        
        return {"status": "error", "message": "Operação de aprendizado não reconhecida"}
    
    def _check_dependencies(self, chunk_name: str) -> List[str]:
        issues = []
        if chunk_name not in self.code_analyzer.code_graph:
            return issues
        
        for dep in self.code_analyzer.code_graph.successors(chunk_name):
            if dep not in self.code_analyzer.code_chunks:
                issues.append(f"⚠️ {chunk_name} depende de {dep} que não foi encontrado")
            else:
                chunk = self.code_analyzer.code_chunks[dep]
                if "last_modified" in chunk:
                    if (datetime.now() - datetime.fromisoformat(chunk["last_modified"])).days < 7:
                        issues.append(f"🚩 {chunk_name} depende de {dep} que foi modificado recentemente")
        
        return issues
    
    def _apply_learned_rules(self, chunk: Dict) -> List[str]:
        findings = []
        for rule, condition in self.code_analyzer.learned_rules.items():
            try:
                if self.aeval(condition, chunk=chunk, graph=self.code_analyzer.code_graph):
                    findings.append(f"🧠 {rule}")
            except Exception as e:
                logger.error("Erro ao aplicar regra", rule=rule, error=str(e))
                findings.append(f"⚠️ Erro ao aplicar regra {rule}: {str(e)}")
        return findings

# --- Módulo de IA com DeepSeek-R1 via OpenRouter ---
class AIModel:
    def __init__(self):
        self.session = aiohttp.ClientSession()
        if not config.OPENROUTER_API_KEY:
            logger.error("Chave OPENROUTER_API_KEY não configurada")
        logger.info("Modelo DeepSeek-R1 configurado via OpenRouter")
    
    async def generate(self, prompt: str, max_length: int = config.MAX_CONTEXT_LENGTH) -> str:
        headers = {
            "Authorization": f"Bearer {config.OPENROUTER_API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": config.MODEL_NAME,
            "messages": [{"role": "user", "content": prompt}],
            # Utiliza a configuração de contexto máximo, respeitando o limite
            # de 160k tokens da API.
            "max_tokens": min(max_length, config.MAX_CONTEXT_LENGTH),
            "temperature": 0.7
        }
        
        start = datetime.now()
        try:
            async with self.session.post(
                config.OPENROUTER_URL,
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=60),
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data['choices'][0]['message']['content']
                else:
                    error = await resp.text()
                    metrics.record_error()
                    logger.error("Erro na chamada ao OpenRouter", status=resp.status, error=error)
                    return f"Erro na API: {resp.status} - {error}"
        except asyncio.TimeoutError:
            metrics.record_error()
            logger.error("Timeout na chamada ao OpenRouter")
            return "Erro: tempo limite excedido ao chamar a API"
        except Exception as e:
            metrics.record_error()
            logger.error("Erro na conexão com OpenRouter", error=str(e))
            return f"Erro de conexão: {str(e)}"
        finally:
            metrics.record_call((datetime.now() - start).total_seconds())
    
    async def close(self):
        await self.session.close()

# --- Classe Principal Integrada ---
class CodeMemoryAI:
    def __init__(self):
        configure_logging()
        
        self.memory = MemoryManager(config.MEMORY_DB, config.EMBEDDING_MODEL)
        self.analyzer = CodeAnalyzer(config.CODE_ROOT, self.memory)
        self.tasks = TaskManager(config.TASK_DEFINITIONS, self.analyzer, self.memory)
        self.log_monitor = LogMonitor(self.memory, config.LOG_DIR)
        self.ai_model = AIModel()
        
        self.app = FastAPI(title="CodeMemoryAI API")
        self._setup_api_routes()
        
        self.background_tasks = set()
        self._start_background_tasks()
        
        logger.info("CodeMemoryAI inicializado com DeepSeek-R1 via OpenRouter")
    
    def _start_background_tasks(self):
        task = asyncio.create_task(self._learning_loop())
        self.background_tasks.add(task)
        task.add_done_callback(self.background_tasks.discard)
        
        task = asyncio.create_task(self.log_monitor.monitor_logs())
        self.background_tasks.add(task)
        task.add_done_callback(self.background_tasks.discard)
        
        task = asyncio.create_task(self.analyzer.deep_scan_app())
        self.background_tasks.add(task)
        task.add_done_callback(self.background_tasks.discard)
    
    def _setup_api_routes(self):
        @self.app.post("/analyze")
        async def analyze_code(query: str):
            return await self.generate_response(query)
        
        @self.app.get("/memory")
        async def search_memory(query: str, top_k: int = 5):
            return self.memory.search(query, top_k)
        
        @self.app.post("/feedback")
        async def submit_feedback(memory_id: int, is_positive: bool):
            self.memory.record_feedback(memory_id, is_positive)
            return {"status": "success"}
        
        @self.app.get("/graph")
        async def show_graph():
            return self.analyzer.get_code_graph()
        
        @self.app.get("/status")
        async def get_status():
            return {
                "code_chunks": len(self.analyzer.code_chunks),
                "memory_items": len(self.memory.indexed_ids),
                "learned_rules": len(self.analyzer.learned_rules),
                "last_activity": self.analyzer.last_analysis_time.isoformat()
            }

        @self.app.get("/metrics")
        async def get_metrics():
            return metrics.summary()
        
        os.makedirs("static", exist_ok=True)
        self.app.mount("/static", StaticFiles(directory="static"), name="static")
    
    async def _learning_loop(self):
        error_count = 0
        while True:
            try:
                await self.analyzer.scan_app()
                await self.log_monitor.monitor_logs()
                await self._run_scheduled_tasks()
                await self._generate_automatic_insights()
                error_count = 0
                await asyncio.sleep(config.LEARNING_LOOP_INTERVAL)
            except Exception as e:
                error_count += 1
                logger.error("Erro no loop de aprendizado", error=str(e), count=error_count)
                wait_time = min(30 * error_count, 300)
                await asyncio.sleep(wait_time)
    
    async def _run_scheduled_tasks(self):
        if datetime.now().hour == 3:
            await self.tasks.run_task("code_review")
            self.memory.cleanup()
    
    async def _generate_automatic_insights(self):
        complex_functions = []
        for name, chunk in self.analyzer.code_chunks.items():
            if len(chunk["dependencies"]) > 10:
                complex_functions.append(name)
        
        if complex_functions:
            self.memory.save({
                "type": "insight",
                "content": f"Funções complexas detectadas: {', '.join(complex_functions)}",
                "metadata": {
                    "functions": complex_functions,
                    "metric": "dependencies_count",
                    "threshold": 10
                },
                "tags": ["insight", "complexity"]
            })
    
    async def generate_response(self, query: str) -> str:
        try:
            contextual_memories = self.memory.search(query)
            memory_context = "\n".join(
                f"// Memória [{m['similarity_score']:.2f}]: {m['content']}\n"
                f"// Tags: {', '.join(m.get('tags', []))}\n"
                for m in contextual_memories[:3]
            )
            
            relevant_chunks = self._find_relevant_code(query)
            code_context = "\n\n".join(
                f"// {chunk['file']} ({chunk['type']} {chunk['name']})\n"
                f"// Dependências: {', '.join(chunk['dependencies'])}\n"
                f"{chunk['code']}"
                for chunk in relevant_chunks[:3]
            )
            
            auto_findings = []
            for chunk in relevant_chunks:
                auto_findings.extend(self.tasks._check_dependencies(chunk["name"]))
                auto_findings.extend(self.tasks._apply_learned_rules(chunk))
            
            prompt = f"""Você é um especialista no código abaixo. Considere o contexto histórico:

{memory_context}

Código relevante:
{code_context}

Verificações automáticas:
{'\n'.join(auto_findings) if auto_findings else '✅ Nenhum problema detectado'}

Pergunta: {query}

Analise cuidadosamente, considere as dependências e histórico antes de responder:
Resposta:"""
            
            # Solicita uma resposta utilizando todo o limite de contexto
            response = await self.ai_model.generate(prompt, max_length=config.MAX_CONTEXT_LENGTH)
            
            self.memory.save({
                "type": "qa",
                "content": query,
                "metadata": {
                    "response": response,
                    "code_context": [c["name"] for c in relevant_chunks],
                    "memory_context": [m["id"] for m in contextual_memories]
                },
                "tags": self._extract_tags(response + " " + query)
            })
            
            logger.info("Resposta gerada", query=query[:50])
            return response
        except Exception as e:
            logger.error("Erro ao gerar resposta", query=query, error=str(e))
            return f"Erro ao processar sua pergunta: {str(e)}"
    
    def _find_relevant_code(self, query: str) -> List[Dict]:
        exact_matches = [chunk for name, chunk in self.analyzer.code_chunks.items() 
                        if name.lower() in query.lower()]
        
        if exact_matches:
            return exact_matches
        
        query_embedding = self.analyzer.embedding_model.encode(query).astype(np.float32)
        
        similarities = []
        for name, chunk in self.analyzer.code_chunks.items():
            chunk_embedding = self.analyzer.embedding_model.encode(
                f"{chunk['name']} {chunk['docstring']}"
            ).astype(np.float32)
            similarity = np.dot(query_embedding, chunk_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(chunk_embedding)
            )
            similarities.append((similarity, chunk))
            
        similarities.sort(reverse=True, key=lambda x: x[0])
        return [chunk for _, chunk in similarities[:5]]
    
    def _extract_tags(self, text: str) -> List[str]:
        tags = set()
        
        if "erro" in text.lower():
            tags.add("erro")
        if "warning" in text.lower():
            tags.add("warning")
        if "sugestão" in text.lower():
            tags.add("sugestão")
        
        emoji_tags = {
            "🚩": "atenção",
            "⚠️": "aviso",
            "⚙️": "técnico",
            "🧠": "aprendizado",
            "✅": "sucesso"
        }
        
        for emoji, tag in emoji_tags.items():
            if emoji in text:
                tags.add(tag)
        
        return list(tags)
    
    async def analyze_impact(self, changed_functions: List[str]) -> List[Dict]:
        impacted = defaultdict(list)
        
        for func in changed_functions:
            if func in self.analyzer.code_graph:
                for dependent in nx.descendants(self.analyzer.code_graph, func):
                    impacted[dependent].append(func)
        
        report = []
        for target, deps in impacted.items():
            analysis = await self.tasks.run_task("impact_analysis", target)
            report.append({
                "target": target,
                "triggers": deps,
                "findings": analysis
            })
        
        logger.info("Análise de impacto concluída", changed_functions=changed_functions)
        return report
    
    async def verify_compliance(self, spec: Dict) -> List[str]:
        findings = []
        
        for func, expected in spec.items():
            if func not in self.analyzer.code_chunks:
                findings.append(f"Função {func} não encontrada no código")
                continue
            
            chunk = self.analyzer.code_chunks[func]
            if "expected_inputs" in expected:
                actual_inputs = self._extract_inputs(chunk["code"])
                if set(actual_inputs) != set(expected["expected_inputs"]):
                    findings.append(f"⚠️ {func} espera inputs {expected['expected_inputs']} mas recebe {actual_inputs}")
            
            if "expected_output" in expected:
                return_type = self._infer_return_type(chunk["code"])
                if return_type and return_type != expected["expected_output"]:
                    findings.append(f"⚠️ {func} parece retornar {return_type} mas deveria retornar {expected['expected_output']}")
        
        return findings
    
    def _extract_inputs(self, code: str) -> List[str]:
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    return [arg.arg for arg in node.args.args]
        except Exception as e:
            logger.error("Erro ao extrair inputs", error=str(e))
        return []
    
    def _infer_return_type(self, code: str) -> Optional[str]:
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Return):
                    if node.value is None:
                        return "None"
                    elif isinstance(node.value, ast.Num):
                        return "number"
                    elif isinstance(node.value, ast.Str):
                        return "string"
                    elif isinstance(node.value, ast.List):
                        return "list"
                    elif isinstance(node.value, ast.Dict):
                        return "dict"
        except Exception as e:
            logger.error("Erro ao inferir tipo de retorno", error=str(e))
        return None
    
    async def run(self):
        config = uvicorn.Config(
            self.app,
            host="0.0.0.0",
            port=config.API_PORT,
            log_level="info"
        )
        server = uvicorn.Server(config)
        await server.serve()

# --- Interface de Linha de Comando ---
async def cli_main():
    print("Inicializando CodeMemoryAI com DeepSeek-R1...")
    ai = CodeMemoryAI()
    
    asyncio.create_task(ai.analyzer.deep_scan_app())
    
    print("\nDev IA Avançado Pronto!")
    print("Comandos disponíveis:")
    print("/memoria <query> - Busca memórias relevantes")
    print("/tarefa <nome> [args] - Executa uma tarefa")
    print("/analisar <função> - Analisa impacto de mudanças")
    print("/verificar - Verifica conformidade com especificação")
    print("/grafo - Mostra grafo de dependências")
    print("/sair - Encerra")
    
    while True:
        try:
            user_input = input("\n>>> ").strip()
            
            if user_input.lower() == "/sair":
                break
            elif user_input.lower().startswith("/memoria"):
                query = user_input[len("/memoria"):].strip() or "recent"
                memories = ai.memory.search(query, top_k=5)
                print("\nMemórias relevantes:")
                for m in memories:
                    print(f"- [{m['similarity_score']:.2f}] {m['content'][:80]}... (tags: {', '.join(m['tags'])})")
            elif user_input.startswith("/tarefa "):
                parts = user_input[len("/tarefa "):].split()
                task_name = parts[0]
                args = parts[1:] if len(parts) > 1 else []
                result = await ai.tasks.run_task(task_name, *args)
                print(json.dumps(result, indent=2))
            elif user_input.startswith("/analisar "):
                func = user_input[len("/analisar "):]
                report = await ai.analyze_impact([func])
                for item in report:
                    print(f"\nImpacto em {item['target']} (gatilhos: {', '.join(item['triggers'])}):")
                    for finding in item["findings"]:
                        if isinstance(finding, dict):
                            print(f"- {finding.get('chunk')}:")
                            for issue in finding.get('issues', []):
                                print(f"  {issue}")
                        else:
                            print(f"- {finding}")
            elif user_input == "/verificar":
                spec = {
                    "calculate_score": {
                        "expected_inputs": ["data", "weights"],
                        "expected_output": "number"
                    }
                }
                findings = await ai.verify_compliance(spec)
                for finding in findings:
                    print(finding)
            elif user_input == "/grafo":
                graph = ai.analyzer.get_code_graph()
                print("Grafo de dependências:")
                for node in graph["nodes"]:
                    print(f"- {node['id']} ({node.get('type', 'function')})")
                print("\nConexões:")
                for link in graph["links"]:
                    print(f"{link['source']} -> {link['target']}")
            else:
                response = await ai.generate_response(user_input)
                print("\nResposta:")
                print(response)
                
        except Exception as e:
            logger.error("Erro na CLI", input=user_input, error=str(e))
            print(f"Erro: {str(e)}")

# --- Ponto de Entrada Principal ---
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="CodeMemoryAI - Assistente de Código Inteligente")
    parser.add_argument("--api", action="store_true", help="Inicia o servidor API")
    parser.add_argument("--cli", action="store_true", help="Inicia a interface de linha de comando")
    
    args = parser.parse_args()
    
    if not config.OPENROUTER_API_KEY:
        print("Erro: A variável de ambiente OPENROUTER_API_KEY não está definida")
        exit(1)
    
    if args.api:
        ai = CodeMemoryAI()
        asyncio.run(ai.run())
    elif args.cli:
        asyncio.run(cli_main())
    else:
        print("Por favor, especifique --api ou --cli para iniciar o aplicativo")