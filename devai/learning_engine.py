# coding: utf-8
"""Learning Engine for symbolic self-improvement.

This engine centralizes how the project extracts lessons from code and
runtime events. It observes new logs, recent code analysis and manual
annotations looking for recurring patterns.

Gatilhos observados:
    - Funções analisadas com sucesso ou falha frequente
    - Logs de erro contendo *Exception* ou *FAIL*
    - Refatorações aprovadas em sequência

Resultado esperado:
    - Registrar explicações detalhadas de trechos de código
    - Armazenar lições de falhas e boas práticas
    - Sugerir padrões positivos para novos módulos
"""

from __future__ import annotations

import asyncio
import json
import time
import hashlib
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

from .config import logger, config

PROGRESS_FILE = Path(config.LOG_DIR) / "learning_progress.json"


def _write_progress(progress: float) -> None:
    """Persist learning progress as a percentage to disk."""
    try:
        PROGRESS_FILE.parent.mkdir(exist_ok=True)
        PROGRESS_FILE.write_text(json.dumps({"progress": int(progress)}, indent=2))
    except Exception:
        logger.warning("Nao foi possivel atualizar learning_progress.json")

def _learning_log_file() -> Path:
    return Path(config.LOG_DIR) / "learning_log.json"


def _load_learning_log() -> Dict[str, str]:
    file = _learning_log_file()
    try:
        if file.exists():
            return json.loads(file.read_text())
    except Exception:
        pass
    return {}


def _save_learning_log(data: Dict[str, str]) -> None:
    file = _learning_log_file()
    try:
        file.parent.mkdir(exist_ok=True)
        file.write_text(json.dumps(data, indent=2))
    except Exception:
        logger.warning("Nao foi possivel atualizar learning_log")
from .memory import MemoryManager
from .ai_model import AIModel
from .analyzer import CodeAnalyzer

LESSONS_FILE = Path(__file__).with_name("lessons.json")


def registrar_licao_negativa(arquivo: str, erro: str) -> None:
    """Store a negative lesson for later reference."""
    try:
        data = json.loads(LESSONS_FILE.read_text()) if LESSONS_FILE.exists() else []
    except Exception:
        data = []
    data.append(
        {
            "arquivo": arquivo,
            "erro": erro,
            "tipo": "licao_negativa",
            "timestamp": datetime.now().isoformat(),
            "processed": False,
        }
    )
    LESSONS_FILE.write_text(json.dumps(data, indent=2))


def listar_licoes_negativas(arquivo: str) -> List[str]:
    """Return negative lessons recorded for a given file."""
    if not LESSONS_FILE.exists():
        return []
    try:
        items = json.loads(LESSONS_FILE.read_text())
    except Exception:
        return []
    return [i.get("erro", "") for i in items if i.get("arquivo") == arquivo]


class LearningEngine:
    """Symbolic learning engine backed by DeepSeek R1."""

    def __init__(
        self,
        analyzer: CodeAnalyzer,
        memory: MemoryManager,
        ai_model: AIModel,
        rate_limit: int | None = None,
    ):
        self.analyzer = analyzer
        self.memory = memory
        self.ai_model = ai_model
        if rate_limit is None:
            rate_limit = getattr(config, "LEARNING_RATE_LIMIT", 5)
        self.rate_limit = rate_limit
        self._call_times: List[float] = []
        self.call_count = 0
        self.score_map = Path("devai/meta/score_map.json")
        self.priority_files: list[str] = []
        self.learning_log: Dict[str, str] = _load_learning_log()
        cur = self.memory.conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS processing_state (
                file TEXT PRIMARY KEY,
                hash TEXT,
                mtime REAL,
                last_processed TEXT
            )
            """
        )
        self.memory.conn.commit()

    def _load_negative_scores(self) -> None:
        """Load negative scored files from score_map.json."""
        if not self.score_map.exists():
            self.priority_files = []
            return
        try:
            scores = json.loads(self.score_map.read_text())
        except Exception:
            scores = {}
        negatives = {f: s for f, s in scores.items() if s < 0}
        self.priority_files = list(negatives)
        for f, score in negatives.items():
            self.memory.save(
                {
                    "type": "reflection",
                    "memory_type": "reflection",
                    "content": f"Arquivo {f} com score negativo {score}",
                    "metadata": {"file": f, "score": score},
                    "tags": ["metacognition", "critico"],
                    "context_level": "short",
                }
            )

    def register_rule(self, rule: str, source: Dict[str, Any]) -> None:
        """Store a learned rule and its origin."""
        metadata = {"source": source}
        self.memory.save(
            {
                "type": "learned_rule",
                "memory_type": "rule",
                "content": rule,
                "metadata": metadata,
                "tags": ["rule", "learning"],
            }
        )
        h = hashlib.sha1(rule.encode("utf-8")).hexdigest()
        self.learning_log[h] = datetime.now().isoformat()
        _save_learning_log(self.learning_log)

    async def _rate_limited_call(self, prompt: str, max_length: int = 800) -> str:
        """Invoke the model while respecting a simple per-minute limit."""
        now = time.time()
        self._call_times = [t for t in self._call_times if now - t < 60]
        if len(self._call_times) >= self.rate_limit:
            await asyncio.sleep(60 - (now - self._call_times[0]))
            self._call_times = [t for t in self._call_times if now - t < 60]
        self._call_times.append(time.time())
        self.call_count += 1
        logger.info("Chamada R1", total=self.call_count)
        return await self.ai_model.safe_api_call(
            prompt, max_length, prompt, self.memory
        )

    def _get_processing_state(self, file: str):
        cur = self.memory.conn.cursor()
        cur.execute(
            "SELECT hash, mtime FROM processing_state WHERE file=?",
            (file,),
        )
        return cur.fetchone()

    def _update_processing_state(self, file: str, h: str, mtime: float) -> None:
        cur = self.memory.conn.cursor()
        cur.execute(
            "INSERT OR REPLACE INTO processing_state (file, hash, mtime, last_processed) VALUES (?, ?, ?, ?)",
            (file, h, mtime, datetime.now().isoformat()),
        )
        self.memory.conn.commit()


    async def learn_from_codebase(self, block_size: int = 10):
        """Scan analyzed chunks and record explanations and best practices."""
        sem = asyncio.Semaphore(self.rate_limit)

        async def limited_call(prompt: str, max_len: int = 800) -> str:
            async with sem:
                return await self._rate_limited_call(prompt, max_len)

        self._load_negative_scores()
        chunks = list(self.analyzer.code_chunks.values())
        if self.priority_files:
            prioritized = [c for c in chunks if c.get("file") in self.priority_files]
            others = [c for c in chunks if c.get("file") not in self.priority_files]
            chunks = prioritized + others
        total = len(chunks)
        processed = 0
        _write_progress(0.0)

        for i in range(0, total, block_size):
            block = chunks[i : i + block_size]
            tasks: list[asyncio.Task] = []
            for chunk in block:
                code = chunk.get("code") or ""
                if not code:
                    continue
                file_path = chunk.get("file") or ""
                mtime = Path(file_path).stat().st_mtime if file_path else 0.0
                state = self._get_processing_state(file_path)
                if state and (state[0] == chunk.get("hash") or float(state[1]) == mtime):
                    continue

                async def process(c=chunk, mt=mtime):
                    meta = {"function": c["name"], "file": c.get("file")}
                    resp1, resp2, resp3 = await asyncio.gather(
                        limited_call(f"Explique o que essa funcao faz:\n{c['code']}") ,
                        limited_call(f"Ha algum risco ou ambiguidade?\n{c['code']}") ,
                        limited_call(f"Essa estrutura esta otimizada?\n{c['code']}")
                    )
                    self.memory.save(
                        {
                            "type": "learning",
                            "memory_type": "explicacao",
                            "content": resp1,
                            "metadata": meta,
                        }
                    )
                    self.memory.save(
                        {
                            "type": "learning",
                            "memory_type": "risco_oculto",
                            "content": resp2,
                            "metadata": meta,
                        }
                    )
                    self.memory.save(
                        {
                            "type": "learning",
                            "memory_type": "boas_praticas",
                            "content": resp3,
                            "metadata": meta,
                        }
                    )
                    self._update_processing_state(c.get("file"), c.get("hash", ""), mt)

                tasks.append(asyncio.create_task(process()))

            if tasks:
                await asyncio.gather(*tasks)

            processed = min(total, i + len(block))
            progress = processed / total * 100 if total else 100
            _write_progress(progress)
            logger.info("Progresso do aprendizado", progress=round(progress, 2))

        logger.info("Aprendizado do codigo concluido", calls=self.call_count)
    async def learn_from_errors(self):

        """Analyze recent log files and capture lessons from failures."""
        log_dir = Path(config.LOG_DIR)
        if not log_dir.exists():
            return
        sem = asyncio.Semaphore(self.rate_limit)

        async def limited_call(prompt: str, max_len: int = 800) -> str:
            async with sem:
                return await self._rate_limited_call(prompt, max_len)

        tasks = []

        for log_file in log_dir.glob("*.log"):
            meta_file = log_file.with_suffix(log_file.suffix + ".meta")
            if meta_file.exists():
                try:
                    meta = json.loads(meta_file.read_text())
                    if meta.get("processed"):
                        continue
                except Exception:
                    pass
            mtime = log_file.stat().st_mtime
            state = self._get_processing_state(str(log_file))
            if state and float(state[1]) == mtime:
                continue

            async def process(file_path=log_file, mt=mtime):
                try:
                    text = file_path.read_text()[-2000:]
                except Exception:
                    return
                if not any(kw in text for kw in ["ERROR", "Exception", "FAIL"]):
                    return
                resp1, resp2 = await asyncio.gather(
                    limited_call(f"Explique esse erro e como evita-lo:\n{text}"),
                    limited_call(f"O que causou esse comportamento?\n{text}")
                )
                self.memory.save(
                    {
                        "type": "erro",
                        "memory_type": "erro_estudado",
                        "content": resp1,
                        "metadata": {"file": str(file_path)},
                    }
                )
                self.memory.save(
                    {
                        "type": "erro",
                        "memory_type": "licao_aprendida",
                        "content": resp2,
                        "metadata": {"file": str(file_path)},
                    }
                )
                meta_file.write_text(
                    json.dumps(
                        {"processed": True, "processed_at": datetime.now().isoformat()},
                        indent=2,
                    )
                )
                self._update_processing_state(str(file_path), "", mt)

            tasks.append(asyncio.create_task(process()))

        if tasks:
            await asyncio.gather(*tasks)
        logger.info("Aprendizado de erros concluido")

    async def extract_positive_patterns(self):
        """Generate reusable good practices from approved refactorings."""
        entries = self.memory.search(
            "refatoracao aprovada", top_k=20, memory_type="refatoracao aprovada"
        )
        # Good snippets are distilled into general advice
        for e in entries:
            code = e.get("metadata", {}).get("code", "")
            prompt = f"Por que esse codigo e considerado bom?\n{code}\nExtraia padroes aplicaveis para novos projetos."
            resp = await self._rate_limited_call(prompt)
            self.memory.save(
                {
                    "type": "pattern",
                    "memory_type": "padrao_positivo",
                    "content": resp,
                    "metadata": {"source_id": e["id"]},
                }
            )
        logger.info("Padroes positivos extraidos", count=len(entries))

    async def reflect_on_internal_knowledge(self):
        """Summarize recent memories and highlight recurring issues."""
        cursor = self.memory.conn.cursor()
        # Retrieve the latest learning entries for a quick self-review
        cursor.execute("SELECT content FROM memory ORDER BY id DESC LIMIT 20")
        lines = [r[0] for r in cursor.fetchall()]
        if not lines:
            return
        prompt = (
            "Resuma o que aprendi recentemente e identifique repeticoes de erros ou falta de padroes:\n"
            + "\n".join(lines)
        )
        resp = await self._rate_limited_call(prompt, max_length=800)
        report = f"## Resumo do aprendizado\n{resp}\n"
        report_path = Path("logs/learning_report.md")
        report_path.write_text(report)
        logger.info("Relatorio de aprendizado salvo", file=str(report_path))

    async def import_external_codebase(self, path: str):
        """Learn from another repository and store the knowledge."""
        path_obj = Path(path)
        if not path_obj.exists():
            logger.error("Caminho externo invalido", path=path)
            return
        ext_analyzer = CodeAnalyzer(str(path_obj), self.memory)
        await ext_analyzer.deep_scan_app()
        # Treat external codebase as an additional source of patterns
        for chunk in ext_analyzer.code_chunks.values():
            code = chunk.get("code") or ""
            if not code:
                continue
            meta = {
                "function": chunk["name"],
                "file": chunk.get("file"),
                "source": str(path_obj),
            }
            resp = await self._rate_limited_call(
                f"Explique o que essa funcao faz:\n{code}"
            )
            self.memory.save(
                {
                    "type": "import",
                    "memory_type": "aprendizado_importado",
                    "content": resp,
                    "metadata": meta,
                }
            )
        logger.info("Aprendizado de codigo externo concluido", source=str(path_obj))

    async def summarize_patterns(self, limit: int = 20) -> list[str]:
        """Consolidate recent memories into generic rules."""
        cursor = self.memory.conn.cursor()
        cursor.execute("SELECT content FROM memory ORDER BY id DESC LIMIT ?", (limit,))
        entries = [r[0] for r in cursor.fetchall()]
        if not entries:
            return []
        prompt = "Resuma boas praticas em formato de regras:\n" + "\n".join(entries)
        resp = await self._rate_limited_call(prompt)
        rules = [r.strip("- ") for r in resp.splitlines() if r.strip()]
        for rule in rules:
            self.register_rule(rule, "summarize_patterns")
        return rules

    def find_symbolic_correlations(self) -> List[str]:
        """Identify common patterns among learned lessons."""
        cursor = self.memory.conn.cursor()
        cursor.execute(
            "SELECT memory_type, content FROM memory WHERE memory_type IN ("
            """
            "'erro_estudado','licao_aprendida','refatoracao_aplicada'"""
            ")"
        )
        rows = cursor.fetchall()
        groups: dict[str, list[str]] = {}
        # Group memories by type and prefix to detect repetition
        for mtype, content in rows:
            key = (mtype, " ".join(content.lower().split()[:5]))
            groups.setdefault(key[0], []).append(key[1])
        summary = []
        for k, items in groups.items():
            if len(items) > 1:
                summary.append(f"{k.replace('_', ' ')} ({len(items)} itens)")
        logger.info("Correlação simbólica gerada", total=len(summary))
        return summary

    async def explain_learning_lessons(self) -> str:
        """Generate a high level summary of learned lessons."""
        cursor = self.memory.conn.cursor()
        cursor.execute(
            "SELECT content FROM memory WHERE memory_type IN ('erro_resolvido','refatoracao_aplicada')"
        )
        lines = [r[0] for r in cursor.fetchall()]
        if not lines:
            return ""
        prompt = "Resuma de forma breve o que foi aprendido:\n" + "\n".join(lines)
        resp = await self.ai_model.safe_api_call(prompt, 800, prompt, self.memory)
        path = Path("logs/learning_summary.md")
        # Summaries are persisted for quick human review
        path.write_text(resp)
        logger.info("Resumo das lições salvo", file=str(path))
        return resp
