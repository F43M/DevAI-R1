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
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

from .config import logger, config
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
        rate_limit: int = 5,
    ):
        self.analyzer = analyzer
        self.memory = memory
        self.ai_model = ai_model
        self.rate_limit = rate_limit
        self._call_times: List[float] = []
        self.call_count = 0

    def register_rule(self, rule: str, source: Dict[str, Any]) -> None:
        """Store a learned rule and its origin."""
        self.memory.save(
            {
                "type": "learned_rule",
                "memory_type": "rule",
                "content": rule,
                "metadata": source,
                "tags": ["rule", "learning"],
            }
        )

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

    async def learn_from_codebase(self):
        """Scan analyzed chunks and record explanations and best practices."""
        # Loop over every known function to extract knowledge for the memory bank
        for chunk in self.analyzer.code_chunks.values():
            code = chunk.get("code") or ""
            if not code:
                continue
            meta = {"function": chunk["name"], "file": chunk.get("file")}
            resp = await self._rate_limited_call(
                f"Explique o que essa funcao faz:\n{code}"
            )
            self.memory.save(
                {
                    "type": "learning",
                    "memory_type": "explicacao",
                    "content": resp,
                    "metadata": meta,
                }
            )
            resp = await self._rate_limited_call(
                f"Ha algum risco ou ambiguidade?\n{code}"
            )
            self.memory.save(
                {
                    "type": "learning",
                    "memory_type": "risco_oculto",
                    "content": resp,
                    "metadata": meta,
                }
            )
            resp = await self._rate_limited_call(
                f"Essa estrutura esta otimizada?\n{code}"
            )
            self.memory.save(
                {
                    "type": "learning",
                    "memory_type": "boas_praticas",
                    "content": resp,
                    "metadata": meta,
                }
            )
        logger.info("Aprendizado do codigo concluido", calls=self.call_count)

    async def learn_from_errors(self):
        """Analyze recent log files and capture lessons from failures."""
        log_dir = Path(config.LOG_DIR)
        if not log_dir.exists():
            return
        # Inspect each log and extract insights when error keywords show up
        for log_file in log_dir.glob("*.log"):
            meta_file = log_file.with_suffix(log_file.suffix + ".meta")
            if meta_file.exists():
                try:
                    meta = json.loads(meta_file.read_text())
                    if meta.get("processed"):
                        continue
                except Exception:
                    pass
            try:
                text = log_file.read_text()[-2000:]
            except Exception:
                continue
            if any(kw in text for kw in ["ERROR", "Exception", "FAIL"]):
                prompt = f"Explique esse erro e como evita-lo:\n{text}"
                resp = await self._rate_limited_call(prompt)
                self.memory.save(
                    {
                        "type": "erro",
                        "memory_type": "erro_estudado",
                        "content": resp,
                        "metadata": {"file": str(log_file)},
                    }
                )
                prompt = f"O que causou esse comportamento?\n{text}"
                resp = await self._rate_limited_call(prompt)
                self.memory.save(
                    {
                        "type": "erro",
                        "memory_type": "licao_aprendida",
                        "content": resp,
                        "metadata": {"file": str(log_file)},
                    }
                )
                meta_file.write_text(
                    json.dumps(
                        {"processed": True, "processed_at": datetime.now().isoformat()},
                        indent=2,
                    )
                )
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
