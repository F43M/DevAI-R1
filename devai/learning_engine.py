# coding: utf-8
"""Learning Engine for symbolic self-improvement.

This module coordinates learning tasks using DeepSeek R1. It relies on
``CodeAnalyzer`` to inspect code, ``MemoryManager`` to store the
knowledge and ``AIModel`` as the interface to the language model.
"""

from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime
from pathlib import Path
from typing import List

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

    def __init__(self, analyzer: CodeAnalyzer, memory: MemoryManager, ai_model: AIModel, rate_limit: int = 5):
        self.analyzer = analyzer
        self.memory = memory
        self.ai_model = ai_model
        self.rate_limit = rate_limit
        self._call_times: List[float] = []
        self.call_count = 0

    async def _rate_limited_call(self, prompt: str, max_length: int = 800) -> str:
        now = time.time()
        self._call_times = [t for t in self._call_times if now - t < 60]
        if len(self._call_times) >= self.rate_limit:
            await asyncio.sleep(60 - (now - self._call_times[0]))
            self._call_times = [t for t in self._call_times if now - t < 60]
        self._call_times.append(time.time())
        self.call_count += 1
        logger.info("Chamada R1", total=self.call_count)
        return await self.ai_model.generate(prompt, max_length=max_length)

    async def learn_from_codebase(self):
        for chunk in self.analyzer.code_chunks.values():
            code = chunk.get("code") or ""
            if not code:
                continue
            meta = {"function": chunk["name"], "file": chunk.get("file")}
            resp = await self._rate_limited_call(f"Explique o que essa funcao faz:\n{code}")
            self.memory.save({"type": "learning", "memory_type": "explicacao", "content": resp, "metadata": meta})
            resp = await self._rate_limited_call(f"Ha algum risco ou ambiguidade?\n{code}")
            self.memory.save({"type": "learning", "memory_type": "risco_oculto", "content": resp, "metadata": meta})
            resp = await self._rate_limited_call(f"Essa estrutura esta otimizada?\n{code}")
            self.memory.save({"type": "learning", "memory_type": "boas_praticas", "content": resp, "metadata": meta})
        logger.info("Aprendizado do codigo concluido", calls=self.call_count)

    async def learn_from_errors(self):
        log_dir = Path(config.LOG_DIR)
        if not log_dir.exists():
            return
        for log_file in log_dir.glob("*.log"):
            try:
                text = log_file.read_text()[-2000:]
            except Exception:
                continue
            if any(kw in text for kw in ["ERROR", "Exception", "FAIL"]):
                prompt = f"Explique esse erro e como evita-lo:\n{text}"
                resp = await self._rate_limited_call(prompt)
                self.memory.save({"type": "erro", "memory_type": "erro_estudado", "content": resp, "metadata": {"file": str(log_file)}})
                prompt = f"O que causou esse comportamento?\n{text}"
                resp = await self._rate_limited_call(prompt)
                self.memory.save({"type": "erro", "memory_type": "licao_aprendida", "content": resp, "metadata": {"file": str(log_file)}})
        logger.info("Aprendizado de erros concluido")

    async def extract_positive_patterns(self):
        entries = self.memory.search("refatoracao aprovada", top_k=20, memory_type="refatoracao aprovada")
        for e in entries:
            code = e.get("metadata", {}).get("code", "")
            prompt = f"Por que esse codigo e considerado bom?\n{code}\nExtraia padroes aplicaveis para novos projetos."
            resp = await self._rate_limited_call(prompt)
            self.memory.save({"type": "pattern", "memory_type": "padrao_positivo", "content": resp, "metadata": {"source_id": e["id"]}})
        logger.info("Padroes positivos extraidos", count=len(entries))

    async def reflect_on_internal_knowledge(self):
        cursor = self.memory.conn.cursor()
        cursor.execute("SELECT content FROM memory ORDER BY id DESC LIMIT 20")
        lines = [r[0] for r in cursor.fetchall()]
        if not lines:
            return
        prompt = "Resuma o que aprendi recentemente e identifique repeticoes de erros ou falta de padroes:\n" + "\n".join(lines)
        resp = await self._rate_limited_call(prompt, max_length=800)
        report = f"## Resumo do aprendizado\n{resp}\n"
        report_path = Path("logs/learning_report.md")
        report_path.write_text(report)
        logger.info("Relatorio de aprendizado salvo", file=str(report_path))

    async def import_external_codebase(self, path: str):
        path_obj = Path(path)
        if not path_obj.exists():
            logger.error("Caminho externo invalido", path=path)
            return
        ext_analyzer = CodeAnalyzer(str(path_obj), self.memory)
        await ext_analyzer.deep_scan_app()
        for chunk in ext_analyzer.code_chunks.values():
            code = chunk.get("code") or ""
            if not code:
                continue
            meta = {"function": chunk["name"], "file": chunk.get("file"), "source": str(path_obj)}
            resp = await self._rate_limited_call(f"Explique o que essa funcao faz:\n{code}")
            self.memory.save({"type": "import", "memory_type": "aprendizado_importado", "content": resp, "metadata": meta})
        logger.info("Aprendizado de codigo externo concluido", source=str(path_obj))
