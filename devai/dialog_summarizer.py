"""Conversation summarization into symbolic memories."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import logger, config

import asyncio

from .ai_model import AIModel
from .memory import MemoryManager


class DialogSummarizer:
    """Extract symbolic memories from conversation history."""

    def summarize_conversation(
        self,
        history: List[Dict[str, Any]],
        memory: Optional[MemoryManager] = None,
    ) -> List[Dict[str, str]]:
        """Analyze conversation pairs and return symbolic memory sentences."""

        memories: List[Dict[str, str]] = []
        seen: set[str] = set()

        for msg in history:
            if msg.get("role") != "user":
                continue
            text = msg.get("content", "")
            lower = text.lower()

            tag = ""
            summary = ""

            if re.search(r"prefir[oa]|gosto de|melhor usar", lower):
                tag = "#preferencia_usuario"
                summary = f"Prefere {text.split('prefiro')[-1].strip()}" if "prefiro" in lower else text
            elif re.search(r"n[aã]o use|evite|n[aã]o utilizar", lower):
                tag = "#licao_aprendida"
                m = re.search(r"(n[aã]o use|evite|n[aã]o utilizar)\s+(.*)", lower)
                phrase = m.group(2) if m else text
                if "eval" in phrase:
                    summary = "IA n\u00e3o deve usar 'eval' por raz\u00f5es de seguran\u00e7a"
                else:
                    summary = f"IA n\u00e3o deve {phrase.strip()}"
            elif re.search(r"est[aá] errado|n[aã]o funciona|quebra|n[aã]o compila", lower):
                tag = "#correcao_ia"
                summary = "Usu\u00e1rio apontou erro na solu\u00e7\u00e3o proposta"
            elif re.search(r"perfeito|exatamente|isso mesmo|agora sim|\u00f3timo", lower):
                tag = "#decisao_confirmada"
                summary = "Usu\u00e1rio aprovou a resposta anterior"
            elif re.search(
                r"sem explica[c\u00e7][a\u00e3]o|s[o\u00f3] c[o\u00f3]digo|modo sniper|s[o\u00f3] fun\u00e7\u00e3o|fun\u00e7[\u00f5]es puras",
                lower,
            ):
                tag = "#estilo_requisitado"
                style = re.search(
                    r"modo sniper|sem explica[c\u00e7][a\u00e3]o|s[o\u00f3] c[o\u00f3]digo|s[o\u00f3] fun\u00e7\u00e3o|fun\u00e7[\u00f5]es puras",
                    lower,
                )
                summary = f"O usu\u00e1rio solicitou '{style.group(0)}'" if style else text

            if tag and summary:
                sent = f"{tag}: {summary.strip().rstrip('.')}."  # normalize ending
                key = f"{tag}:{sent.lower()}"
                if key not in seen:
                    seen.add(key)
                    memories.append({"tag": tag, "content": sent})

        if not memories:
            logger.info(
                "\u26a0\ufe0f Nenhuma mem\u00f3ria simb\u00f3lica extra\u00edda da conversa."
            )
            self._register_fallback()
            if config.ENABLE_AI_SUMMARY:
                try:
                    recent = [m.get("content", "") for m in history[-4:]]
                    prompt = (
                        "Resuma em ate 3 pontos as preferencias ou licoes do usuario:\n"
                        + "\n".join(recent)
                    )

                    async def _call() -> str:
                        ai = AIModel()
                        try:
                            return await ai.safe_api_call(prompt, 150)
                        finally:
                            await ai.close()

                    resp = asyncio.run(_call())
                    extracted: List[Dict[str, str]] = []
                    for line in resp.splitlines():
                        m = re.match(r"-?\s*(#[\w_]+)[:\-]?\s*(.*)", line.strip())
                        if not m:
                            continue
                        tag = m.group(1)
                        text = m.group(2).strip()
                        sent = f"{tag}: {text.rstrip('.')}." if text else tag
                        key = f"{tag}:{sent.lower()}"
                        if key in seen:
                            continue
                        seen.add(key)
                        entry = {"tag": tag, "content": sent}
                        extracted.append(entry)
                        if memory is not None:
                            try:
                                memory.save(
                                    {
                                        "type": "dialog",
                                        "memory_type": "dialog_summary",
                                        "content": sent,
                                        "metadata": {"tag": tag, "origin": "ai_summary"},
                                        "tags": [tag],
                                    }
                                )
                            except Exception:
                                pass
                    if memory is not None:
                        return []
                    memories.extend(extracted)
                except Exception as e:  # pragma: no cover - unexpected errors
                    logger.error("summary_fallback_failed", error=str(e))

        return memories

    def _register_fallback(self) -> None:
        """Record fallback notice in INTERNAL_DOCS."""

        path = Path("INTERNAL_DOCS.md")
        try:
            lines = path.read_text().splitlines() if path.exists() else []
        except Exception:
            lines = []

        header = "## pending_features"
        if header not in lines:
            lines.append(header)
        if "- memory_extraction_fallback" not in lines:
            idx = lines.index(header) + 1 if header in lines else len(lines)
            lines.insert(idx, "- memory_extraction_fallback")
        try:
            path.write_text("\n".join(lines) + "\n")
            logger.info("#resumo_pendente")
        except Exception as e:  # pragma: no cover - file system issues
            logger.error("Erro ao registrar fallback", error=str(e))
