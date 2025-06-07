from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path
from typing import Sequence, Dict, List, Optional, Any
import json

SCORE_MAP = Path("devai/meta/score_map.json")

SELF_LOG = Path("devai/logs/self_reflection.md")


def build_metacognition_prompt(history: Sequence[Dict]) -> str:
    """Compose prompt asking model for next steps based on decision history."""
    lines = []
    for item in history[-5:]:
        ts = item.get("timestamp", "")
        lines.append(f"{ts} - {item.get('tipo')} em {item.get('modulo')}")
    hist_text = "\n".join(lines)
    return (
        f"Com base nas decisões abaixo, qual próximo passo você recomendaria?\n{hist_text}\nResposta:".strip()
    )


class MetacognitionLoop:
    """Scheduled self-reflection loop (enhanced)."""

    def __init__(self, history_file: str = "decision_log.yaml", memory: Optional[Any] = None) -> None:
        self.history_file = Path(history_file)
        self.memory = memory

    async def run(self, interval_hours: int = 24 * 7) -> None:
        """Run periodic analysis; default weekly."""
        while True:
            await self._analyze()
            await asyncio.sleep(interval_hours * 3600)

    async def _analyze(self) -> None:
        if not self.history_file.exists():
            return
        try:
            import yaml  # type: ignore

            data: List[Dict] = yaml.safe_load(self.history_file.read_text()) or []
        except Exception:
            data = []
        now = datetime.now()
        recent = []
        for item in data:
            try:
                ts = datetime.fromisoformat(item.get("timestamp", "1970-01-01"))
            except Exception:
                ts = now
            if (now - ts).total_seconds() < 86400:
                recent.append(item)

        scores: Dict[str, int] = {}
        reflections = []
        for item in recent:
            file = item.get("modulo", "desconhecido")
            scores[file] = scores.get(file, 0) + (1 if item.get("tipo") != "erro" else -1)
            reflections.append(
                {
                    "contexto": file,
                    "acao": item.get("tipo"),
                    "resultado": item.get("decision_score", ""),
                    "alternativa": "Revisar abordagem",
                }
            )

        SCORE_MAP.parent.mkdir(parents=True, exist_ok=True)
        SCORE_MAP.write_text(json.dumps(scores, indent=2))

        lines = [f"# Reflexão gerada em {now.isoformat()}"]
        for r in reflections:
            lines.append(
                f"- Contexto: {r['contexto']}\n  Ação: {r['acao']}\n  Resultado: {r['resultado']}\n  Alternativa sugerida: {r['alternativa']}"
            )
        SELF_LOG.parent.mkdir(parents=True, exist_ok=True)
        SELF_LOG.write_text("\n".join(lines))

        if self.memory:
            for f, score in scores.items():
                if score < -2:
                    summary = f"Arquivo {f} apresentou recorrência de erros."
                    self.memory.save(
                        {
                            "type": "reflection",
                            "memory_type": "licao aprendida",
                            "content": summary,
                            "metadata": {"file": f, "score": score},
                            "tags": ["licao"],
                            "context_level": "short",
                        }
                    )
