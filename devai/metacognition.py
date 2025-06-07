from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path
from typing import Sequence, Dict, List

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
    """Scheduled self-reflection loop (skeleton)."""

    def __init__(self, history_file: str = "decision_log.yaml") -> None:
        self.history_file = Path(history_file)

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
        # TODO: analisar padroes de erros e propor melhorias
        lines = [
            f"# Reflexão gerada em {datetime.now().isoformat()}",
            "\nTODO: análise automática de padrões ainda não implementada.",
        ]
        SELF_LOG.parent.mkdir(parents=True, exist_ok=True)
        SELF_LOG.write_text("\n".join(lines))
