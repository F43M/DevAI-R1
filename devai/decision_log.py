from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path
from typing import Tuple

try:
    import yaml
except Exception:  # pragma: no cover - optional dependency
    yaml = None

from .config import logger


def log_decision(tipo: str, modulo: str, motivo: str, modelo: str, resultado: str, score: str | None = None, fallback: bool | None = None) -> Tuple[str, str]:
    """Register a decision entry in decision_log.yaml."""
    path = Path("decision_log.yaml")
    data = []
    if path.exists() and yaml is not None:
        try:
            data = yaml.safe_load(path.read_text()) or []
        except Exception:
            data = []
    entry_id = f"{len(data) + 1:03d}"
    h = hashlib.sha256(resultado.encode()).hexdigest()[:8]
    entry = {
        "id": entry_id,
        "tipo": tipo,
        "modulo": modulo,
        "motivo": motivo,
        "modelo_ia": modelo,
        "hash_resultado": h,
        "timestamp": datetime.now().isoformat(),
    }
    if score is not None:
        entry["decision_score"] = score
    if fallback is not None:
        entry["fallback"] = fallback
    data.append(entry)
    if yaml is not None:
        path.write_text(yaml.safe_dump(data, allow_unicode=True))
    logger.info("Decisao registrada", id=entry_id, tipo=tipo)
    return entry_id, h
