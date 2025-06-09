from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path
from typing import Tuple

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - fallback when PyYAML is missing
    from . import yaml_fallback as yaml

from .config import logger


def log_decision(
    tipo: str,
    modulo: str,
    motivo: str,
    modelo: str,
    resultado: str,
    score: str | None = None,
    fallback: bool | None = None,
    remember: bool | None = None,
) -> Tuple[str, str]:
    """Register a decision entry in ``decision_log.yaml`` and ``decision_log.md``."""
    path = Path("decision_log.yaml")
    data = []
    if path.exists():
        try:
            data = yaml.safe_load(path.read_text()) or []
        except Exception:
            data = []
        if not isinstance(data, list):
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
    if remember is not None:
        entry["remember"] = remember
    data.append(entry)
    path.write_text(yaml.safe_dump(data, allow_unicode=True))

    # Also append a short summary in Markdown for quick human inspection
    md_path = Path("decision_log.md")
    md_line = f"- {entry['timestamp']} [{entry['id']}] {tipo} {modulo} - {motivo}\n"
    with md_path.open("a", encoding="utf-8") as f:
        f.write(md_line)

    logger.info("Decisao registrada", id=entry_id, tipo=tipo)
    return entry_id, h


def is_remembered(action: str, path: str) -> bool:
    """Return True if a decision for ``action`` and ``path`` is marked to remember."""
    log_path = Path("decision_log.yaml")
    if not log_path.exists():
        return False
    try:
        data = yaml.safe_load(log_path.read_text()) or []
    except Exception:
        return False
    for entry in reversed(data):
        if not isinstance(entry, dict):
            continue
        if (
            entry.get("tipo") == action
            and entry.get("modulo") == path
            and entry.get("remember")
        ):
            return True
    return False
