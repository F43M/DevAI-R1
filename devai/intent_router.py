from __future__ import annotations

from typing import Dict
from pathlib import Path

from .intent_classifier import predict_intent

INTENT_KEYWORDS: Dict[str, list[str]] = {
    "debug": ["erro", "stack", "bug", "falha"],
    "create": ["crie", "novo", "gerar"],
    "edit": [
        "edite",
        "modificar",
        "ajuste",
        "refator",
        "corrigir",
        "alterar",
        "patch",
        "diff",
    ],
    "tests": ["teste", "pytest"],
    "review": ["revisao", "revise", "avaliar"],
    "architecture": ["arquitetura", "modulo", "depend"],
}


def _keyword_intent(query: str) -> str:
    q = query.lower()
    for intent, kws in INTENT_KEYWORDS.items():
        if any(k in q for k in kws):
            return intent
    return "generic"


def detect_intent(query: str) -> str:
    """Return intent using model when available with keyword fallback."""
    model_path = Path("intent_model.pkl")
    if model_path.exists():
        try:
            return predict_intent(query)
        except Exception:
            pass
    return _keyword_intent(query)
