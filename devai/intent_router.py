from __future__ import annotations

from typing import Dict

INTENT_KEYWORDS: Dict[str, list[str]] = {
    "debug": ["erro", "stack", "bug", "falha"],
    "create": ["crie", "novo", "gerar"],
    "edit": ["edite", "modificar", "ajuste"],
    "tests": ["teste", "pytest"],
    "review": ["revisao", "revise", "avaliar"],
    "architecture": ["arquitetura", "modulo", "depend"],
}


def detect_intent(query: str) -> str:
    """Simple rule-based intent detection."""
    q = query.lower()
    for intent, kws in INTENT_KEYWORDS.items():
        if any(k in q for k in kws):
            return intent
    return "generic"
