import re
import hashlib
from typing import Tuple, Dict

from .config import logger


def evaluate_ai_response(response: str) -> Tuple[str, Dict]:
    """Evaluate AI response for clarity, integrity and testability."""
    try:
        clarity = 2 if len(response.strip().split()) > 20 else 1
        integrity = 5
        testability = 5
        if re.search(r"TODO|FIXME", response, re.IGNORECASE):
            integrity -= 2
        func_calls = set(re.findall(r"([A-Za-z_][A-Za-z0-9_]*)\(", response))
        defs = set(re.findall(r"def ([A-Za-z_][A-Za-z0-9_]*)\(", response))
        missing = func_calls - defs
        if missing:
            integrity -= 1
        if "pytest" in response or "unittest" in response:
            testability += 2
        score = f"C{clarity}I{max(integrity,0)}T{max(testability,0)}"
        detail = {"missing_functions": list(missing)}
    except Exception as e:  # pragma: no cover - unforeseen errors
        logger.error("Falha na avaliacao", error=str(e))
        score = "C0I0T0"
        detail = {}
    logger.info("Resposta avaliada", score=score)
    return score, detail
