from typing import Dict, List
from .config import config

previous_hashes: Dict[str, str] = {}


def tag_memory_entry(metadata: Dict) -> List[str]:
    """Return symbolic tags based on code chunk metadata."""
    tags: List[str] = []
    name = metadata.get("name", "")
    h = metadata.get("hash", "")
    if not name:
        return tags
    prev = previous_hashes.get(name)
    if prev is None:
        tags.append("@nova_funcao")
    elif prev != h:
        tags.append("@refatorado")
    previous_hashes[name] = h
    doc = metadata.get("docstring", "").lower()
    if "erro" in doc or "bug" in doc:
        tags.append("@erro_corrigido")
    if "deprecated" in doc or "obsoleto" in doc:
        tags.append("@descontinuado")
    complexity = metadata.get("complexity")
    try:
        if float(complexity) > config.COMPLEXITY_TAG_THRESHOLD:
            tags.append("@complexo")
    except Exception:
        pass
    return tags
