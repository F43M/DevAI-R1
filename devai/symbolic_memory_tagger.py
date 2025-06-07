from typing import Dict, List

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
    return tags
