from typing import Dict, List

from .config import logger
from .analyzer import CodeAnalyzer
from .memory import MemoryManager


async def run_autoreview(analyzer: CodeAnalyzer, memory: MemoryManager) -> Dict[str, List[str]]:
    """Scan project and suggest symbolic refactors."""
    undocumented = [n for n, c in analyzer.code_chunks.items() if not c.get("docstring")]
    unused = [n for n in analyzer.code_chunks if analyzer.code_graph.out_degree(n) == 0]
    suggestions: List[str] = []
    if undocumented:
        suggestions.append(f"Funcoes sem docstring: {', '.join(undocumented[:5])}")
    if unused:
        suggestions.append(f"Possiveis funcoes nao utilizadas: {', '.join(unused[:5])}")
    memory.save({
        "type": "autoreview",
        "content": "AutoReview executado",
        "metadata": {"undocumented": undocumented, "unused": unused},
        "tags": ["autoreview"],
        "context_level": "short",
    })
    logger.info("AutoReview executado", undocumented=len(undocumented), unused=len(unused))
    return {"suggestions": suggestions}
