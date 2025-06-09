from typing import Dict, List

from .config import logger, config
from .analyzer import CodeAnalyzer
from .memory import MemoryManager
from .complexity_tracker import ComplexityTracker


async def run_autoreview(analyzer: CodeAnalyzer, memory: MemoryManager) -> Dict[str, List[str]]:
    """Scan project and suggest symbolic refactors."""
    undocumented = [n for n, c in analyzer.code_chunks.items() if not c.get("docstring")]
    unused = [n for n in analyzer.code_chunks if analyzer.code_graph.out_degree(n) == 0]
    threshold = config.AUTO_REVIEW_COMPLEXITY_THRESHOLD
    complex_funcs = [
        n
        for n, c in analyzer.code_chunks.items()
        if c.get("complexity", 0) > threshold
    ]
    complexities = [c.get("complexity", 0) for c in analyzer.code_chunks.values()]
    avg_complexity = sum(complexities) / len(complexities) if complexities else 0
    ComplexityTracker(config.COMPLEXITY_HISTORY).record(avg_complexity)

    suggestions: List[str] = []
    if undocumented:
        suggestions.append(f"Funcoes sem docstring: {', '.join(undocumented[:5])}")
    if unused:
        suggestions.append(f"Possiveis funcoes nao utilizadas: {', '.join(unused[:5])}")
    if complex_funcs:
        suggestions.append(f"Funcoes complexas: {', '.join(complex_funcs[:5])}")

    memory.save(
        {
            "type": "autoreview",
            "content": "AutoReview executado",
            "metadata": {
                "undocumented": undocumented,
                "unused": unused,
                "complex": complex_funcs,
                "avg_complexity": avg_complexity,
            },
            "tags": ["autoreview"],
            "context_level": "short",
        }
    )
    logger.info(
        "AutoReview executado",
        undocumented=len(undocumented),
        unused=len(unused),
        complex=len(complex_funcs),
    )
    return {"suggestions": suggestions}
