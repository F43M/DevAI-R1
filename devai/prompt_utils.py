from __future__ import annotations

from typing import List, Dict, Sequence


def build_user_query_prompt(query: str, memories: Sequence[Dict], chunks: Sequence[Dict]) -> str:
    """Compose a prompt for answering a user query."""
    memory_context = "\n".join(
        f"// Memória [{m['similarity_score']:.2f}]: {m['content']}\n// Tags: {', '.join(m.get('tags', []))}\n"
        for m in memories[:3]
    )
    code_context = "\n\n".join(
        f"// {c['file']} ({c['type']} {c['name']})\n// Dependências: {', '.join(c['dependencies'])}\n{c['code']}"
        for c in chunks[:3]
    )
    return f"{memory_context}\n{code_context}\nUsuário: {query}\nIA:".strip()


def build_analysis_prompt(code: str, issues: Sequence[str]) -> str:
    """Prompt asking the model to review a code snippet."""
    return (
        "Analise o código a seguir e sugira melhorias de forma breve:\n"
        f"{code}\nProblemas detectados: {', '.join(issues)}"
    )


def build_refactor_prompt(code: str) -> str:
    """Prompt asking for a refactored version of the code."""
    return (
        "Refatore o código a seguir mantendo a funcionalidade e melhore o estilo:\n"
        f"{code}\n### Código refatorado:\n"
    )

