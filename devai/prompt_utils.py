from __future__ import annotations

from typing import List, Dict, Sequence
from .prompt_engine import SYSTEM_PROMPT_CONTEXT
from .learning_engine import listar_licoes_negativas


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


def build_cot_prompt(query: str, memories: Sequence[Dict], chunks: Sequence[Dict]) -> str:
    """Compose a prompt encouraging step-by-step reasoning."""
    base = build_user_query_prompt(query, memories, chunks)
    return base + "\nVamos pensar passo a passo antes de responder."


def build_analysis_prompt(code: str, issues: Sequence[str]) -> str:
    """Prompt asking the model to review a code snippet."""
    return (
        f"{SYSTEM_PROMPT_CONTEXT}\nAnalise o código a seguir e sugira melhorias de forma breve:\n"
        f"{code}\nProblemas detectados: {', '.join(issues)}\nVamos pensar passo a passo antes de responder."
    )


def build_refactor_prompt(code: str, file_path: str | None = None) -> str:
    """Prompt asking for a refactored version of the code."""
    lesson_text = ""
    if file_path:
        licoes = listar_licoes_negativas(file_path)
        if licoes:
            items = "\n".join(f"- {e}" for e in licoes)
            lesson_text = (
                "Atenção: Neste arquivo, os seguintes padrões de erro já ocorreram:\n"
                f"{items}\nEvite repetir esses problemas.\n"
            )
    return (
        f"{SYSTEM_PROMPT_CONTEXT}\n{lesson_text}Refatore o código a seguir mantendo a funcionalidade e melhore o estilo:\n"
        f"{code}\n### Código refatorado:\nVamos pensar passo a passo antes de responder.\n"
    )

