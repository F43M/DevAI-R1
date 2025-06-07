from __future__ import annotations

import json
from typing import Any, Dict, Sequence
from uuid import uuid4

from .config import logger

# Templates for different task types
PROMPT_TEMPLATES = {
    "create": "{identity}\nCrie o código solicitado.\n{context}",
    "edit": "{identity}\nEdite o código conforme a orientação.\n{context}",
    "debug": "{identity}\nAnalise o erro e sugira correções.\n{context}\nErro:\n{error}\nLogs:\n{logs}",
    "architecture": "{identity}\nForneça orientações de arquitetura.\n{context}",
    "tests": "{identity}\nEscreva testes para o trecho informado.\n{context}",
    "review": "{identity}\nRevise o seguinte código.\n{context}",
}

AGENT_IDENTITY = (
    "Você é o agente DevAI-R1.\n"
    "Seu papel é gerar código inteligente, testado e funcional com base no projeto atual.\n"
    "Trabalhe com precisão cirúrgica."
)


def _format_memories(memories: Sequence[Dict]) -> str:
    lines = []
    for m in memories[:3]:
        tag_str = " ".join(f"@{t}" for t in m.get("tags", []))
        lines.append(
            f"// Memória relevante: {m['content']} {tag_str}\n"
            f"// Função: {m['metadata'].get('name', '')}, Similaridade: {m['similarity_score']:.2f}"
        )
    return "\n".join(lines)


def _format_graph(graph_summary: str) -> str:
    return f"Grafo de dependências:\n{graph_summary}" if graph_summary else ""


def _format_actions(actions: Sequence[str]) -> str:
    if not actions:
        return ""
    joined = "\n".join(f"- {a}" for a in actions)
    return f"Últimas ações:\n{joined}"


def build_cot_prompt(
    comando: str,
    contexto: str,
    grafo: str,
    memoria: Sequence[Dict],
    logs: str = "",
    actions: Sequence[str] | None = None,
) -> str:
    parts = [AGENT_IDENTITY]
    if memoria:
        parts.append(_format_memories(memoria))
    if grafo:
        parts.append(_format_graph(grafo))
    if contexto:
        parts.append(contexto)
    if actions:
        parts.append(_format_actions(actions))
    if logs:
        parts.append(f"Logs recentes:\n{logs}")
    parts.append(f"Usuário: {comando}\nIA:")
    return "\n".join(filter(None, parts))


def build_debug_prompt(erro: str, logs: str, trecho: str) -> str:
    template = PROMPT_TEMPLATES["debug"]
    ctx = f"Código:\n{trecho}"
    return template.format(identity=AGENT_IDENTITY, context=ctx, error=erro, logs=logs)


def build_task_prompt(task_yaml: Dict[str, Any], status_anterior: str) -> str:
    task_info = json.dumps(task_yaml, ensure_ascii=False, indent=2)
    template = PROMPT_TEMPLATES.get(task_yaml.get("type"), PROMPT_TEMPLATES["create"])
    return template.format(identity=AGENT_IDENTITY, context=f"Tarefa:\n{task_info}\nStatus anterior:\n{status_anterior}")

