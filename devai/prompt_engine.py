from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Sequence

from .config import config, logger
from .feedback import listar_preferencias

SYSTEM_PROMPT_CONTEXT = (
    "Você atua como engenheiro simbólico. "
    "Você é um assistente de desenvolvimento focado em manter continuidade de raciocínio. "
    "Sempre explique antes de agir e justifique cada modificação. "
    "Estrutura: contexto simbólico -> raciocínio -> código."
)
try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - fallback when PyYAML is missing
    from . import yaml_fallback as yaml

TEMPLATES: Dict[str, str] = {
    "create": "Crie o código solicitado com comentários e exemplos.",
    "edit": "Edite o trecho conforme indicado mantendo estilo e testes.",
    "debug": "Analise o erro e proponha correção passo a passo.",
    "architecture": "Explique a arquitetura e sugira melhorias.",
    "tests": "Gere testes unitários completos para o código em questão.",
    "review": "Faça revisão de código apontando problemas e correções.",
}


def _load_project_identity() -> tuple[str, dict]:
    path = Path("project_identity.yaml")
    if path.exists():
        data = yaml.safe_load(path.read_text()) or {}
        text = data.get("objetivo", "")
        return text, data
    return "", {}


def _format_memories(memories: Sequence[Dict]) -> str:
    parts = []
    for m in memories[:3]:
        content = m.get("content", "")
        name = m.get("metadata", {}).get("name", "?")
        score = m.get("similarity_score", 0.0)
        tags = ",".join(m.get("tags", []))
        parts.append(
            f"// Memória relevante: {content}\n// Função: {name}, Similaridade: {score:.2f} Tags:{tags}"
        )
    return "\n".join(parts)


def collect_recent_logs(lines: int = 50) -> str:
    """Return the last N lines from the main log file."""
    log_file = Path(config.LOG_DIR) / "ai_core.log"
    if not log_file.exists():
        return ""
    data = log_file.read_text().splitlines()[-lines:]
    return "\n".join(data)


def build_cot_prompt(
    command: str,
    graph_summary: str,
    memories: Sequence[Dict],
    actions: Sequence[Dict],
    logs: str,
) -> str:
    """Compose a reasoning prompt with full context."""
    prefs = listar_preferencias()
    pref_text = ""
    if prefs:
        pref_lines = "\n".join(f"- {p}" for p in prefs)
        pref_text = (
            "Estas são as preferências do usuário para estilo de código:\n"
            f"{pref_lines}\n\n"
        )
    identity = SYSTEM_PROMPT_CONTEXT
    proj_text, proj_cfg = _load_project_identity()
    mem_text = _format_memories(memories)
    acts = "\n".join(
        f"- {a.get('task')}" for a in list(actions)[-3:]
    )
    step_mode = proj_cfg.get("task_mode", "normal") == "step_by_step"
    extra = (
        "\nPor favor, forneça um plano de execução, checklist e questione informação faltante." if step_mode else ""
    )
    prompt = (
        f"{pref_text}{identity}\n{proj_text}\n{mem_text}\n\n{graph_summary}\n\n"
        f"Ultimas ações:\n{acts}\n\n"
        f"Logs recentes:\n{logs}\n\nComando do usuário: {command}\n"
        "Antes de gerar código, descreva em 2-3 etapas a lógica da solução. "
        "Depois apresente o código com uma justificativa." + extra
    )
    return prompt


def build_debug_prompt(error: str, logs: str, snippet: str) -> str:
    identity = SYSTEM_PROMPT_CONTEXT
    return (
        f"{identity}\nErro detectado: {error}\nContexto:\n{snippet}\nLogs:\n{logs}\n"
        "Explique a causa e proponha correções passo a passo."
    )


def build_task_prompt(task_yaml: Dict[str, Any], status: str) -> str:
    identity = SYSTEM_PROMPT_CONTEXT
    desc = task_yaml.get("description", "")
    ttype = task_yaml.get("type", "generic")
    template = TEMPLATES.get(ttype, "")
    return (
        f"{identity}\nTarefa: {task_yaml.get('name')} ({ttype})\n"
        f"Descrição: {desc}\nStatus anterior: {status}\n{template}"
    )


def build_dynamic_prompt(query: str, context_blocks: Dict[str, Any], mode: str) -> str:
    """Return a prompt using only context relevant to the query."""
    q = query.lower()
    included = []

    prefs = listar_preferencias()
    pref_text = ""
    if prefs:
        pref_lines = "\n".join(f"- {p}" for p in prefs)
        pref_text = (
            "Estas são as preferências do usuário para estilo de código:\n"
            f"{pref_lines}\n\n"
        )
    identity = SYSTEM_PROMPT_CONTEXT
    proj_text, proj_cfg = _load_project_identity()
    step_mode = proj_cfg.get("task_mode", "normal") == "step_by_step"
    extra = (
        "\nPor favor, forneça um plano de execução, checklist e questione informação faltante."  # noqa: E501
        if step_mode
        else ""
    )

    parts = [f"{pref_text}{identity}\n{proj_text}"]

    memories = context_blocks.get("memories") or []
    mem_text = _format_memories(memories)
    if mem_text:
        parts.append(mem_text)
        included.append("memories")

    symbolic = context_blocks.get("symbolic_memories") or []
    sym_text = _format_memories(symbolic)
    if sym_text:
        parts.append(sym_text)
        included.append("memorias_simbolicas")

    if any(k in q for k in ["arquitetura", "módulo", "modulo", "depend", "grafo"]):
        graph = context_blocks.get("graph")
        if graph:
            parts.append(graph)
            included.append("grafo_simbólico")

    if any(k in q for k in ["erro", "falha", "log", "stack", "execu"]):
        logs = context_blocks.get("logs")
        if logs:
            parts.append(f"Logs recentes:\n{logs}")
            included.append("logs_recentes")
        actions = context_blocks.get("actions")
        if actions:
            acts = "\n".join(f"- {a.get('task')}" for a in actions[-3:])
            parts.append(f"Ultimas ações:\n{acts}")
            included.append("ultima_acao")

    code = context_blocks.get("code")
    if code and any(k in q for k in ["melhor", "refator", "código", "codigo"]):
        parts.append(code)
        included.append("trecho_codigo")

    if len(included) == 0:
        graph = context_blocks.get("graph")
        if graph:
            parts.append(graph)
        actions = context_blocks.get("actions")
        if actions:
            acts = "\n".join(f"- {a.get('task')}" for a in actions[-3:])
            parts.append(f"Ultimas ações:\n{acts}")
        logs = context_blocks.get("logs")
        if logs:
            parts.append(f"Logs recentes:\n{logs}")
        logger.info(
            "Fallback: prompt completo não foi simplificado por falta de sinal contextual"
        )

    prompt = "\n\n".join(p for p in parts if p)
    prompt += f"\n\nComando do usuário: {query}\n"

    keywords = ["por que", "por quê", "analise", "detalhe", "explique", "entenda"]
    if mode == "deep" or any(k in q for k in keywords):
        prompt += "Explique antes de responder." + extra
    else:
        prompt += extra

    logger.info("Prompt dinâmico", included_blocks=included, mode=mode)
    return prompt

