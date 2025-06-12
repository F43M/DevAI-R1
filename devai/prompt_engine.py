from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Sequence
import re
import asyncio

from .config import config, logger
from .feedback import listar_preferencias


_logs_cache: Dict[int, Dict[str, Any]] = {}

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
    tokens = 0
    limit = getattr(config, "MAX_PROMPT_TOKENS", 1000)
    for m in memories[:3]:
        content = m.get("content", "")
        name = m.get("metadata", {}).get("name", "?")
        score = m.get("similarity_score", 0.0)
        tags = ",".join(m.get("tags", []))
        text = f"// Memória relevante: {content}\n// Função: {name}, Similaridade: {score:.2f} Tags:{tags}"
        count = len(text.split())
        if tokens + count > limit:
            break
        tokens += count
        parts.append(text)
    return "\n".join(parts)


def collect_recent_logs(lines: int = 50) -> str:
    """Return the last N lines from the main log file."""
    log_file = Path(config.LOG_DIR) / "ai_core.log"
    if not log_file.exists():
        return ""
    mtime = log_file.stat().st_mtime
    cached = _logs_cache.get(lines)
    if cached and cached.get("mtime") == mtime:
        return cached["data"]
    data = log_file.read_text().splitlines()[-lines:]
    text = "\n".join(data)
    _logs_cache[lines] = {"mtime": mtime, "data": text}
    return text


async def collect_recent_logs_async(lines: int = 50) -> str:
    """Async version of log collection."""
    from aiofiles import open as aio_open  # type: ignore

    log_file = Path(config.LOG_DIR) / "ai_core.log"
    if not log_file.exists():
        return ""
    mtime = log_file.stat().st_mtime
    cached = _logs_cache.get(lines)
    if cached and cached.get("mtime") == mtime:
        return cached["data"]
    async with aio_open(log_file, "r") as f:
        data = (await f.read()).splitlines()[-lines:]
    text = "\n".join(data)
    _logs_cache[lines] = {"mtime": mtime, "data": text}
    return text


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


def build_dynamic_prompt(
    query: str,
    context_blocks: Dict[str, Any],
    mode: str,
    intent: str | None = None,
) -> str:
    """Return a prompt using only context relevant to the query."""
    q = query.lower()
    included: list[str] = []
    reasons: list[str] = []

    def _add(name: str, text: str | None, reason: str) -> None:
        if text:
            parts.append(text)
            included.append(name)
            reasons.append(f"{name}:{reason}")

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
    _add("memories", mem_text, "memorias relevantes")

    symbolic = context_blocks.get("symbolic_memories") or []
    sym_text = _format_memories(symbolic)
    _add("memorias_simbolicas", sym_text, "memorias simbolicas")

    if intent == "architecture" or any(
        k in q for k in ["arquitetura", "módulo", "modulo", "depend", "grafo"]
    ):
        graph = context_blocks.get("graph")
        _add("grafo_simbólico", graph, "pergunta de arquitetura")

    if intent == "debug" or any(k in q for k in ["erro", "falha", "log", "stack", "execu"]):
        logs = context_blocks.get("logs")
        _add("logs_recentes", f"Logs recentes:\n{logs}" if logs else None, "pergunta de erro")
        actions = context_blocks.get("actions")
        if actions:
            acts = "\n".join(f"- {a.get('task')}" for a in actions[-3:])
            _add("ultima_acao", f"Ultimas ações:\n{acts}", "pergunta de erro")

    code = context_blocks.get("code")
    if code and (
        intent in {"edit", "create"}
        or any(k in q for k in ["melhor", "refator", "código", "codigo"])
    ):
        _add("trecho_codigo", code, "contexto de codigo")

    if len(included) == 0:
        graph = context_blocks.get("graph")
        _add("grafo_simbólico", graph, "fallback")
        actions = context_blocks.get("actions")
        if actions:
            acts = "\n".join(f"- {a.get('task')}" for a in actions[-3:])
            _add("ultima_acao", f"Ultimas ações:\n{acts}", "fallback")
        logs = context_blocks.get("logs")
        _add("logs_recentes", f"Logs recentes:\n{logs}" if logs else None, "fallback")
        logger.info(
            "Fallback: prompt completo não foi simplificado por falta de sinal contextual"
        )

    prompt = "\n\n".join(p for p in parts if p)
    prompt += f"\n\nComando do usuário: {query}\n"

    keywords = ["por que", "por quê", "analise", "detalhe", "explique", "entenda"]
    explain_intents = {"debug", "architecture", "review"}
    if mode == "deep" or intent in explain_intents or any(k in q for k in keywords):
        prompt += "Explique antes de responder." + extra
    else:
        prompt += extra
    logger.info("Prompt dinâmico", included_blocks=included, reasons=reasons, mode=mode)
    return prompt


async def build_dynamic_prompt_async(
    query: str,
    context_blocks: Dict[str, Any],
    mode: str,
    intent: str | None = None,
) -> str:
    """Async wrapper for build_dynamic_prompt."""
    return await asyncio.to_thread(build_dynamic_prompt, query, context_blocks, mode, intent)


def split_plan_response(text: str) -> tuple[str, str]:
    """Split plan and final answer if both are present."""
    m = re.search(r"===\s*RESPOSTA\s*===", text, re.IGNORECASE)
    if m:
        return text[: m.start()].strip(), text[m.end():].strip()
    return text.strip(), ""


async def gather_context_async(ai: Any, query: str) -> tuple[Dict[str, Any], Sequence[Dict]]:
    """Collect memories, suggestions, graph summary and logs concurrently."""
    mem_task = asyncio.to_thread(ai.memory.search, query, level="short")
    sugg_task = asyncio.to_thread(ai.memory.search, query, top_k=1)
    graph_fn = getattr(ai.analyzer, "graph_summary_async", None)
    graph_task = graph_fn() if graph_fn else asyncio.sleep(0, result="")
    logs_task = collect_recent_logs_async()
    actions = ai.tasks.last_actions()
    mems, suggestions, graph_summary, logs = await asyncio.gather(
        mem_task, sugg_task, graph_task, logs_task
    )
    context_blocks = {
        "memories": mems,
        "graph": graph_summary,
        "actions": actions,
        "logs": logs,
    }
    return context_blocks, suggestions


async def generate_plan_async(
    ai: Any,
    query: str,
    context_blocks: Dict[str, Any],
    intent: str | None = None,
) -> str:
    """Generate a reasoning plan for the given query and context."""
    prompt = build_dynamic_prompt(query, context_blocks, "deep", intent=intent)
    system_msg = (
        "Você é um assistente especialista em análise de código. "
        "Elabore um plano de ação passo a passo para atender ao pedido do usuário."
    )
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": prompt},
    ]
    raw = await ai.ai_model.safe_api_call(
        messages, 4096, prompt, ai.memory, temperature=0.2
    )
    plan, _ = split_plan_response(raw)
    return plan


async def generate_final_async(
    ai: Any,
    query: str,
    context_blocks: Dict[str, Any],
    plan: str,
    history: Sequence[Dict] | None = None,
    intent: str | None = None,
) -> str:
    """Generate the final answer using a pre-computed plan."""
    prompt = build_dynamic_prompt(query, context_blocks, "deep", intent=intent)
    prompt = f"{plan}\n{prompt}"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_CONTEXT},
        *(history or []),
        {"role": "user", "content": prompt},
    ]
    result, _ = await asyncio.gather(
        ai.ai_model.safe_api_call(
            messages, config.MAX_CONTEXT_LENGTH, prompt, ai.memory
        ),
        ai._prefetch_related(query),
    )
    return result.strip()

