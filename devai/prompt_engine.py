from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Sequence

from .config import config, logger
import yaml

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
    identity = (
        "Você é o agente DevAI-R1.\n"
        "Seu papel é gerar código inteligente, testado e funcional com base no projeto atual.\n"
        "Trabalhe com precisão cirúrgica."
    )
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
        f"{identity}\n{proj_text}\n{mem_text}\n\n{graph_summary}\n\n"
        f"Ultimas ações:\n{acts}\n\n"
        f"Logs recentes:\n{logs}\n\nComando do usuário: {command}\n"
        "Vamos pensar passo a passo antes de responder." + extra
    )
    return prompt


def build_debug_prompt(error: str, logs: str, snippet: str) -> str:
    identity = (
        "Você é o agente DevAI-R1.\nTrabalhe com precisão cirúrgica."
    )
    return (
        f"{identity}\nErro detectado: {error}\nContexto:\n{snippet}\nLogs:\n{logs}\n"
        "Explique a causa e proponha correções passo a passo."
    )


def build_task_prompt(task_yaml: Dict[str, Any], status: str) -> str:
    identity = (
        "Você é o agente DevAI-R1.\nTrabalhe com precisão cirúrgica."
    )
    desc = task_yaml.get("description", "")
    ttype = task_yaml.get("type", "generic")
    template = TEMPLATES.get(ttype, "")
    return (
        f"{identity}\nTarefa: {task_yaml.get('name')} ({ttype})\n"
        f"Descrição: {desc}\nStatus anterior: {status}\n{template}"
    )

