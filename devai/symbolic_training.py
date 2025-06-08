from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Dict, List
import json

from .learning_engine import LESSONS_FILE

from .config import config, logger
from .memory import MemoryManager
from .analyzer import CodeAnalyzer
from .ai_model import AIModel


async def run_symbolic_training(
    analyzer: CodeAnalyzer,
    memory: MemoryManager,
    ai_model: AIModel,
    max_logs: int = 100,
) -> Dict[str, object]:
    """Execute the symbolic deep training cycle returning a friendly summary."""
    await analyzer.scan_app()
    code_root = Path(config.CODE_ROOT)

    try:
        items = json.loads(LESSONS_FILE.read_text()) if LESSONS_FILE.exists() else []
    except Exception:
        items = []
    items = sorted(items, key=lambda x: x.get("timestamp", ""), reverse=True)[:max_logs]
    lesson_map: Dict[str, List[str]] = {}
    for it in items:
        lesson_map.setdefault(it.get("arquivo", ""), []).append(it.get("erro", ""))

    total_files = 0
    history_files = 0
    at_risk = 0
    suggestions = 0
    sensitive: List[str] = []
    patterns: List[str] = []
    sugg_lines: List[str] = []
    unique_rules: Dict[str, None] = {}

    for file_path in code_root.rglob("*.py"):
        total_files += 1
        history = lesson_map.get(str(file_path), [])
        if history:
            history_files += 1
            sensitive.append(f"- {file_path} → [{len(history)} erros anteriores]")
        try:
            code = file_path.read_text()
        except Exception:
            code = ""
        meta = {"file": str(file_path)}
        if history:
            meta["historico_erros"] = history
        explain = await ai_model.safe_api_call(
            f"O que esse codigo faz?\n{code}",
            config.MAX_CONTEXT_LENGTH,
            code,
            memory,
        )
        bad = await ai_model.safe_api_call(
            f"Ha padroes ruins aqui?\n{code}",
            config.MAX_CONTEXT_LENGTH,
            code,
            memory,
        )
        risk = await ai_model.safe_api_call(
            f"Considerando erros passados {history}, que erro pode se repetir?\n{code}",
            config.MAX_CONTEXT_LENGTH,
            code,
            memory,
        )
        improve = await ai_model.safe_api_call(
            f"Como melhorar isso com seguranca? Pense passo a passo.\n{code}",
            config.MAX_CONTEXT_LENGTH,
            code,
            memory,
        )
        memory.save(
            {"type": "symbolic_training", "memory_type": "ponto_critico", "content": explain, "metadata": meta}
        )
        memory.save(
            {"type": "symbolic_training", "memory_type": "risco_reincidente", "content": risk, "metadata": meta}
        )
        memory.save(
            {"type": "symbolic_training", "memory_type": "refatoracao_sugerida", "content": improve, "metadata": meta}
        )
        if bad.strip():
            patterns.append(f"- {bad.strip().splitlines()[0]}")
        if risk.strip():
            at_risk += 1
        if improve.strip():
            suggestions += 1
            first = improve.strip().splitlines()[0]
            sugg_lines.append(f"- {first}")
            unique_rules[first] = None
        await asyncio.sleep(0)

    report = ["# Relatório de Treinamento Simbólico Profundo", ""]
    if sensitive:
        report.append("## Arquivos sensíveis")
        report.extend(sensitive)
        report.append("")
    if patterns:
        report.append("## Padrões negativos detectados")
        report.extend(patterns)
        report.append("")
    if sugg_lines:
        report.append("## Sugestões geradas")
        report.extend(sugg_lines)

    Path(config.LOG_DIR).mkdir(exist_ok=True)
    Path(config.LOG_DIR, "symbolic_training_report.md").write_text("\n".join(report))
    logger.info("Relatorio de treinamento salvo", file="symbolic_training_report.md")

    error_counts: Dict[str, int] = {}
    for it in items:
        typ = it.get("erro", "").split(":")[0].split()[0]
        if typ:
            error_counts[typ] = error_counts.get(typ, 0) + 1
    if error_counts:
        cause = ", ".join(f"{k} ({v})" for k, v in error_counts.items())
        cause_msg = f"Baseado em {len(items)} erros do tipo {cause}."
    else:
        cause_msg = "Regras adicionadas com base em logs de erro anteriores."

    rules = list(unique_rules.keys())
    lines = ["🧠 Treinamento Concluído", ""]
    if rules:
        lines.append(f"✅ {len(rules)} novas regras de qualidade adicionadas à base de conhecimento:")
        for i, r in enumerate(rules, 1):
            lines.append(f"📌 [{i}] {r}")
            rule_id = f"rule_{len(analyzer.learned_rules) + i}"
            analyzer.learned_rules[rule_id] = {
                "rule": r,
                "source": "user_correction:conversation_042",
            }
    else:
        lines.append("Nenhum aprendizado novo encontrado desta vez.")
    lines.append("")
    lines.append(f"🔎 Causa: {cause_msg}")

    return {
        "report": "\n".join(lines),
        "data": {
            "total_files": total_files,
            "arquivos_com_erro": history_files,
            "em_risco": at_risk,
            "sugestoes": suggestions,
            "new_rules": len(rules),
            "rules_added": rules,
            "errors_processed": len(items),
        },
    }
