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
    pending_items = [i for i in items if not i.get("processed")]
    pending_items = sorted(
        pending_items, key=lambda x: x.get("timestamp", ""), reverse=True
    )[:max_logs]
    lesson_map: Dict[str, List[str]] = {}
    for it in pending_items:
        lesson_map.setdefault(it.get("arquivo", ""), []).append(it.get("erro", ""))

    total_files = 0
    history_files = 0
    at_risk = 0
    suggestions = 0
    sensitive: List[str] = []
    patterns: List[str] = []
    sugg_lines: List[str] = []
    unique_rules: Dict[str, Dict[str, set[str] | set[int]]] = {}

    file_line_map: Dict[str, int] = {}
    for chunk in analyzer.code_chunks.values():
        f = chunk.get("file")
        if not f:
            continue
        line = chunk.get("line_start", 1)
        if f not in file_line_map:
            file_line_map[f] = line
        else:
            file_line_map[f] = min(file_line_map[f], line)

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
            {
                "type": "symbolic_training",
                "memory_type": "ponto_critico",
                "content": explain,
                "metadata": meta,
            }
        )
        memory.save(
            {
                "type": "symbolic_training",
                "memory_type": "risco_reincidente",
                "content": risk,
                "metadata": meta,
            }
        )
        memory.save(
            {
                "type": "symbolic_training",
                "memory_type": "refatoracao_sugerida",
                "content": improve,
                "metadata": meta,
            }
        )
        if bad.strip():
            patterns.append(f"- {bad.strip().splitlines()[0]}")
        if risk.strip():
            at_risk += 1
        if improve.strip():
            suggestions += 1
            first = improve.strip().splitlines()[0]
            sugg_lines.append(f"- {first}")
            if first not in unique_rules:
                unique_rules[first] = {"files": set(), "logs": set(), "lines": set()}
            unique_rules[first]["files"].add(str(file_path))
            unique_rules[first]["logs"].update(history)
            line_no = file_line_map.get(str(file_path), 1)
            unique_rules[first]["lines"].add(line_no)
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
    if unique_rules:
        report.append("")
        report.append("## Origem das regras")
        for idx, (rule, info) in enumerate(unique_rules.items(), 1):
            files = ", ".join(sorted(info["files"]))
            logs_ref = ", ".join(sorted(info["logs"]))
            log_part = f" | Logs: {logs_ref}" if logs_ref else ""
            report.append(f"- Regra {idx}: {files}{log_part}")

    Path(config.LOG_DIR).mkdir(exist_ok=True)
    Path(config.LOG_DIR, "symbolic_training_report.md").write_text("\n".join(report))
    logger.info("Relatorio de treinamento salvo", file="symbolic_training_report.md")

    error_counts: Dict[str, int] = {}
    for it in pending_items:
        typ = it.get("erro", "").split(":")[0].split()[0]
        if typ:
            error_counts[typ] = error_counts.get(typ, 0) + 1
    if error_counts:
        cause = ", ".join(f"{k} ({v})" for k, v in error_counts.items())
        cause_msg = f"Baseado em {len(pending_items)} erros do tipo {cause}."
    else:
        cause_msg = "Regras adicionadas com base em logs de erro anteriores."

    rules = list(unique_rules.items())
    lines = ["🧠 Treinamento Concluído", ""]
    if rules:
        lines.append(
            f"✅ {len(rules)} novas regras de qualidade adicionadas à base de conhecimento:"
        )
        for i, (r, info) in enumerate(rules, 1):
            lines.append(f"📌 [{i}] {r}")
            arquivos = ", ".join(sorted(info["files"]))
            logs_ref = ", ".join(sorted(info["logs"]))
            log_part = f" (logs: {logs_ref})" if logs_ref else ""
            lines.append(f"🔗 Origem: {arquivos}{log_part}")
            rule_id = f"rule_{len(analyzer.learned_rules) + i}"
            analyzer.learned_rules[rule_id] = {
                "rule": r,
                "source": {
                    "files": list(info["files"]),
                    "lines": list(info.get("lines", [])),
                    "logs": list(info["logs"]),
                },
            }
            memory.save(
                {
                    "type": "learned_rule",
                    "memory_type": "rule",
                    "content": r,
                    "metadata": {
                        "files": list(info["files"]),
                        "lines": list(info.get("lines", [])),
                        "logs": list(info["logs"]),
                    },
                    "tags": ["rule", "learning"],
                }
            )
    else:
        lines.append("Nenhum aprendizado novo encontrado desta vez.")
    lines.append("")
    lines.append(f"🔎 Causa: {cause_msg}")

    # mark logs as processed
    for entry in items:
        if entry in pending_items:
            entry["processed"] = True
    try:
        LESSONS_FILE.write_text(json.dumps(items, indent=2))
    except Exception:
        logger.warning("Nao foi possivel atualizar lessons.json")

    return {
        "report": "\n".join(lines),
        "data": {
            "total_files": total_files,
            "arquivos_com_erro": history_files,
            "em_risco": at_risk,
            "sugestoes": suggestions,
            "new_rules": len(rules),
            "rules_added": [r for r, _ in rules],
            "rule_sources": {
                r: {
                    "files": list(info["files"]),
                    "lines": list(info.get("lines", [])),
                    "logs": list(info["logs"]),
                }
                for r, info in rules
            },
            "errors_processed": len(pending_items),
        },
    }
