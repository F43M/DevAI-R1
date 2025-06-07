from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Dict, List

from .config import config, logger
from .learning_engine import listar_licoes_negativas
from .memory import MemoryManager
from .analyzer import CodeAnalyzer
from .ai_model import AIModel


async def run_symbolic_training(
    analyzer: CodeAnalyzer,
    memory: MemoryManager,
    ai_model: AIModel,
) -> Dict[str, int]:
    """Execute the symbolic deep training cycle."""
    await analyzer.scan_app()
    code_root = Path(config.CODE_ROOT)
    total_files = 0
    history_files = 0
    at_risk = 0
    suggestions = 0
    sensitive: List[str] = []
    patterns: List[str] = []
    sugg_lines: List[str] = []

    for file_path in code_root.rglob("*.py"):
        total_files += 1
        history = listar_licoes_negativas(str(file_path))
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
        explain = await ai_model.generate(f"O que esse codigo faz?\n{code}")
        bad = await ai_model.generate(f"Ha padroes ruins aqui?\n{code}")
        risk = await ai_model.generate(
            f"Considerando erros passados {history}, que erro pode se repetir?\n{code}"
        )
        improve = await ai_model.generate(
            f"Como melhorar isso com seguranca? Pense passo a passo.\n{code}"
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
            sugg_lines.append(f"- {improve.strip().splitlines()[0]}")
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
    return {
        "total_files": total_files,
        "arquivos_com_erro": history_files,
        "em_risco": at_risk,
        "sugestoes": suggestions,
    }
