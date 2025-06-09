from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict

from .config import config, logger
from .memory import MemoryManager
from .analyzer import CodeAnalyzer
from .ai_model import AIModel
from .symbolic_training import run_symbolic_training
from collections import defaultdict

# track how many cycles executed per stage
stage_counters: Dict[str, int] = defaultdict(int)


async def auto_monitor_cycle(
    analyzer: CodeAnalyzer,
    memory: MemoryManager,
    ai_model: AIModel,
) -> Dict[str, Any]:
    """Run an internal health check and trigger symbolic training if needed."""
    log_dir = Path(config.LOG_DIR)
    log_dir.mkdir(exist_ok=True)
    monitor_log = log_dir / "monitoring_history.md"
    decision_log = log_dir / "self_triggered_analysis.md"
    training_report = log_dir / "symbolic_training_report.md"

    logs: list[str] = []

    def _log(msg: str) -> None:
        logs.append(msg)
        logger.info(msg)

    stage = "monitor"
    stage_counters[stage] += 1
    limit = getattr(config, "MAX_CYCLES_PER_STAGE", 10)
    if stage_counters[stage] > limit:
        logger.warning(f"Estágio {stage} excedeu limite de ciclos.")
        return {"report": "limite excedido", "logs": "", "data": {"training_executed": False}}

    now = datetime.now()
    last_training = datetime.fromtimestamp(0)
    if training_report.exists():
        last_training = datetime.fromtimestamp(training_report.stat().st_mtime)

    _log("🔎 Verificando novos logs de erro…")
    day_ago = (now - timedelta(hours=24)).isoformat()
    cursor = memory.conn.cursor()
    cursor.execute(
        "SELECT COUNT(*) FROM memory WHERE memory_type IN ('erro_reincidente','falha') AND created_at >= ?",
        (day_ago,),
    )
    failures = cursor.fetchone()[0]
    _log("OK")

    _log("📂 Checando arquivos modificados…")
    changed_files = [
        f
        for f in Path(config.CODE_ROOT).rglob("*.py")
        if f.stat().st_mtime > last_training.timestamp()
    ]
    new_files = len(changed_files)
    _log("OK")

    hours_since = (now - last_training).total_seconds() / 3600.0

    reasons = []
    if failures >= config.AUTO_MONITOR_FAILURES:
        reasons.append(f"{failures} falhas em 24h")
    if new_files >= config.AUTO_MONITOR_FILES:
        reasons.append(f"{new_files} arquivos modificados")
    if hours_since > config.AUTO_MONITOR_HOURS:
        reasons.append(f"{int(hours_since)}h sem treinamento")

    triggered = bool(reasons)
    training_executed = False
    result_data: Dict[str, Any] = {
        "triggered": triggered,
        "reason": "; ".join(reasons) if reasons else "criterios nao atingidos",
        "training_executed": False,
        "new_rules": 0,
        "errors_processed": 0,
        "rule_sources": {},
    }

    _log("🧠 Avaliando necessidade de novo treinamento simbólico…")
    if triggered:
        training_executed = True
        _log("EXECUTADO")
        logger.info("Monitor acionou treinamento simbólico")
        from .learning_engine import LearningEngine

        engine = LearningEngine(analyzer, memory, ai_model)
        training_result = await run_symbolic_training(
            analyzer, memory, ai_model, learning_engine=engine
        )
        result_data.update(training_result.get("data", {}))
        result_data["training_executed"] = True
        decision_log.write_text(
            f"[{now.isoformat()}] Trigger: {'; '.join(reasons)}\n",
        )
        monitor_log.open("a").write(
            f"[{now.strftime('%Y-%m-%d %H:%M')}] Auto-análise executada: {new_files} arquivos alterados + {failures} falhas = treinamento simbólico disparado\n"
        )
    else:
        _log("Nenhum treinamento necessário")
        monitor_log.open("a").write(
            f"[{now.strftime('%Y-%m-%d %H:%M')}] Auto-análise executada: {new_files} arquivos alterados + {failures} falhas\n"
        )

    lines = ["🧭 Autoavaliação Concluída", ""]
    if training_executed:
        lines.append("🔁 Treinamento simbólico automático executado.")
        rules = result_data.get("new_rules", 0)
        if rules:
            plural = "s" if rules != 1 else ""
            lines.append(f"📌 {rules} nova{plural} regra{plural} aprendida.")
            sources = result_data.get("rule_sources", {})
            for idx, (rule, info) in enumerate(sources.items(), 1):
                arquivos = ", ".join(info.get("files", [])) or "?"
                log_info = ", ".join(info.get("logs", []))
                if log_info:
                    lines.append(f"🔗 Regra {idx} em {arquivos} (logs: {log_info})")
                else:
                    lines.append(f"🔗 Regra {idx} em {arquivos}")
        else:
            lines.append("🧠 Regras simbólicas atualizadas com base em novos erros.")
        logs_proc = result_data.get("errors_processed", 0)
        if logs_proc:
            lines.append(f"📂 {logs_proc} novos logs de erro processados com sucesso.")
        lines.append("✅ Sistema atualizado com base em dados recentes.")
    else:
        lines.append("✅ Nenhuma anomalia identificada.")
        lines.append("🧠 Nenhum aprendizado novo foi necessário neste ciclo.")
        lines.append("ℹ️ Sistema está atualizado e operando normalmente.")

    await asyncio.sleep(0)

    result = {
        "report": "\n".join(lines),
        "logs": "\n".join(logs),
        "data": result_data,
    }
    cur = memory.conn.cursor()
    cur.execute(
        "INSERT INTO monitoring_history (timestamp, reason, training_executed, new_rules) VALUES (?, ?, ?, ?)",
        (
            now.isoformat(),
            result_data["reason"],
            int(training_executed),
            result_data.get("new_rules", 0),
        ),
    )
    memory.conn.commit()
    memory.save(
        {
            "type": "monitor",
            "memory_type": "monitoring",
            "content": result["report"],
            "metadata": {
                "reason": result_data["reason"],
                "training_executed": training_executed,
            },
        }
    )
    return result
