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


async def auto_monitor_cycle(
    analyzer: CodeAnalyzer,
    memory: MemoryManager,
    ai_model: AIModel,
) -> Dict[str, Any]:
    """Check internal signals and trigger symbolic training if needed."""
    log_dir = Path(config.LOG_DIR)
    log_dir.mkdir(exist_ok=True)
    monitor_log = log_dir / "monitoring_history.md"
    decision_log = log_dir / "self_triggered_analysis.md"
    training_report = log_dir / "symbolic_training_report.md"

    now = datetime.now()
    last_training = datetime.fromtimestamp(0)
    if training_report.exists():
        last_training = datetime.fromtimestamp(training_report.stat().st_mtime)

    # Count negative memories in the last 24h
    day_ago = (now - timedelta(hours=24)).isoformat()
    cursor = memory.conn.cursor()
    cursor.execute(
        "SELECT COUNT(*) FROM memory WHERE memory_type IN ('erro_reincidente','falha') AND created_at >= ?",
        (day_ago,),
    )
    failures = cursor.fetchone()[0]

    # Files changed since last training
    changed_files = [
        f
        for f in Path(config.CODE_ROOT).rglob("*.py")
        if f.stat().st_mtime > last_training.timestamp()
    ]
    new_files = len(changed_files)

    hours_since = (now - last_training).total_seconds() / 3600.0

    reasons = []
    if failures >= 3:
        reasons.append(f"{failures} falhas em 24h")
    if new_files >= 5:
        reasons.append(f"{new_files} arquivos modificados")
    if hours_since > 72:
        reasons.append(f"{int(hours_since)}h sem treinamento")

    triggered = bool(reasons)
    training_executed = False
    if triggered:
        training_executed = True
        logger.info("Monitor acionou treinamento simbólico")
        await run_symbolic_training(analyzer, memory, ai_model)
        decision_log.write_text(
            f"[{now.isoformat()}] Trigger: {'; '.join(reasons)}\n",
        )
        monitor_log.open("a").write(
            f"[{now.strftime('%Y-%m-%d %H:%M')}] Auto-análise executada: {new_files} arquivos alterados + {failures} falhas = treinamento simbólico disparado\n"
        )
    else:
        monitor_log.open("a").write(
            f"[{now.strftime('%Y-%m-%d %H:%M')}] Auto-análise executada: {new_files} arquivos alterados + {failures} falhas\n"
        )

    return {
        "triggered": triggered,
        "reason": "; ".join(reasons) if reasons else "criterios nao atingidos",
        "training_executed": training_executed,
    }
