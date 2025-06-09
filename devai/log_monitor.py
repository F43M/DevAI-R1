import asyncio
import json
import re
from datetime import datetime
from pathlib import Path

import aiofiles
import aiofiles.os

from .config import logger, config
from .memory import MemoryManager


class LogMonitor:
    def __init__(self, memory: MemoryManager, log_dir: str = "./logs"):
        self.memory = memory
        self.log_dir = Path(log_dir)
        self.patterns = {
            "error": r"ERROR|CRITICAL|FAILED|Exception|Traceback",
            "warning": r"WARNING|Deprecation",
            "performance": r"Timeout|Slow|Latency",
            "test_failure": r"FAILURES|AssertionError",
        }
        self.last_checked = datetime.now()
        self.state_file = self.log_dir / "log_monitor.state"
        try:
            if self.state_file.exists():
                self.file_positions = json.loads(self.state_file.read_text())
            else:
                self.file_positions = {}
        except Exception:
            self.file_positions = {}

    async def _persist_state(self) -> None:
        try:
            if not self.log_dir.exists():
                await aiofiles.os.makedirs(self.log_dir, exist_ok=True)
            async with aiofiles.open(self.state_file, "w") as sf:
                await sf.write(json.dumps(self.file_positions))
        except Exception as e:
            logger.error("Erro ao salvar estado do monitor", error=str(e))

    async def monitor_logs(self, interval: int | None = None):
        if interval is None:
            interval = config.LOG_MONITOR_INTERVAL
        while True:
            try:
                if not self.log_dir.exists():
                    await aiofiles.os.makedirs(self.log_dir, exist_ok=True)
                log_files = [f for f in self.log_dir.glob("*.log") if f.is_file()]
                for log_file in log_files:
                    await self._analyze_log_file(log_file)
                self.last_checked = datetime.now()
                await self._persist_state()
                await asyncio.sleep(interval)
            except Exception as e:
                logger.error("Erro no monitor de logs", error=str(e))
                await asyncio.sleep(30)

    async def _analyze_log_file(self, log_file: Path):
        try:
            pos = self.file_positions.get(str(log_file), 0)
            async with aiofiles.open(log_file, "r") as f:
                await f.seek(pos)
                lines = await f.readlines()
                self.file_positions[str(log_file)] = await f.tell()
            for line in lines:
                if not line.strip():
                    continue
                timestamp = datetime.now().isoformat()
                log_entry = json.loads(line) if line.startswith("{") else {"message": line.strip()}
                detected = []
                for p_type, pattern in self.patterns.items():
                    if re.search(pattern, line, re.IGNORECASE):
                        detected.append(p_type)
                if detected:
                    context = "\n".join(lines[max(0, lines.index(line) - 5) : lines.index(line) + 1])
                    self.memory.save(
                        {
                            "type": "log_analysis",
                            "content": f"Padrão detectado em logs: {', '.join(detected)}",
                            "metadata": {
                                "file": str(log_file),
                                "patterns": detected,
                                "context": context,
                                "timestamp": timestamp,
                            },
                            "tags": ["log"] + detected,
                        }
                    )
        except json.JSONDecodeError:
            logger.warning("Formato de log inválido", file=str(log_file))
        except Exception as e:
            logger.error("Erro ao analisar arquivo de log", file=str(log_file), error=str(e))
