import asyncio
import json
import re
from datetime import datetime
from pathlib import Path

import aiofiles
import aiofiles.os

from .config import logger
from .memory import MemoryManager


class LogMonitor:
    def __init__(self, memory: MemoryManager, log_dir: str = "./logs"):
        self.memory = memory
        self.log_dir = Path(log_dir)
        self.patterns = {
            "error": r"ERROR|CRITICAL|FAILED|Exception",
            "warning": r"WARNING|Deprecation",
            "performance": r"Timeout|Slow|Latency",
        }
        self.last_checked = datetime.now()

    async def monitor_logs(self):
        while True:
            try:
                if not self.log_dir.exists():
                    await aiofiles.os.makedirs(self.log_dir, exist_ok=True)
                log_files = [f for f in self.log_dir.glob("*.log") if f.is_file()]
                for log_file in log_files:
                    await self._analyze_log_file(log_file)
                self.last_checked = datetime.now()
                await asyncio.sleep(60)
            except Exception as e:
                logger.error("Erro no monitor de logs", error=str(e))
                await asyncio.sleep(30)

    async def _analyze_log_file(self, log_file: Path):
        try:
            async with aiofiles.open(log_file, "r") as f:
                lines = await f.readlines()
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
