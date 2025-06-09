import subprocess
import resource
from datetime import datetime
from pathlib import Path
from typing import Tuple
import re

from .config import config, logger
from .sandbox import Sandbox


def _parse_output(output: str) -> str:
    """Highlight individual failures and produce a concise summary."""
    fail_re = re.compile(r"^FAILED (.+::.+?) - (.+)$")
    failures = []
    summary = ""
    for line in output.splitlines():
        m = fail_re.match(line.strip())
        if m:
            failures.append(f"{m.group(1)} - {m.group(2)}")
        if not summary and ("failed" in line or "passed" in line) and "in" in line:
            summary = line.strip()
    if not failures and not summary:
        return output
    parts = []
    if failures:
        parts.append("Falhas encontradas:")
        parts.extend(f"- {f}" for f in failures)
    if summary:
        parts.append(f"Resumo: {summary}")
    return "\n".join(parts)


def _preexec(cpu: int, mem: int) -> None:
    """Apply resource limits before executing the child process."""
    if cpu > 0:
        resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
    if mem > 0:
        resource.setrlimit(resource.RLIMIT_AS, (mem, mem))


def run_pytest(path: str | Path, timeout: int = 30) -> Tuple[bool, str]:
    """Execute pytest with optional isolation."""
    cwd = Path(path)
    sb = Sandbox(
        image="python:3.10-slim",
        cpus=str(config.TEST_CPU_LIMIT or 1),
        memory=f"{config.TEST_MEMORY_LIMIT_MB or 512}m",
    )
    if config.TESTS_USE_ISOLATION and sb.enabled:
        try:
            out = sb.run(["pytest", "-q"], timeout=timeout)
            log_dir = Path(config.LOG_DIR)
            log_dir.mkdir(exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            (log_dir / f"sandbox_{ts}.log").write_text(out)
            return True, _parse_output(out)
        except Exception as e:
            logger.error("Erro na sandbox", error=str(e))
            return False, str(e)
        finally:
            sb.shutdown()
    else:

        def _limits() -> None:
            _preexec(config.TEST_CPU_LIMIT, config.TEST_MEMORY_LIMIT_MB * 1024 * 1024)

        try:
            proc = subprocess.run(
                ["pytest", "-q"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                cwd=cwd,
                timeout=timeout,
                text=True,
                preexec_fn=_limits,
            )
            return proc.returncode == 0, _parse_output(proc.stdout)
        except subprocess.TimeoutExpired:
            return False, f"\U0001f6d1 Tempo excedido: mais de {timeout}s"
        except Exception as e:  # pragma: no cover - unexpected errors
            return False, str(e)
