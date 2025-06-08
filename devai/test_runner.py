import subprocess
import resource
from pathlib import Path
from typing import Tuple

from .config import config, logger
from .sandbox import Sandbox


def _preexec(cpu: int, mem: int) -> None:
    """Apply resource limits before executing the child process."""
    if cpu > 0:
        resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
    if mem > 0:
        resource.setrlimit(resource.RLIMIT_AS, (mem, mem))


def run_pytest(path: str | Path, timeout: int = 30) -> Tuple[bool, str]:
    """Execute pytest with optional isolation."""
    cwd = Path(path)
    if config.TESTS_USE_ISOLATION:
        sb = Sandbox(
            image="python:3.10-slim",
            cpus=str(config.TEST_CPU_LIMIT or 1),
            memory=f"{config.TEST_MEMORY_LIMIT_MB or 512}m",
        )
        try:
            out = sb.run(["pytest", "-q"], timeout=timeout)
            return True, out
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
            return proc.returncode == 0, proc.stdout
        except subprocess.TimeoutExpired:
            return False, f"\U0001f6d1 Tempo excedido: mais de {timeout}s"
        except Exception as e:  # pragma: no cover - unexpected errors
            return False, str(e)
