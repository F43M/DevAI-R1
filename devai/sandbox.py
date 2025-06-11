import os
import subprocess
import shutil
import platform
from typing import List

from .config import config, logger


class Sandbox:
    """Isolated execution environment using containers or local execution."""

    def __init__(self, image: str | None = None, cpus: str | None = None, memory: str | None = None) -> None:
        self.image = image or getattr(config, "SANDBOX_IMAGE", "python:3.10-slim")
        self.cpus = str(cpus or getattr(config, "SANDBOX_CPUS", "1"))
        self.memory = memory or getattr(config, "SANDBOX_MEMORY", "512m")
        docker_path = shutil.which("docker") or shutil.which("docker.exe")
        system = platform.system()
        if system in {"Linux", "Darwin"}:
            self.enabled = bool(docker_path)
        elif system == "Windows":
            if docker_path:
                self.enabled = True
            else:
                logger.warning(
                    "Docker Desktop não encontrado. Comandos rodarão sem isolamento."
                )
                self.enabled = False
        else:
            self.enabled = bool(docker_path)
        self._processes: List[subprocess.Popen] = []

    def run_command(self, cmd: List[str], timeout: int = 30) -> str:
        """Execute ``cmd`` isolated in Docker or directly if disabled."""
        if self.enabled:
            docker_cmd = [
                "docker",
                "run",
                "--rm",
                "-v",
                f"{os.getcwd()}:/app",
                "--workdir",
                "/app",
                "--network",
                "none",
                "--cpus",
                self.cpus,
                "--memory",
                self.memory,
                self.image,
                *cmd,
            ]
        else:
            docker_cmd = cmd

        proc = subprocess.Popen(
            docker_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        self._processes.append(proc)
        try:
            out, _ = proc.communicate(timeout=timeout)
            return out
        except subprocess.TimeoutExpired as e:
            proc.kill()
            raise TimeoutError(f"Command timed out after {timeout}s") from e
        finally:
            if proc in self._processes:
                self._processes.remove(proc)

    def run(self, command: List[str], timeout: int = 30) -> str:
        """Backward compatible wrapper around :func:`run_command`."""
        return self.run_command(command, timeout)

    def shutdown(self) -> None:
        for p in list(self._processes):
            p.kill()
            self._processes.remove(p)

    def __enter__(self) -> "Sandbox":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.shutdown()


def run_in_sandbox(command: List[str], timeout: int = 30) -> str:
    """Execute ``command`` using :class:`Sandbox` with default configuration."""
    sb = Sandbox()
    try:
        return sb.run(command, timeout)
    finally:
        sb.shutdown()

