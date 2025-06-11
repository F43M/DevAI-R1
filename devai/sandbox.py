import os
import subprocess
import shutil
import platform
from typing import List
from uuid import uuid4

from .config import config, logger


class Sandbox:
    """Isolated execution environment using containers or local execution."""

    def __init__(
        self,
        image: str | None = None,
        cpus: str | None = None,
        memory: str | None = None,
        network: str | None = None,
        allowed_hosts: list[str] | None = None,
    ) -> None:
        self.image = image or getattr(config, "SANDBOX_IMAGE", "python:3.10-slim")
        self.cpus = str(cpus or getattr(config, "SANDBOX_CPUS", "1"))
        self.memory = memory or getattr(config, "SANDBOX_MEMORY", "512m")
        self.network = network or getattr(config, "SANDBOX_NETWORK", "none")
        self.allowed_hosts = allowed_hosts or list(
            getattr(config, "SANDBOX_ALLOWED_HOSTS", [])
        )
        docker_path = shutil.which("docker") or shutil.which("docker.exe")
        system = platform.system()
        if system in {"Linux", "Darwin"}:
            self.enabled = bool(docker_path)
        elif system == "Windows":
            if docker_path:
                self.enabled = True
            else:
                logger.warning(
                    "Docker Desktop/WSL2 não encontrado. Comandos rodarão sem isolamento, limites ou controle de rede."
                )
                self.enabled = False
        else:
            self.enabled = bool(docker_path)
        self._processes: List[subprocess.Popen] = []

    def run_command(self, cmd: List[str], timeout: int = 30) -> str:
        """Execute ``cmd`` isolated in Docker or directly if disabled."""
        if self.enabled:
            net = self.network
            tmp_net = None
            chain = None
            if self.allowed_hosts:
                tmp_net = f"devai_{uuid4().hex}"
                try:
                    subprocess.run(["docker", "network", "create", tmp_net], check=True)
                    ipt = shutil.which("iptables")
                    if ipt:
                        chain = f"DEVAI_{uuid4().hex}"
                        subprocess.run([ipt, "-N", chain], check=True)
                        for host in self.allowed_hosts:
                            subprocess.run([ipt, "-A", chain, "-d", host, "-j", "ACCEPT"], check=True)
                        subprocess.run([ipt, "-A", chain, "-j", "DROP"], check=True)
                        subprocess.run([ipt, "-I", "DOCKER-USER", "-j", chain], check=True)
                except Exception as e:  # pragma: no cover - network creation may fail
                    logger.warning("Falha ao criar rede temporária", error=str(e))
                    tmp_net = None
                    chain = None
            docker_cmd = [
                "docker",
                "run",
                "--rm",
                "-v",
                f"{os.getcwd()}:/app",
                "--workdir",
                "/app",
                "--network",
                tmp_net or net,
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
            if self.enabled and 'tmp_net' in locals() and tmp_net:
                try:
                    subprocess.run(["docker", "network", "rm", tmp_net], check=False)
                except Exception:
                    pass
            if self.enabled and 'chain' in locals() and chain:
                ipt = shutil.which("iptables")
                if ipt:
                    try:
                        subprocess.run([ipt, "-D", "DOCKER-USER", "-j", chain], check=False)
                        subprocess.run([ipt, "-F", chain], check=False)
                        subprocess.run([ipt, "-X", chain], check=False)
                    except Exception:
                        pass

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

