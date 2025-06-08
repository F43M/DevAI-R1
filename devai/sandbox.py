import subprocess
from typing import List


class Sandbox:
    """Isolated execution environment using containers."""

    def __init__(self, image: str = "python:3.10-slim", cpus: str = "1", memory: str = "512m"):
        self.image = image
        self.cpus = cpus
        self.memory = memory
        self._processes: List[subprocess.Popen] = []

    def run(self, command: List[str], timeout: int = 30) -> str:
        """Run a command inside a container using Docker."""
        docker_cmd = [
            "docker",
            "run",
            "--rm",
            "--cpus",
            self.cpus,
            "--memory",
            self.memory,
            self.image,
            *command,
        ]
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

    def shutdown(self) -> None:
        for p in list(self._processes):
            p.kill()
            self._processes.remove(p)

    def __enter__(self) -> "Sandbox":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.shutdown()
