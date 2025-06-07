import subprocess
from typing import List


class Sandbox:
    """Isolated execution environment using containers."""

    def __init__(self, image: str = "python:3.10-slim"):
        self.image = image

    def run(self, command: List[str], timeout: int = 30) -> str:
        """Run a command inside a container.

        TODO: integrate with Docker/Podman for real isolation.
        """
        raise NotImplementedError("Sandbox execution not implemented yet")
