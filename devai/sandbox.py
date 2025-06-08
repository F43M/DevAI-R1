import subprocess
from typing import List


class Sandbox:
    """Isolated execution environment using containers."""

    def __init__(self, image: str = "python:3.10-slim"):
        self.image = image

    def run(self, command: List[str], timeout: int = 30) -> str:
        """Run a command inside a container using Docker.

        Parameters
        ----------
        command:
            Command and arguments to execute inside the container.
        timeout:
            Maximum time in seconds to allow the command to run.

        Returns
        -------
        str
            Captured standard output from the command.
        """
        docker_cmd = ["docker", "run", "--rm", self.image, *command]
        try:
            proc = subprocess.run(
                docker_cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return proc.stdout
        except subprocess.TimeoutExpired as e:
            raise TimeoutError(f"Command timed out after {timeout}s") from e
