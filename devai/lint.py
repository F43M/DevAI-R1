from pathlib import Path
from typing import Dict, List
from .config import logger

"""Very small linting helpers used by some tasks."""


def simple_lint_file(path: Path) -> List[str]:
    """Return a list of TODO comments found in ``path``."""
    issues = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if "TODO" in line:
                issues.append(f"TODO encontrado na linha {i}")
    logger.info("Lint executado", file=str(path), issues=len(issues))
    return issues


class Linter:
    """Run ``simple_lint_file`` across a project tree."""

    def __init__(self, root: str):
        """Store the root path for later linting."""
        self.root = Path(root)

    def lint_all(self) -> Dict[str, List[str]]:
        """Lint every Python file under ``self.root``."""
        results = {}
        for file in self.root.rglob("*.py"):
            issues = simple_lint_file(file)
            if issues:
                results[str(file)] = issues
        return results
