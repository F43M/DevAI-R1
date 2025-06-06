from pathlib import Path
from typing import Dict, List
from .config import logger

def simple_lint_file(path: Path) -> List[str]:
    issues = []
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f, 1):
            if 'TODO' in line:
                issues.append(f'TODO encontrado na linha {i}')
    logger.info('Lint executado', file=str(path), issues=len(issues))
    return issues

class Linter:
    def __init__(self, root: str):
        self.root = Path(root)

    def lint_all(self) -> Dict[str, List[str]]:
        results = {}
        for file in self.root.rglob('*.py'):
            issues = simple_lint_file(file)
            if issues:
                results[str(file)] = issues
        return results
