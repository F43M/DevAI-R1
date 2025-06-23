from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import List, Set, Dict


@dataclass
class CodeContext:
    """Container for incremental code generation context."""

    history: List[str] = field(default_factory=list)
    imports: Set[str] = field(default_factory=set)
    functions: Set[str] = field(default_factory=set)
    variables: Set[str] = field(default_factory=set)

    def update(self, text: str) -> None:
        """Append ``text`` and analyze its structure using ``ast``."""

        self.history.append(text)
        try:
            tree = ast.parse(self.text())
        except SyntaxError:
            return

        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in node.names:
                    self.imports.add(alias.asname or alias.name.split(".")[0])
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self.functions.add(node.name)
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    self._collect_names(target)
            elif isinstance(node, ast.AnnAssign):
                self._collect_names(node.target)

    def _collect_names(self, node: ast.AST) -> None:
        if isinstance(node, ast.Name):
            self.variables.add(node.id)
        elif isinstance(node, (ast.Tuple, ast.List)):
            for elt in node.elts:
                self._collect_names(elt)

    def append(self, text: str) -> None:
        """Backward compatible wrapper for :meth:`update`."""

        self.update(text)

    def text(self) -> str:
        """Return full context text."""

        return "\n".join(self.history)

    def get_summary(self) -> Dict[str, List[str]]:
        """Return a summary of captured imports, functions and variables."""

        return {
            "imports": sorted(self.imports),
            "functions": sorted(self.functions),
            "variables": sorted(self.variables),
        }
