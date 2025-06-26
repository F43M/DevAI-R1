from __future__ import annotations

import ast

from .base import BaseParser


class PythonParser(BaseParser):
    """Parser for Python code using :mod:`ast`."""

    def parse(self, code: str) -> bool:
        try:
            ast.parse(code)
            return True
        except SyntaxError:
            return False

    def extract_context(self, code: str) -> str:
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return ""
        func = next(
            (
                n
                for n in tree.body
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            ),
            None,
        )
        if func:
            ds = ast.get_docstring(func)
            if ds:
                return ds.strip().splitlines()[0]
        return ""
