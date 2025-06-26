"""Utilities for parsing code and calculating complexity metrics."""

from __future__ import annotations

import ast
from typing import Any, Dict

try:
    from tree_sitter import Parser  # type: ignore
    from tree_sitter_languages import get_language  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    Parser = None
    get_language = None


def parse_code(code: str, language: str):
    """Parse ``code`` using ``language`` rules.

    Parameters
    ----------
    code: str
        Source code.
    language: str
        Programming language name.

    Returns
    -------
    Any
        Parsed AST object or ``None`` if parsing fails.
    """
    lang = language.lower()
    if lang in {"python", "py"}:
        try:
            return ast.parse(code)
        except SyntaxError:
            return None
    if Parser and get_language:
        try:
            ts_lang = get_language(lang)
            parser = Parser()
            parser.set_language(ts_lang)
            tree = parser.parse(code.encode("utf-8"))
            return tree
        except Exception:  # pragma: no cover - tree-sitter errors
            return None
    return None


def _cyclomatic_complexity_py(func: ast.AST) -> int:
    """Compute cyclomatic complexity for a Python function node."""
    complexity = 1
    for node in ast.walk(func):
        if isinstance(
            node,
            (
                ast.If,
                ast.For,
                ast.While,
                ast.And,
                ast.Or,
                ast.ExceptHandler,
                ast.With,
                ast.Try,
                ast.BoolOp,
                ast.IfExp,
                ast.comprehension,
                ast.Assert,
            ),
        ):
            complexity += 1
    return complexity


def get_functions_complexity(code: str, language: str = "python") -> Dict[str, int]:
    """Return the complexity of each function defined in ``code``."""
    if language.lower() not in {"python", "py"}:
        return {}

    tree = parse_code(code, language)
    if tree is None:
        return {}

    funcs = [
        n
        for n in ast.walk(tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    return {func.name: _cyclomatic_complexity_py(func) for func in funcs}


__all__ = ["parse_code", "get_functions_complexity"]
