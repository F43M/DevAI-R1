"""Utility functions for validating and repairing generated Python code."""

from __future__ import annotations

import ast

import autopep8

_BLOCK_KEYWORDS = (
    "def ",
    "class ",
    "if ",
    "elif ",
    "else",
    "for ",
    "while ",
    "try",
    "except",
    "finally",
)


def is_valid_python(code: str) -> bool:
    """Return ``True`` if *code* can be parsed as valid Python."""

    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def _add_missing_colons(code: str) -> str:
    """Add colons to block statements that are missing them."""

    lines = code.splitlines()
    for i, line in enumerate(lines):
        stripped = line.strip()
        if any(
            stripped.startswith(k) for k in _BLOCK_KEYWORDS
        ) and not stripped.endswith(":"):
            lines[i] = line.rstrip() + ":"
    return "\n".join(lines)


def fix_code(code: str) -> str:
    """Return ``code`` formatted with autopep8 and basic syntax fixes."""

    formatted = code
    if not is_valid_python(formatted):
        formatted = _add_missing_colons(formatted)
    try:
        formatted = autopep8.fix_code(formatted)
    except Exception:
        pass
    return formatted if is_valid_python(formatted) else code
