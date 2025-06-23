from __future__ import annotations

import ast
from typing import Any

import black


def is_valid_python(code: str) -> bool:
    """Check if ``code`` compiles without SyntaxError."""
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def fix_code(code: str) -> str:
    """Attempt to format ``code`` using black, returning original on failure."""
    try:
        return black.format_str(code, mode=black.FileMode())
    except Exception:
        return code
