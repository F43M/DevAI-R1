"""Utilities for analyzing source code snippets."""

from __future__ import annotations

import ast
import subprocess
import sys
from typing import Any, Dict, Iterable, List, Tuple

try:
    from tree_sitter import Parser  # type: ignore
    from tree_sitter_languages import get_language  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    Parser = None
    get_language = None

try:
    from pygments import lex
    from pygments.lexers import get_lexer_by_name, guess_lexer
except Exception:  # pragma: no cover - optional dependency
    lex = None
    get_lexer_by_name = None
    guess_lexer = None


def _get_lexer(code: str, language: str | None = None):
    lang = language or ""
    if get_lexer_by_name:
        try:
            if lang:
                return get_lexer_by_name(lang)
        except Exception:
            pass
    if guess_lexer:
        try:
            return guess_lexer(code)
        except Exception:
            return None
    return None


def extract_ast(code: str, language: str | None = None) -> Any:
    """Return an AST-like representation of ``code``.

    The function attempts to use Tree-sitter if available, otherwise
    falls back to Python's :mod:`ast` or Pygments tokens.
    """

    lang = (language or "python").lower()
    if Parser and get_language:
        try:
            ts_lang = get_language(lang)
            parser = Parser()
            parser.set_language(ts_lang)
            return parser.parse(code.encode("utf-8"))
        except Exception:
            pass
    if lang in {"python", "py"}:
        try:
            return ast.parse(code)
        except Exception:
            return None
    lexer = _get_lexer(code, lang)
    if lexer and lex:
        try:
            return list(lex(code, lexer))
        except Exception:
            return None
    return None


def extract_metadata(code: str, language: str | None = None) -> Dict[str, Any]:
    """Return simple metadata about ``code``.

    Currently counts lines and tokens using Pygments when available.
    """

    lexer = _get_lexer(code, language)
    tokens: Iterable[Any] = []
    if lexer and lex:
        try:
            tokens = list(lex(code, lexer))
        except Exception:
            tokens = []
    return {"lines": len(code.splitlines()), "tokens": len(list(tokens))}


def check_execution(code: str, language: str = "python") -> Tuple[bool, str]:
    """Compile and execute ``code`` returning success status and output."""

    if language not in {"python", "py"}:
        return False, "unsupported language"
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.returncode == 0, (result.stdout + result.stderr).strip()
    except Exception as e:  # pragma: no cover - subprocess errors
        return False, str(e)


def validate_snippet(record: Dict[str, Any]) -> Dict[str, Any]:
    """Check compilation and execution of ``record['content']``.

    Adds ``exec_ok`` and ``exec_output`` fields to the record.
    """

    code = record.get("content", "")
    lang = record.get("metadata", {}).get("code_language", "python")
    ok, output = check_execution(code, lang)
    record["exec_ok"] = ok
    record["exec_output"] = output
    return record


def generate_io_pairs(items: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Convert competition items to input/output pairs."""

    pairs = []
    for it in items:
        inp = it.get("problem", "")
        out = it.get("solution", "")
        if inp and out:
            pairs.append({"input": inp, "output": out})
    return pairs


__all__ = [
    "extract_ast",
    "extract_metadata",
    "check_execution",
    "validate_snippet",
    "generate_io_pairs",
]
