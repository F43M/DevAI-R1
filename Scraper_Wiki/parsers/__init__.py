"""Simple parser wrappers and grammars."""

from __future__ import annotations

from .base import BaseParser
from .python_parser import PythonParser
from .javascript_parser import JavaScriptParser
from .java_parser import JavaParser


_PARSERS = {
    "python": PythonParser,
    "py": PythonParser,
    "javascript": JavaScriptParser,
    "js": JavaScriptParser,
    "java": JavaParser,
}


def get_parser(language: str) -> BaseParser | None:
    """Return parser instance for ``language`` if available."""
    lang = language.lower()
    cls = _PARSERS.get(lang)
    return cls() if cls else None


__all__ = ["BaseParser", "PythonParser", "JavaScriptParser", "JavaParser", "get_parser"]
