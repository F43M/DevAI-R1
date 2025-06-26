from __future__ import annotations


class BaseParser:
    """Simple interface for language parsers."""

    def parse(self, code: str) -> bool:
        """Return True if ``code`` is syntactically valid."""
        raise NotImplementedError

    def extract_context(self, code: str) -> str:
        """Return relevant context string from ``code`` (e.g., docstring)."""
        return ""
